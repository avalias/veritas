//! Integer GEMV on the GPU (wgpu → Metal on this machine).
//!
//! The committed semantics define a dot product as 64-lane i32 partials
//! summed in i64 (kernels::dot_w8_x16_scalar). The shader reproduces the
//! SAME structure: each row's sixteen 64-lane partials are computed in i32
//! on the GPU and the i64 reduction + round-half-even happens on the host.
//! Integer arithmetic is associativity-exact, so CPU/GPU equality is a
//! THEOREM checked by tests, not a tolerance.

use wgpu::util::DeviceExt;

const SHADER: &str = r#"
struct Params { rows: u32, cols: u32 }
@group(0) @binding(0) var<storage, read> w8: array<u32>;   // 4 i8 per word
@group(0) @binding(1) var<storage, read> x16: array<u32>;  // 2 i16 per word
@group(0) @binding(2) var<storage, read_write> partials: array<i32>;
@group(0) @binding(3) var<uniform> p: Params;

fn sext8(v: u32) -> i32 {
    let b = i32(v & 0xffu);
    return select(b, b - 256, b > 127);
}
fn sext16(v: u32) -> i32 {
    let h = i32(v & 0xffffu);
    return select(h, h - 65536, h > 32767);
}

// One thread = one (row, 64-lane chunk) partial — mirrors the scalar
// definition exactly. partials[row*chunks + chunk].
@compute @workgroup_size(64)
fn gemv(@builtin(global_invocation_id) gid: vec3<u32>) {
    let chunks = p.cols / 64u;
    let total = p.rows * chunks;
    let idx = gid.x;
    if (idx >= total) { return; }
    let row = idx / chunks;
    let chunk = idx % chunks;
    let wbase = (row * p.cols + chunk * 64u) / 4u; // u32 words
    let xbase = (chunk * 64u) / 2u;
    var acc: i32 = 0;
    for (var k = 0u; k < 16u; k = k + 1u) {        // 16 words = 64 i8 lanes
        let wword = w8[wbase + k];
        let x0 = x16[xbase + 2u * k];
        let x1 = x16[xbase + 2u * k + 1u];
        acc = acc + sext8(wword)        * sext16(x0);
        acc = acc + sext8(wword >> 8u)  * sext16(x0 >> 16u);
        acc = acc + sext8(wword >> 16u) * sext16(x1);
        acc = acc + sext8(wword >> 24u) * sext16(x1 >> 16u);
    }
    partials[idx] = acc;
}
"#;

pub struct GpuGemv {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    pub adapter_name: String,
}

impl GpuGemv {
    pub fn new() -> Option<Self> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .ok()?;
        let name = adapter.get_info().name.clone();
        // The LM-head weight buffer is ~148 MB — request the adapter's real
        // limits instead of wgpu's conservative 128 MB default.
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .ok()?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gemv"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gemv"),
            layout: None,
            module: &module,
            entry_point: Some("gemv"),
            compilation_options: Default::default(),
            cache: None,
        });
        Some(Self { device, queue, pipeline, adapter_name: name })
    }

    /// Upload weights ONCE (they live in GPU memory across the whole run —
    /// exactly how a real deployment works; only activations move per call).
    pub fn upload_weights(&self, w: &[u8]) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("resident weights"),
            contents: w,
            usage: wgpu::BufferUsages::STORAGE,
        })
    }

    /// rows×cols GEMV with resident weights: only x (2·cols bytes) crosses
    /// to the GPU; i32 partials come back (committed structure).
    pub fn dots_resident(&self, wbuf: &wgpu::Buffer, x: &[u8], rows: usize, cols: usize) -> Vec<i64> {
        assert!(cols.is_multiple_of(64) && x.len() == 2 * cols);
        self.dots_inner(wbuf, x, rows, cols)
    }

    /// rows×cols GEMV: w (i8 bytes), x (i16 LE bytes) → i64 dots with the
    /// committed 64-lane-partial structure. cols must be a multiple of 64.
    pub fn dots(&self, w: &[u8], x: &[u8], rows: usize, cols: usize) -> Vec<i64> {
        assert!(cols.is_multiple_of(64) && w.len() == rows * cols && x.len() == 2 * cols);
        let wbuf = self.upload_weights(w);
        self.dots_inner(&wbuf, x, rows, cols)
    }

    fn dots_inner(&self, wbuf: &wgpu::Buffer, x: &[u8], rows: usize, cols: usize) -> Vec<i64> {
        let chunks = cols / 64;
        let xbuf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: x,
            usage: wgpu::BufferUsages::STORAGE,
        });

        let psize = (rows * chunks * 4) as u64;
        let pbuf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: psize,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let params_bytes: Vec<u8> =
            [(rows as u32).to_le_bytes(), (cols as u32).to_le_bytes()].concat();
        let ubuf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: &params_bytes,
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let layout = self.pipeline.get_bind_group_layout(0);
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wbuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: xbuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: pbuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: ubuf.as_entire_binding() },
            ],
        });
        let read = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: psize,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            let total = (rows * chunks) as u32;
            pass.dispatch_workgroups(total.div_ceil(64), 1, 1);
        }
        enc.copy_buffer_to_buffer(&pbuf, 0, &read, 0, psize);
        self.queue.submit([enc.finish()]);
        let slice = read.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        self.device.poll(wgpu::PollType::Wait).ok();
        rx.recv().unwrap().unwrap();
        let bytes = slice.get_mapped_range();
        // Host-side i64 reduction over the i32 partials — identical to the
        // scalar definition's structure.
        let mut out = vec![0i64; rows];
        for (r, slot) in out.iter_mut().enumerate() {
            let mut acc = 0i64;
            for c in 0..chunks {
                let off = (r * chunks + c) * 4;
                acc += i32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as i64;
            }
            *slot = acc;
        }
        out
    }
}

