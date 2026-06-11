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


// ---------------------------------------------------------------------------
// FW-6: committed FLOAT GEMV on the GPU
// ---------------------------------------------------------------------------

/// WGSL implementation of the committed float tree (kernels::fkernels):
/// bf16 weights widened by bit-shift (exact), 4×vec4<f32> accumulators with
/// component-wise fma over 16-lane strides, pinned combine (a0+a1)+(a2+a3),
/// pinned horizontal (x+y)+(z+w), sequential block chain. One thread per
/// row so the whole row's order is owned by one invocation.
///
/// Float addition is not associative, so this only matches the CPU if the
/// GPU compiler performs NO reassociation/contraction beyond the fma we
/// wrote — which WGSL semantics promise. fgpu_check tests that promise on
/// real hardware against all 151,936 LM-head rows.
const FSHADER: &str = r#"
struct Params { rows: u32, cols: u32 }
@group(0) @binding(0) var<storage, read> wbf: array<u32>;  // 2 bf16 / word
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

fn blo(w: u32) -> f32 { return bitcast<f32>((w & 0xffffu) << 16u); }
fn bhi(w: u32) -> f32 { return bitcast<f32>(w & 0xffff0000u); }

@compute @workgroup_size(64)
fn fgemv(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= p.rows) { return; }
    var acc: f32 = 0.0;
    var b: u32 = 0u;
    loop {
        if (b >= p.cols) { break; }
        var a0 = vec4<f32>(0.0);
        var a1 = vec4<f32>(0.0);
        var a2 = vec4<f32>(0.0);
        var a3 = vec4<f32>(0.0);
        for (var i = 0u; i < 4u; i = i + 1u) {
            let base = b + 16u * i;
            let wb = (row * p.cols + base) / 2u;
            let w0 = wbf[wb];      let w1 = wbf[wb + 1u];
            let w2 = wbf[wb + 2u]; let w3 = wbf[wb + 3u];
            let w4 = wbf[wb + 4u]; let w5 = wbf[wb + 5u];
            let w6 = wbf[wb + 6u]; let w7 = wbf[wb + 7u];
            let wv0 = vec4<f32>(blo(w0), bhi(w0), blo(w1), bhi(w1));
            let wv1 = vec4<f32>(blo(w2), bhi(w2), blo(w3), bhi(w3));
            let wv2 = vec4<f32>(blo(w4), bhi(w4), blo(w5), bhi(w5));
            let wv3 = vec4<f32>(blo(w6), bhi(w6), blo(w7), bhi(w7));
            let x0 = vec4<f32>(x[base], x[base + 1u], x[base + 2u], x[base + 3u]);
            let x1 = vec4<f32>(x[base + 4u], x[base + 5u], x[base + 6u], x[base + 7u]);
            let x2 = vec4<f32>(x[base + 8u], x[base + 9u], x[base + 10u], x[base + 11u]);
            let x3 = vec4<f32>(x[base + 12u], x[base + 13u], x[base + 14u], x[base + 15u]);
            a0 = fma(wv0, x0, a0);
            a1 = fma(wv1, x1, a1);
            a2 = fma(wv2, x2, a2);
            a3 = fma(wv3, x3, a3);
        }
        let s = (a0 + a1) + (a2 + a3);
        let pb = (s.x + s.y) + (s.z + s.w);
        acc = acc + pb;
        b = b + 64u;
    }
    out[row] = acc;
}
"#;

pub struct GpuFGemv {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    pub adapter_name: String,
}

impl GpuFGemv {
    pub fn new() -> Option<Self> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .ok()?;
        let name = adapter.get_info().name.clone();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .ok()?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fgemv"),
            source: wgpu::ShaderSource::Wgsl(FSHADER.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("fgemv"),
            layout: None,
            module: &module,
            entry_point: Some("fgemv"),
            compilation_options: Default::default(),
            cache: None,
        });
        Some(Self { device, queue, pipeline, adapter_name: name })
    }

    pub fn upload_weights(&self, w_bf16: &[u16]) -> wgpu::Buffer {
        let bytes: Vec<u8> = w_bf16.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bf16 weights"),
            contents: &bytes,
            usage: wgpu::BufferUsages::STORAGE,
        })
    }

    /// rows×cols committed-float GEMV with resident bf16 weights.
    pub fn fgemv(&self, wbuf: &wgpu::Buffer, x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        assert!(cols % 64 == 0 && x.len() == cols);
        let xbytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
        let xbuf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: &xbytes,
            usage: wgpu::BufferUsages::STORAGE,
        });
        let osize = (rows * 4) as u64;
        let obuf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: osize,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let params: Vec<u8> =
            [(rows as u32).to_le_bytes(), (cols as u32).to_le_bytes()].concat();
        let ubuf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: &params,
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let layout = self.pipeline.get_bind_group_layout(0);
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wbuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: xbuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: obuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: ubuf.as_entire_binding() },
            ],
        });
        let read = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: osize,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups((rows as u32).div_ceil(64), 1, 1);
        }
        enc.copy_buffer_to_buffer(&obuf, 0, &read, 0, osize);
        self.queue.submit([enc.finish()]);
        let slice = read.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        self.device.poll(wgpu::PollType::Wait).ok();
        rx.recv().unwrap().unwrap();
        let bytes = slice.get_mapped_range();
        let mut out = vec![0f32; rows];
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
        }
        out
    }
}

// ---------------------------------------------------------------------------
// FW-6 on DIRECT Metal: fastMathEnabled = false
// ---------------------------------------------------------------------------
//
// MEASURED: through wgpu the same canonical-tree shader diverges from the
// CPU by 1 ulp on ~90% of rows — wgpu-hal compiles MSL with Metal's
// default fastMathEnabled=YES (no API to disable), licensing the compiler
// to reassociate float adds. The committed tree therefore needs the one
// flag wgpu doesn't expose. This path compiles the SAME kernel with
// metal-rs and fast-math OFF.

/// MSL twin of kernels::fkernels::fdot_row — same lanes, same combine,
/// same block chain; `fma` is IEEE-fused; fast-math disabled at compile.
#[cfg(target_os = "macos")]
const FMSL: &str = r#"
#include <metal_stdlib>
using namespace metal;
struct Params { uint rows; uint cols; };

kernel void fgemv(device const ushort* wbf [[buffer(0)]],
                  device const float*  x   [[buffer(1)]],
                  device float*        outv[[buffer(2)]],
                  constant Params&     p   [[buffer(3)]],
                  uint row [[thread_position_in_grid]]) {
    if (row >= p.rows) { return; }
    float acc = 0.0f;
    for (uint b = 0u; b < p.cols; b += 64u) {
        float4 a0 = float4(0.0f), a1 = float4(0.0f), a2 = float4(0.0f), a3 = float4(0.0f);
        for (uint i = 0u; i < 4u; i++) {
            const uint base = b + 16u * i;
            device const ushort* wp = wbf + row * p.cols + base;
            device const float*  xp = x + base;
            float4 wv0 = float4(as_type<float>(uint(wp[0])  << 16), as_type<float>(uint(wp[1])  << 16),
                                as_type<float>(uint(wp[2])  << 16), as_type<float>(uint(wp[3])  << 16));
            float4 wv1 = float4(as_type<float>(uint(wp[4])  << 16), as_type<float>(uint(wp[5])  << 16),
                                as_type<float>(uint(wp[6])  << 16), as_type<float>(uint(wp[7])  << 16));
            float4 wv2 = float4(as_type<float>(uint(wp[8])  << 16), as_type<float>(uint(wp[9])  << 16),
                                as_type<float>(uint(wp[10]) << 16), as_type<float>(uint(wp[11]) << 16));
            float4 wv3 = float4(as_type<float>(uint(wp[12]) << 16), as_type<float>(uint(wp[13]) << 16),
                                as_type<float>(uint(wp[14]) << 16), as_type<float>(uint(wp[15]) << 16));
            float4 x0 = float4(xp[0],  xp[1],  xp[2],  xp[3]);
            float4 x1 = float4(xp[4],  xp[5],  xp[6],  xp[7]);
            float4 x2 = float4(xp[8],  xp[9],  xp[10], xp[11]);
            float4 x3 = float4(xp[12], xp[13], xp[14], xp[15]);
            a0 = fma(wv0, x0, a0);
            a1 = fma(wv1, x1, a1);
            a2 = fma(wv2, x2, a2);
            a3 = fma(wv3, x3, a3);
        }
        float4 s = (a0 + a1) + (a2 + a3);
        float pb = (s.x + s.y) + (s.z + s.w);
        acc = acc + pb;
    }
    outv[row] = acc;
}
"#;

#[cfg(target_os = "macos")]
pub struct MetalFGemv {
    device: metal::Device,
    queue: metal::CommandQueue,
    pipeline: metal::ComputePipelineState,
    pub name: String,
}

#[cfg(target_os = "macos")]
impl MetalFGemv {
    pub fn new() -> Option<Self> {
        let device = metal::Device::system_default()?;
        let name = device.name().to_string();
        let opts = metal::CompileOptions::new();
        // THE flag: forbid reassociation/contraction beyond written fma.
        opts.set_fast_math_enabled(false);
        let lib = device.new_library_with_source(FMSL, &opts).ok()?;
        let f = lib.get_function("fgemv", None).ok()?;
        let pipeline = device.new_compute_pipeline_state_with_function(&f).ok()?;
        let queue = device.new_command_queue();
        Some(Self { device, queue, pipeline, name })
    }

    pub fn upload_weights(&self, w_bf16: &[u16]) -> metal::Buffer {
        self.device.new_buffer_with_data(
            w_bf16.as_ptr() as *const core::ffi::c_void,
            (w_bf16.len() * 2) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        )
    }

    /// rows×cols committed-float GEMV; fast-math OFF.
    #[allow(unsafe_code)] // metal-rs buffer reads are raw pointers; bounds are ours
    pub fn fgemv(&self, wbuf: &metal::Buffer, x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        assert!(cols % 64 == 0 && x.len() == cols);
        let xbuf = self.device.new_buffer_with_data(
            x.as_ptr() as *const core::ffi::c_void,
            (x.len() * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let obuf = self.device.new_buffer(
            (rows * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let params: [u32; 2] = [rows as u32, cols as u32];
        let pbuf = self.device.new_buffer_with_data(
            params.as_ptr() as *const core::ffi::c_void,
            8,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let cb = self.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pipeline);
        enc.set_buffer(0, Some(wbuf), 0);
        enc.set_buffer(1, Some(&xbuf), 0);
        enc.set_buffer(2, Some(&obuf), 0);
        enc.set_buffer(3, Some(&pbuf), 0);
        enc.dispatch_threads(
            metal::MTLSize { width: rows as u64, height: 1, depth: 1 },
            metal::MTLSize { width: 64, height: 1, depth: 1 },
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let mut out = vec![0f32; rows];
        // SAFETY: obuf is rows*4 bytes of shared memory we own; read once
        // after wait_until_completed.
        unsafe {
            std::ptr::copy_nonoverlapping(obuf.contents() as *const f32, out.as_mut_ptr(), rows);
        }
        out
    }
}
