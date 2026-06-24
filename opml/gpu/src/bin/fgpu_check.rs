//! FW-6 GPU determinism: the committed-float LM-head GEMV — all 151,936
//! logits over the REAL Qwen3-0.6B bf16 tied-embedding matrix — on the GPU
//! vs the CPU kernel, compared BIT FOR BIT (no tolerance).
//!
//!   cargo run -p gpu --release --bin fgpu_check
//!
//! Two GPU paths, because the result is a finding in itself:
//!  1. wgpu/WGSL — wgpu-hal compiles Metal shaders with the default
//!     fastMathEnabled=YES and exposes no way to disable it, so the Metal
//!     compiler may reassociate float adds: MEASURED 1-ulp divergence on
//!     ~90% of rows. (Printed, not asserted — it documents the gap.)
//!  2. direct Metal (metal-rs) with set_fast_math_enabled(false) — the
//!     SAME kernel, reassociation forbidden: asserted BIT-IDENTICAL.
#![allow(clippy::float_arithmetic)] // FW-6 committed-float host code

use kernels::fkernels::fgemv;
use kernels::Pool;
use qwen::config::QwenConfig;
use qwen::tensors::SafeTensors;
use std::time::Instant;

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/qwen/artifacts");

fn diff_count(a: &[f32], b: &[f32]) -> (usize, Option<usize>) {
    let mut n = 0;
    let mut first = None;
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        if x.to_bits() != y.to_bits() {
            n += 1;
            if first.is_none() {
                first = Some(i);
            }
        }
    }
    (n, first)
}

fn main() {
    let cfg = QwenConfig::load(&format!("{DIR}/config.json"));
    println!("· loading the real bf16 LM-head matrix ({}×{})…", cfg.vocab_size, cfg.hidden_size);
    let st = SafeTensors::load(&format!("{DIR}/model.safetensors"));
    let w: Vec<u16> = st.bf16_bits("model.embed_tokens.weight");
    drop(st);
    let (rows, cols) = (cfg.vocab_size, cfg.hidden_size);

    // Deterministic, realistically-scaled activation vector.
    let mut s = 0x5EEDu64;
    let mut rng = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let x: Vec<f32> = (0..cols).map(|_| (rng() % 4000) as f32 / 200.0 - 10.0).collect();

    // CPU committed kernel (== scalar definition; tested bitwise).
    let pool = Pool::new(std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8));
    let mut cpu_out = vec![0f32; rows];
    let t0 = Instant::now();
    fgemv(&pool, &w, &x, rows, cols, &mut cpu_out);
    // Only read in the macos GPU-vs-CPU timing print below; unused on other targets.
    #[cfg_attr(not(target_os = "macos"), allow(unused_variables))]
    let cpu_us = t0.elapsed().as_micros();

    // Path 1: wgpu/WGSL (fast-math not controllable) — document.
    if let Some(g) = gpu::GpuFGemv::new() {
        let wbuf = g.upload_weights(&w);
        let out = g.fgemv(&wbuf, &x, rows, cols);
        let (n, first) = diff_count(&out, &cpu_out);
        println!("· wgpu/WGSL ({}): {} of {} logits differ from CPU", g.adapter_name, n, rows);
        if let Some(r) = first {
            println!(
                "  e.g. row {r}: gpu {:08x} vs cpu {:08x} (1-ulp fast-math reassociation)",
                out[r].to_bits(),
                cpu_out[r].to_bits()
            );
        }
        println!("  (wgpu-hal compiles MSL with fastMathEnabled=YES and has no off switch)");
    }

    // Path 2: direct Metal, fast-math OFF — assert.
    #[cfg(target_os = "macos")]
    {
        let Some(m) = gpu::MetalFGemv::new() else {
            println!("no Metal device");
            return;
        };
        let wbuf = m.upload_weights(&w);
        let t0 = Instant::now();
        let out = m.fgemv(&wbuf, &x, rows, cols);
        let gpu_us = t0.elapsed().as_micros();
        let (n, first) = diff_count(&out, &cpu_out);
        println!("· direct Metal, fastMathEnabled=false ({}): {} differences", m.name, n);
        if let Some(r) = first {
            println!("  row {r}: gpu {:08x} vs cpu {:08x}", out[r].to_bits(), cpu_out[r].to_bits());
        }
        println!("  gpu {gpu_us} µs vs cpu(pool) {cpu_us} µs (resident weights; informational)");
        assert_eq!(n, 0, "committed-float GPU must be bit-identical with fast-math off");
        println!(
            "✓ FW-6 ON GPU: all {} committed-float logits BIT-IDENTICAL — real Qwen bf16 head, Metal vs CPU",
            rows
        );
    }
}
