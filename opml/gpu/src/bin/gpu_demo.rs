//! GPU integer GEMV demo: the LM head (the model's biggest operator,
//! 151,936 × 1024 ≈ 156 MMAC) computed on the actual GPU with COMMITTED
//! semantics — logits asserted bit-identical to the CPU kernels, then a
//! throughput number. Run: cargo run -p gpu --release --bin gpu_demo
use gpu::GpuGemv;
use std::time::Instant;

fn xorshift(s: &mut u64) -> u64 {
    *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; *s
}

fn main() {
    let Some(g) = GpuGemv::new() else {
        println!("no GPU adapter — skipping");
        return;
    };
    println!("GPU adapter: {}", g.adapter_name);
    let (rows, cols) = (151_936usize, 1024usize);
    let mut s = 0xC0FFEE;
    let w: Vec<u8> = (0..rows * cols).map(|_| xorshift(&mut s) as u8).collect();
    let x: Vec<u8> = (0..2 * cols).map(|_| xorshift(&mut s) as u8).collect();

    // Correctness: bit-equality with the CPU kernel (which equals scalar).
    let t0 = Instant::now();
    let gpu_dots = g.dots(&w, &x, rows, cols);
    let first_us = t0.elapsed().as_micros();
    let pool = kernels::Pool::new(10);
    let mut cpu = vec![0i64; rows];
    let t1 = Instant::now();
    kernels::gemv_logits_bytes(&pool, &w, &x, rows, cols, &mut cpu);
    let cpu_us = t1.elapsed().as_micros();
    let gpu_logits: Vec<i64> = gpu_dots.iter().map(|&d| vm::exec::rnd(d, 11)).collect();
    assert_eq!(gpu_logits, cpu, "GPU must equal CPU bit-for-bit");
    println!("✓ 151,936 logits BIT-IDENTICAL on GPU vs CPU (committed semantics)");

    let reps = 10u128;
    let t2 = Instant::now();
    for _ in 0..reps {
        let d = g.dots(&w, &x, rows, cols);
        std::hint::black_box(d[0]);
    }
    let per = t2.elapsed().as_micros() / reps;
    // The deployment-realistic number: weights RESIDENT on the GPU,
    // only the 2 KiB activation vector moves per token.
    let wbuf = g.upload_weights(&w);
    let t3 = Instant::now();
    for _ in 0..reps {
        let d = g.dots_resident(&wbuf, &x, rows, cols);
        std::hint::black_box(d[0]);
    }
    let per_res = t3.elapsed().as_micros() / reps;
    println!("GPU head GEMV: {per} µs/call incl. weight upload (first {first_us} µs)");
    println!("GPU head GEMV, weights RESIDENT: {per_res} µs/call");
    println!("CPU pool+NEON same op: {cpu_us} µs");
}
