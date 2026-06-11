//! Thermal-robust A/B of the blocked-GEMV kernels. Single-run tok/s drifts
//! ±20% with CPU thermals; the ROBUST measure is the RATIO of two kernels
//! timed back-to-back in the same thermal state. Alternates A/B/A/B… and
//! reports median ns/GEMV and the speedup, so the comparison is invariant to
//! throttling.
//!
//!   cargo run -p benches --release --bin kernel_ab

use kernels::{gemv_blocked_bytes, gemv_blocked_dot_bytes, gemv_blocked_legacy_bytes, Pool};
use std::time::Instant;

fn main() {
    // Representative Qwen projection: gate/up are the widest (3072×1024).
    let (rows, cols) = (3072usize, 1024usize);
    let blocks = cols / 64;
    let mut s = 0x1234_9876u64;
    let mut rng = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let w: Vec<u8> = (0..rows * cols).map(|_| rng() as u8).collect();
    // i16 activations incl. saturated extremes (exercise the limb edge).
    let x: Vec<u8> = (0..cols)
        .flat_map(|_| {
            let v: i16 = match rng() % 8 {
                0 => 32767,
                1 => -32768,
                _ => rng() as i16,
            };
            v.to_le_bytes()
        })
        .collect();
    let m: Vec<i32> = (0..rows * blocks).map(|_| (rng() % (1 << 24)) as i32 + 1).collect();

    let pool = Pool::new(std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8));
    let mut out_a = vec![0i64; rows];
    let mut out_b = vec![0i64; rows];

    // Correctness gate: the two kernels MUST agree bit-for-bit.
    gemv_blocked_bytes(&pool, &w, &x, rows, cols, &m, 20, &mut out_a);
    gemv_blocked_dot_bytes(&pool, &w, &x, rows, cols, &m, 20, &mut out_b);
    assert_eq!(out_a, out_b, "sdot path must equal vmlal path bit-for-bit");

    let mut out_c = vec![0i64; rows];
    gemv_blocked_legacy_bytes(&pool, &w, &x, rows, cols, &m, 20, &mut out_c);
    assert_eq!(out_a, out_c, "legacy path must also agree bit-for-bit");

    let iters = 400usize;
    let inner = 8usize; // GEMVs per timed sample (amortize Instant)
    let mut tl = Vec::with_capacity(iters); // legacy dot-per-block
    let mut ta = Vec::with_capacity(iters); // fused block_partial (current)
    let mut tb = Vec::with_capacity(iters); // sdot two-limb
    let time = |f: &mut dyn FnMut()| {
        let t = Instant::now();
        for _ in 0..inner {
            f();
        }
        t.elapsed().as_nanos() / inner as u128
    };
    for _ in 0..iters {
        tl.push(time(&mut || gemv_blocked_legacy_bytes(&pool, &w, &x, rows, cols, &m, 20, &mut out_c)));
        ta.push(time(&mut || gemv_blocked_bytes(&pool, &w, &x, rows, cols, &m, 20, &mut out_a)));
        tb.push(time(&mut || gemv_blocked_dot_bytes(&pool, &w, &x, rows, cols, &m, 20, &mut out_b)));
    }
    tl.sort_unstable();
    ta.sort_unstable();
    tb.sort_unstable();
    let (ml, ma, mb) = (tl[iters / 2], ta[iters / 2], tb[iters / 2]);
    println!("GEMV {rows}×{cols} blocked, median of {iters} alternated samples:");
    println!("  legacy dot-per-block:   {ml:>6} ns/GEMV  (baseline)");
    println!(
        "  fused block_partial:    {ma:>6} ns/GEMV  ({}.{:02}× vs legacy)",
        ml / ma.max(1),
        (ml * 100 / ma.max(1)) % 100
    );
    println!(
        "  sdot two-limb (asm):    {mb:>6} ns/GEMV  ({}.{:02}× vs legacy)",
        ml / mb.max(1),
        (ml * 100 / mb.max(1)) % 100
    );
}
