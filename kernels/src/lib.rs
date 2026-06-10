//! Integer performance kernels (SPEC §9.1 checkpoint-mode freedoms only:
//! lane reordering inside dots + parallelism across independent output
//! cells). Everything here is bit-exact by integer algebra and verified by
//! equality tests against scalar references — `unsafe` is for speed, never
//! for semantics.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use vm::exec::rnd;

// ---------------------------------------------------------------------------
// Persistent worker pool (replaces ~200 thread spawns per token)
// ---------------------------------------------------------------------------

/// Type-erased job. SAFETY: `Pool::run` keeps the referents alive until
/// every worker acks, so the 'static erasure never outlives reality.
#[derive(Clone, Copy)]
struct Job {
    f: *const (dyn Fn(usize, usize) + Sync + 'static),
    counter: *const AtomicUsize,
    total: usize,
    chunk: usize,
}
unsafe impl Send for Job {}

/// Work-stealing fork-join pool. `run` blocks until all chunks complete;
/// the borrowed closure never outlives the call (acks are counted before
/// returning), which is what makes the lifetime erasure sound.
pub struct Pool {
    senders: Vec<mpsc::Sender<Job>>,
    acks: mpsc::Receiver<()>,
    ack_tx: mpsc::Sender<()>,
    pub threads: usize,
}

unsafe impl Send for Pool {}
unsafe impl Sync for Pool {}

impl Pool {
    pub fn new(threads: usize) -> Self {
        let threads = threads.max(1);
        let (ack_tx, acks) = mpsc::channel();
        let mut senders = Vec::new();
        for _ in 0..threads - 1 {
            let (tx, rx) = mpsc::channel::<Job>();
            let ack = ack_tx.clone();
            std::thread::spawn(move || {
                while let Ok(job) = rx.recv() {
                    // SAFETY: see Job — referents outlive the job.
                    let f = unsafe { &*job.f };
                    let counter = unsafe { &*job.counter };
                    loop {
                        let start = counter.fetch_add(job.chunk, Ordering::Relaxed);
                        if start >= job.total {
                            break;
                        }
                        f(start, (start + job.chunk).min(job.total));
                    }
                    let _ = ack.send(());
                }
            });
            senders.push(tx);
        }
        Self { senders, acks, ack_tx, threads }
    }

    /// Run `f` over [0, total) in chunks, on all threads (caller included).
    pub fn run(&self, total: usize, chunk: usize, f: &(dyn Fn(usize, usize) + Sync)) {
        if total == 0 {
            return;
        }
        let counter = AtomicUsize::new(0);
        // SAFETY: lifetime erasure; `run` barriers on acks before returning.
        let f_static: *const (dyn Fn(usize, usize) + Sync + 'static) =
            unsafe { std::mem::transmute(f as *const (dyn Fn(usize, usize) + Sync)) };
        let job = Job { f: f_static, counter: &counter, total, chunk: chunk.max(1) };
        for tx in &self.senders {
            tx.send(job).expect("worker alive");
        }
        // Caller participates.
        loop {
            let start = counter.fetch_add(chunk.max(1), Ordering::Relaxed);
            if start >= total {
                break;
            }
            f(start, (start + chunk.max(1)).min(total));
        }
        // Barrier: closure must stay alive until every worker is done.
        for _ in 0..self.senders.len() {
            self.acks.recv().expect("ack");
        }
        let _ = &self.ack_tx;
    }
}

// ---------------------------------------------------------------------------
// GEMV: i8 weights × i16 activations → requantized i64 rows
// ---------------------------------------------------------------------------

/// Scalar reference — the semantic definition (and non-aarch64 fallback).
/// 64-lane i32 partials (≤ 64·127·32767 < 2^29), summed in i64.
pub fn dot_w8_x16_scalar(w: &[i8], x: &[i16]) -> i64 {
    let mut acc = 0i64;
    for (wc, xc) in w.chunks(64).zip(x.chunks(64)) {
        let mut part = 0i32;
        for (a, b) in wc.iter().zip(xc) {
            part += (*a as i32) * (*b as i32);
        }
        acc += part as i64;
    }
    acc
}

/// NEON dot: widen i8→i16, pairwise smlal into 4×i32x4 accumulators,
/// drain to i64 every 256 columns (8-lane i32 accumulates 32 iterations of
/// ≤2^22 products: < 2^27 per drain — no overflow).
#[cfg(target_arch = "aarch64")]
pub fn dot_w8_x16(w: &[i8], x: &[i16]) -> i64 {
    use std::arch::aarch64::*;
    let n = w.len();
    debug_assert_eq!(n, x.len());
    if !n.is_multiple_of(16) {
        return dot_w8_x16_scalar(w, x);
    }
    let mut acc = 0i64;
    let mut i = 0;
    unsafe {
        while i < n {
            let block_end = (i + 256).min(n);
            let mut a0 = vdupq_n_s32(0);
            let mut a1 = vdupq_n_s32(0);
            let mut a2 = vdupq_n_s32(0);
            let mut a3 = vdupq_n_s32(0);
            while i < block_end {
                let wv = vld1q_s8(w.as_ptr().add(i));
                let wl = vmovl_s8(vget_low_s8(wv));
                let wh = vmovl_s8(vget_high_s8(wv));
                let x0 = vld1q_s16(x.as_ptr().add(i));
                let x1 = vld1q_s16(x.as_ptr().add(i + 8));
                a0 = vmlal_s16(a0, vget_low_s16(wl), vget_low_s16(x0));
                a1 = vmlal_high_s16(a1, wl, x0);
                a2 = vmlal_s16(a2, vget_low_s16(wh), vget_low_s16(x1));
                a3 = vmlal_high_s16(a3, wh, x1);
                i += 16;
            }
            acc += vaddlvq_s32(a0) + vaddlvq_s32(a1) + vaddlvq_s32(a2) + vaddlvq_s32(a3);
        }
    }
    acc
}

#[cfg(not(target_arch = "aarch64"))]
pub fn dot_w8_x16(w: &[i8], x: &[i16]) -> i64 {
    dot_w8_x16_scalar(w, x)
}

/// Reinterpret LE byte slices (callers in float-free crates can't cast).
/// SAFETY: alignment asserted; little-endian targets only (we assert in
/// the workspace's golden tests anyway — LE is SPEC §2.1 normative).
pub fn bytes_as_i8(b: &[u8]) -> &[i8] {
    unsafe { std::slice::from_raw_parts(b.as_ptr() as *const i8, b.len()) }
}

pub fn bytes_as_i16(b: &[u8]) -> &[i16] {
    assert!((b.as_ptr() as usize).is_multiple_of(2) && b.len().is_multiple_of(2));
    unsafe { std::slice::from_raw_parts(b.as_ptr() as *const i16, b.len() / 2) }
}

/// Byte-slice convenience wrappers (weights = i8 bytes, x = i16 LE bytes).
#[allow(clippy::too_many_arguments)]
pub fn gemv_bytes(
    pool: &Pool,
    w: &[u8],
    x: &[u8],
    rows: usize,
    cols: usize,
    m: &[i32],
    shift: u8,
    out: &mut [i64],
) {
    gemv_w8_x16(pool, bytes_as_i8(w), bytes_as_i16(x), rows, cols, m, shift, out)
}

pub fn gemv_logits_bytes(pool: &Pool, w: &[u8], x: &[u8], rows: usize, cols: usize, out: &mut [i64]) {
    gemv_logits(pool, bytes_as_i8(w), bytes_as_i16(x), rows, cols, out)
}

/// Row-parallel projection: out[r] = rnd(dot(w_row_r, x) · m[r], shift).
#[allow(clippy::too_many_arguments)]
pub fn gemv_w8_x16(
    pool: &Pool,
    w: &[i8],
    x: &[i16],
    rows: usize,
    cols: usize,
    m: &[i32],
    shift: u8,
    out: &mut [i64],
) {
    debug_assert_eq!(out.len(), rows);
    let out_addr = SendPtr(out.as_mut_ptr());
    pool.run(rows, 16, &move |start, end| {
        let out_ptr = out_addr;
        for r in start..end {
            let d = dot_w8_x16(&w[r * cols..(r + 1) * cols], x);
            // SAFETY: rows are disjoint across chunks.
            unsafe { *out_ptr.0.add(r) = rnd(d.wrapping_mul(m[r] as i64), shift) };
        }
    });
}

/// Raw-dot variant (no requant) — LM head logits: out[r] = rnd(dot, 11).
pub fn gemv_logits(pool: &Pool, w: &[i8], x: &[i16], rows: usize, cols: usize, out: &mut [i64]) {
    let out_addr = SendPtr(out.as_mut_ptr());
    pool.run(rows, 64, &move |start, end| {
        let out_ptr = out_addr;
        for r in start..end {
            let d = dot_w8_x16(&w[r * cols..(r + 1) * cols], x);
            unsafe { *out_ptr.0.add(r) = rnd(d, 11) };
        }
    });
}

/// Parallel map over disjoint equal-size chunks of an i64 buffer —
/// f(chunk_index, chunk). The unsafe split lives HERE, not in callers.
pub fn run_disjoint_i64(
    pool: &Pool,
    out: &mut [i64],
    chunk_len: usize,
    f: &(dyn Fn(usize, &mut [i64]) + Sync),
) {
    assert!(chunk_len > 0 && out.len().is_multiple_of(chunk_len));
    let n = out.len() / chunk_len;
    let base = SendPtr(out.as_mut_ptr());
    pool.run(n, 1, &move |start, end| {
        let b = base;
        for i in start..end {
            // SAFETY: disjoint chunks by construction.
            let piece = unsafe { std::slice::from_raw_parts_mut(b.0.add(i * chunk_len), chunk_len) };
            f(i, piece);
        }
    });
}

#[derive(Clone, Copy)]
struct SendPtr(*mut i64);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

#[cfg(test)]
mod tests {
    use super::*;

    fn xorshift(s: &mut u64) -> u64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        *s
    }

    /// THE kernel test: NEON must equal scalar bit-for-bit, always.
    #[test]
    fn neon_equals_scalar_exactly() {
        let mut s = 0x1234_5678_9abc_def1u64;
        for trial in 0..300 {
            let n = [64usize, 128, 256, 1024, 3072, 4096][trial % 6];
            let w: Vec<i8> = (0..n).map(|_| xorshift(&mut s) as i8).collect();
            let x: Vec<i16> = (0..n).map(|_| xorshift(&mut s) as i16).collect();
            assert_eq!(dot_w8_x16(&w, &x), dot_w8_x16_scalar(&w, &x), "n={n}");
        }
        // Extremes: all-max lanes at the longest row.
        let w = vec![i8::MIN; 4096];
        let x = vec![i16::MIN; 4096];
        assert_eq!(dot_w8_x16(&w, &x), dot_w8_x16_scalar(&w, &x));
    }

    #[test]
    fn pool_gemv_matches_single_thread() {
        let pool = Pool::new(8);
        let mut s = 7u64;
        let (rows, cols) = (333, 1024);
        let w: Vec<i8> = (0..rows * cols).map(|_| xorshift(&mut s) as i8).collect();
        let x: Vec<i16> = (0..cols).map(|_| xorshift(&mut s) as i16).collect();
        let m: Vec<i32> = (0..rows).map(|_| (xorshift(&mut s) % 100_000) as i32 + 1).collect();
        let mut out = vec![0i64; rows];
        gemv_w8_x16(&pool, &w, &x, rows, cols, &m, 20, &mut out);
        for r in 0..rows {
            let d = dot_w8_x16_scalar(&w[r * cols..(r + 1) * cols], &x);
            assert_eq!(out[r], rnd(d.wrapping_mul(m[r] as i64), 20), "row {r}");
        }
    }
}
