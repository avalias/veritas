//! Integer performance kernels (SPEC §9.1 checkpoint-mode freedoms only:
//! lane reordering inside dots + parallelism across independent output
//! cells). Everything here is bit-exact by integer algebra and verified by
//! equality tests against scalar references — `unsafe` is for speed, never
//! for semantics.
#![cfg_attr(feature = "nightly_dotprod", feature(stdarch_neon_dotprod))]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use vm::exec::rnd;

pub mod fkernels;

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

/// Scalar reference for ONE 64-element block partial: Σ w_j·x_j as i64
/// (fits i32; ≤ 64·127·32767 < 2^29). The blocked GEMV multiplies this by
/// the per-block M before the i64 accumulate.
#[cfg(not(target_arch = "aarch64"))]
fn block_partial(w: &[i8], x: &[i16]) -> i64 {
    let mut p = 0i32;
    for (a, b) in w.iter().zip(x) {
        p += (*a as i32) * (*b as i32);
    }
    p as i64
}

/// NEON: ONE 64-element block partial, fused. Four i32 accumulators for MAC
/// throughput, combined with cheap vector adds, then a SINGLE horizontal
/// reduction — vs `dot_w8_x16`'s generic 256-drain path which paid four
/// `vaddlvq` per 64-block. Bit-identical to `block_partial` (associative
/// integer sum). Caller guarantees w.len() == x.len() == 64.
#[cfg(target_arch = "aarch64")]
#[inline]
fn block_partial(w: &[i8], x: &[i16]) -> i64 {
    use std::arch::aarch64::*;
    debug_assert_eq!(w.len(), 64);
    debug_assert_eq!(x.len(), 64);
    unsafe {
        let mut a0 = vdupq_n_s32(0);
        let mut a1 = vdupq_n_s32(0);
        let mut a2 = vdupq_n_s32(0);
        let mut a3 = vdupq_n_s32(0);
        let mut i = 0;
        while i < 64 {
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
        // Combine 4 accumulators with vector adds (1-cycle throughput), then
        // one widening horizontal reduction — partial < 2^29 so no overflow.
        let s = vaddq_s32(vaddq_s32(a0, a1), vaddq_s32(a2, a3));
        vaddlvq_s32(s)
    }
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

/// BLOCKED projection: activations carry per-64-channel-block scales and
/// each (row, block) has its own multiplier — out[r] =
/// rnd(Σ_b partial_rb·M[r][b], shift), accumulated EXACTLY in i64
/// (partial ≤ 2^29, M ≤ 2^24, 16 blocks ⇒ ≤ 2^57 — no intermediate
/// rounding, half-even applied once).
#[allow(clippy::too_many_arguments)]
pub fn gemv_w8_x16_blocked(
    pool: &Pool,
    w: &[i8],
    x: &[i16],
    rows: usize,
    cols: usize,
    m_blocks: &[i32], // row-major [rows][cols/64]
    shift: u8,
    out: &mut [i64],
) {
    debug_assert_eq!(out.len(), rows);
    let blocks = cols / 64;
    debug_assert_eq!(m_blocks.len(), rows * blocks);
    let out_addr = SendPtr(out.as_mut_ptr());
    pool.run(rows, 16, &move |start, end| {
        let out_ptr = out_addr;
        for r in start..end {
            let wrow = &w[r * cols..(r + 1) * cols];
            let mrow = &m_blocks[r * blocks..(r + 1) * blocks];
            let mut acc = 0i64;
            for b in 0..blocks {
                let p = block_partial(&wrow[b * 64..(b + 1) * 64], &x[b * 64..(b + 1) * 64]);
                acc = acc.wrapping_add(p.wrapping_mul(mrow[b] as i64));
            }
            unsafe { *out_ptr.0.add(r) = rnd(acc, shift) };
        }
    });
}

// ---------------------------------------------------------------------------
// sdot two-limb path (ARMv8.2 DotProd): 4× the MAC density of vmlal_s16.
//
// `sdot` does i8×i8 → i32 (16 MACs/instruction) but our activations are i16.
// Decompose each i16 x into two i8 limbs:  x = 256·xh + xl,  with
//   xh = (x + 128) >> 8   ∈ [-128, 128]   (rounded high limb)
//   xl = x - 256·xh        ∈ [-128, 127]   (fits i8 exactly)
// Then  w·x = 256·(w·xh) + (w·xl)  — two sdot passes. The lone overflow is
// xh = 128 (only when x = 32767): store xh clamped to 127 and add the per-row
// correction +256·w at that lane. Bit-identical to the scalar i64 dot.
// ---------------------------------------------------------------------------

/// Preprocess an i16 activation vector into the two i8 limb arrays + the list
/// of clamped lane indices per 64-block. Done ONCE per GEMV (shared by all
/// rows). `clamp` holds, per block, the local lane indices where xh == 128.
#[cfg(target_arch = "aarch64")]
fn split_limbs(x: &[i16]) -> (Vec<i8>, Vec<i8>, Vec<Vec<u8>>) {
    let cols = x.len();
    let blocks = cols / 64;
    let mut xl = vec![0i8; cols];
    let mut xh = vec![0i8; cols];
    let mut clamp = vec![Vec::new(); blocks];
    for i in 0..cols {
        let xi = x[i] as i32;
        let h = (xi + 128) >> 8; // ∈ [-128, 128]
        let l = xi - 256 * h; // ∈ [-128, 127]
        xl[i] = l as i8;
        if h == 128 {
            xh[i] = 127;
            clamp[i / 64].push((i % 64) as u8);
        } else {
            xh[i] = h as i8;
        }
    }
    (xl, xh, clamp)
}

/// One 64-element block: returns (Σ w·xl, Σ w·xh) as i32 via eight `sdot`
/// instructions (16 MACs each). `vdotq_s32` is unstable on stable Rust, so
/// the DotProd instruction is issued through inline `asm!`. Lane sums ≤
/// 64·127·128 < 2^21, well within i32. Pointers must address 64 valid bytes.
#[cfg(all(target_arch = "aarch64", not(feature = "nightly_dotprod")))]
#[inline]
unsafe fn block_dot_sdot(w: *const i8, xl: *const i8, xh: *const i8) -> (i32, i32) {
    let sum_l: u64;
    let sum_h: u64;
    core::arch::asm!(
        "movi v0.4s, #0",
        "movi v1.4s, #0",
        "ldp q2, q3, [{w}]",
        "ldp q4, q5, [{w}, #32]",
        "ldp q6, q7, [{xl}]",
        "ldp q16, q17, [{xl}, #32]",
        "sdot v0.4s, v2.16b, v6.16b",
        "sdot v0.4s, v3.16b, v7.16b",
        "sdot v0.4s, v4.16b, v16.16b",
        "sdot v0.4s, v5.16b, v17.16b",
        "ldp q6, q7, [{xh}]",
        "ldp q16, q17, [{xh}, #32]",
        "sdot v1.4s, v2.16b, v6.16b",
        "sdot v1.4s, v3.16b, v7.16b",
        "sdot v1.4s, v4.16b, v16.16b",
        "sdot v1.4s, v5.16b, v17.16b",
        "addv s0, v0.4s",
        "addv s1, v1.4s",
        "fmov {sl:w}, s0",
        "fmov {sh:w}, s1",
        w = in(reg) w,
        xl = in(reg) xl,
        xh = in(reg) xh,
        sl = out(reg) sum_l,
        sh = out(reg) sum_h,
        out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _,
        out("v5") _, out("v6") _, out("v7") _, out("v16") _, out("v17") _,
        options(nostack, readonly),
    );
    (sum_l as u32 as i32, sum_h as u32 as i32)
}

/// One 64-element block partial via the two-limb sdot decomposition, with the
/// xh==128 overflow correction. Bit-identical to the scalar i64 dot.
///
/// Stable build: the `asm!` path (measured 0.75× — the asm barrier blocks
/// cross-block scheduling). Nightly + `nightly_dotprod`: the `vdotq_s32`
/// intrinsic, which the compiler schedules freely — the experiment to find
/// the real sdot ceiling.
#[cfg(all(target_arch = "aarch64", not(feature = "nightly_dotprod")))]
#[inline]
unsafe fn block_partial_sdot(w: &[i8], xl: &[i8], xh: &[i8], clamp: &[u8]) -> i64 {
    let (sl, sh) = block_dot_sdot(w.as_ptr(), xl.as_ptr(), xh.as_ptr());
    let mut p = sl as i64 + 256 * (sh as i64);
    for &ci in clamp {
        p += 256 * (w[ci as usize] as i64); // xh was 128, stored 127
    }
    p
}

#[cfg(all(target_arch = "aarch64", feature = "nightly_dotprod"))]
#[target_feature(enable = "dotprod")]
unsafe fn block_partial_sdot(w: &[i8], xl: &[i8], xh: &[i8], clamp: &[u8]) -> i64 {
    use std::arch::aarch64::*;
    let mut al = vdupq_n_s32(0);
    let mut ah = vdupq_n_s32(0);
    let mut i = 0;
    while i < 64 {
        let wv = vld1q_s8(w.as_ptr().add(i));
        al = vdotq_s32(al, wv, vld1q_s8(xl.as_ptr().add(i)));
        ah = vdotq_s32(ah, wv, vld1q_s8(xh.as_ptr().add(i)));
        i += 16;
    }
    let mut p = vaddvq_s32(al) as i64 + 256 * (vaddvq_s32(ah) as i64);
    for &ci in clamp {
        p += 256 * (w[ci as usize] as i64);
    }
    p
}

/// One 64-element i8×i8 block partial via a SINGLE sdot pass (no limb split).
/// This is the kernel a *dynamic per-block i8 activation* design would use —
/// the measurement of the speed ceiling the i16→i8 quality re-derivation
/// would unlock. `asm!` issues the DotProd instruction on stable.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn block_dot_i8(w: *const i8, x: *const i8) -> i32 {
    let sum: u64;
    core::arch::asm!(
        "movi v0.4s, #0",
        "ldp q2, q3, [{w}]",
        "ldp q4, q5, [{w}, #32]",
        "ldp q6, q7, [{x}]",
        "ldp q16, q17, [{x}, #32]",
        "sdot v0.4s, v2.16b, v6.16b",
        "sdot v0.4s, v3.16b, v7.16b",
        "sdot v0.4s, v4.16b, v16.16b",
        "sdot v0.4s, v5.16b, v17.16b",
        "addv s0, v0.4s",
        "fmov {s:w}, s0",
        w = in(reg) w,
        x = in(reg) x,
        s = out(reg) sum,
        out("v0") _, out("v2") _, out("v3") _, out("v4") _,
        out("v5") _, out("v6") _, out("v7") _, out("v16") _, out("v17") _,
        options(nostack, readonly),
    );
    sum as u32 as i32
}

/// Blocked GEMV with i8 activations, single sdot per block (the dynamic-i8
/// design's hot kernel). NOT bit-comparable to the i16 path (different
/// activation width) — exists purely to MEASURE the speed ceiling. Scalar
/// reference for the equality test is `block`-wise i8·i8.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
pub fn gemv_i8_blocked(
    pool: &Pool,
    w: &[i8],
    x: &[i8],
    rows: usize,
    cols: usize,
    m_blocks: &[i32],
    shift: u8,
    out: &mut [i64],
) {
    let blocks = cols / 64;
    let out_addr = SendPtr(out.as_mut_ptr());
    pool.run(rows, 16, &move |start, end| {
        let out_ptr = out_addr;
        for r in start..end {
            let wrow = &w[r * cols..(r + 1) * cols];
            let mrow = &m_blocks[r * blocks..(r + 1) * blocks];
            let mut acc = 0i64;
            for b in 0..blocks {
                let p = unsafe { block_dot_i8(wrow[b * 64..].as_ptr(), x[b * 64..].as_ptr()) };
                acc = acc.wrapping_add((p as i64).wrapping_mul(mrow[b] as i64));
            }
            unsafe { *out_ptr.0.add(r) = rnd(acc, shift) };
        }
    });
}

/// Blocked GEMV via the sdot two-limb path. Falls back to the vmlal kernel if
/// DotProd is unavailable. Bit-identical to `gemv_w8_x16_blocked`.
#[allow(clippy::too_many_arguments)]
pub fn gemv_w8_x16_blocked_dot(
    pool: &Pool,
    w: &[i8],
    x: &[i16],
    rows: usize,
    cols: usize,
    m_blocks: &[i32],
    shift: u8,
    out: &mut [i64],
) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            let blocks = cols / 64;
            let (xl, xh, clamp) = split_limbs(x);
            let out_addr = SendPtr(out.as_mut_ptr());
            let (xl, xh, clamp) = (&xl, &xh, &clamp);
            pool.run(rows, 16, &move |start, end| {
                let out_ptr = out_addr;
                for r in start..end {
                    let wrow = &w[r * cols..(r + 1) * cols];
                    let mrow = &m_blocks[r * blocks..(r + 1) * blocks];
                    let mut acc = 0i64;
                    for b in 0..blocks {
                        let p = unsafe {
                            block_partial_sdot(
                                &wrow[b * 64..(b + 1) * 64],
                                &xl[b * 64..(b + 1) * 64],
                                &xh[b * 64..(b + 1) * 64],
                                &clamp[b],
                            )
                        };
                        acc = acc.wrapping_add(p.wrapping_mul(mrow[b] as i64));
                    }
                    unsafe { *out_ptr.0.add(r) = rnd(acc, shift) };
                }
            });
            return;
        }
    }
    gemv_w8_x16_blocked(pool, w, x, rows, cols, m_blocks, shift, out);
}

/// LEGACY blocked GEMV: calls the generic `dot_w8_x16` (256-drain path) once
/// per 64-block. Kept ONLY for the thermal-robust A/B that justified the
/// fused `block_partial` (which does one reduction per block instead of the
/// generic path's four). Not used in production.
#[allow(clippy::too_many_arguments)]
pub fn gemv_w8_x16_blocked_legacy(
    pool: &Pool,
    w: &[i8],
    x: &[i16],
    rows: usize,
    cols: usize,
    m_blocks: &[i32],
    shift: u8,
    out: &mut [i64],
) {
    let blocks = cols / 64;
    let out_addr = SendPtr(out.as_mut_ptr());
    pool.run(rows, 16, &move |start, end| {
        let out_ptr = out_addr;
        for r in start..end {
            let wrow = &w[r * cols..(r + 1) * cols];
            let mrow = &m_blocks[r * blocks..(r + 1) * blocks];
            let mut acc = 0i64;
            for b in 0..blocks {
                let p = dot_w8_x16(&wrow[b * 64..(b + 1) * 64], &x[b * 64..(b + 1) * 64]);
                acc = acc.wrapping_add(p.wrapping_mul(mrow[b] as i64));
            }
            unsafe { *out_ptr.0.add(r) = rnd(acc, shift) };
        }
    });
}

/// Bytes wrapper for the legacy blocked path (A/B only).
#[allow(clippy::too_many_arguments)]
pub fn gemv_blocked_legacy_bytes(
    pool: &Pool,
    w: &[u8],
    x: &[u8],
    rows: usize,
    cols: usize,
    m_blocks: &[i32],
    shift: u8,
    out: &mut [i64],
) {
    gemv_w8_x16_blocked_legacy(pool, bytes_as_i8(w), bytes_as_i16(x), rows, cols, m_blocks, shift, out)
}

/// Bytes wrapper for the sdot blocked path.
#[allow(clippy::too_many_arguments)]
pub fn gemv_blocked_dot_bytes(
    pool: &Pool,
    w: &[u8],
    x: &[u8],
    rows: usize,
    cols: usize,
    m_blocks: &[i32],
    shift: u8,
    out: &mut [i64],
) {
    gemv_w8_x16_blocked_dot(pool, bytes_as_i8(w), bytes_as_i16(x), rows, cols, m_blocks, shift, out)
}

#[allow(clippy::too_many_arguments)]
pub fn gemv_blocked_bytes(
    pool: &Pool,
    w: &[u8],
    x: &[u8],
    rows: usize,
    cols: usize,
    m_blocks: &[i32],
    shift: u8,
    out: &mut [i64],
) {
    gemv_w8_x16_blocked(pool, bytes_as_i8(w), bytes_as_i16(x), rows, cols, m_blocks, shift, out)
}

/// One blocked-GEMV job in a fused group. `out` is a raw pointer to `rows`
/// disjoint i64 slots (caller owns the allocation and guarantees disjointness
/// across jobs in the group).
pub struct BlockedJob<'a> {
    pub w: &'a [u8],
    pub x: &'a [u8],
    pub m: &'a [i32],
    pub shift: u8,
    pub rows: usize,
    pub cols: usize,
    pub out: *mut i64,
}
// SAFETY: each job's `out` points to a disjoint, caller-owned region.
unsafe impl Send for BlockedJob<'_> {}
unsafe impl Sync for BlockedJob<'_> {}

/// Fused group of independent blocked GEMVs under ONE pool dispatch — the
/// q/k/v projections (all read xn) and gate/up (both read xn2) are
/// data-independent, so batching them *could* remove pool barriers.
///
/// MEASURED SLOWER (night-3): wiring this into the Qwen forward dropped
/// 32→29.6 tok/s. The persistent pool's ack barrier is cheap, so little was
/// saved, while the per-row job lookup + indirection cost more. Conclusion:
/// barriers are NOT the bottleneck — i16 `vmlal` MAC density is. Kept
/// (tested, bit-exact) as the recorded negative result; the forward uses
/// separate `gemv_blocked_bytes` calls. Bit-identical to running each job
/// via `gemv_blocked_bytes`: the per-row math is unchanged.
pub fn gemv_blocked_group(pool: &Pool, jobs: &[BlockedJob]) {
    let mut prefix = Vec::with_capacity(jobs.len() + 1);
    prefix.push(0usize);
    for j in jobs {
        prefix.push(prefix.last().unwrap() + j.rows);
    }
    let total = *prefix.last().unwrap();
    if total == 0 {
        return;
    }
    pool.run(total, 16, &move |start, end| {
        for g in start..end {
            // Which job owns global row g? (small linear scan — jobs.len() ≤ 3)
            let mut j = 0;
            while prefix[j + 1] <= g {
                j += 1;
            }
            let job = &jobs[j];
            let r = g - prefix[j];
            let wi = bytes_as_i8(job.w);
            let xi = bytes_as_i16(job.x);
            let blocks = job.cols / 64;
            let wrow = &wi[r * job.cols..(r + 1) * job.cols];
            let mrow = &job.m[r * blocks..(r + 1) * blocks];
            let mut acc = 0i64;
            for b in 0..blocks {
                let p = block_partial(&wrow[b * 64..(b + 1) * 64], &xi[b * 64..(b + 1) * 64]);
                acc = acc.wrapping_add(p.wrapping_mul(mrow[b] as i64));
            }
            // SAFETY: disjoint per-job output regions, distinct local rows.
            unsafe { *job.out.add(r) = rnd(acc, job.shift) };
        }
    });
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

/// f32 twin of SendPtr for the committed-float kernels (fkernels).
#[derive(Clone, Copy)]
pub(crate) struct SendPtrF(pub *mut f32);
unsafe impl Send for SendPtrF {}
unsafe impl Sync for SendPtrF {}

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
    fn blocked_gemv_exact() {
        let pool = Pool::new(8);
        let mut s = 11u64;
        let (rows, cols) = (97, 1024);
        let blocks = cols / 64;
        let w: Vec<i8> = (0..rows * cols).map(|_| xorshift(&mut s) as i8).collect();
        let x: Vec<i16> = (0..cols).map(|_| xorshift(&mut s) as i16).collect();
        let m: Vec<i32> = (0..rows * blocks).map(|_| (xorshift(&mut s) % (1 << 24)) as i32 + 1).collect();
        let mut out = vec![0i64; rows];
        gemv_w8_x16_blocked(&pool, &w, &x, rows, cols, &m, 20, &mut out);
        for r in 0..rows {
            let mut acc = 0i64;
            for b in 0..blocks {
                let p = dot_w8_x16_scalar(
                    &w[r * cols + b * 64..r * cols + (b + 1) * 64],
                    &x[b * 64..(b + 1) * 64],
                );
                acc = acc.wrapping_add(p.wrapping_mul(m[r * blocks + b] as i64));
            }
            assert_eq!(out[r], rnd(acc, 20), "row {r}");
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn i8_blocked_equals_scalar() {
        let pool = Pool::new(8);
        let mut s = 0x5151u64;
        let (rows, cols) = (97, 1024);
        let blocks = cols / 64;
        let w: Vec<i8> = (0..rows * cols).map(|_| xorshift(&mut s) as i8).collect();
        let x: Vec<i8> = (0..cols).map(|_| xorshift(&mut s) as i8).collect();
        let m: Vec<i32> = (0..rows * blocks).map(|_| (xorshift(&mut s) % (1 << 24)) as i32 + 1).collect();
        let mut out = vec![0i64; rows];
        gemv_i8_blocked(&pool, &w, &x, rows, cols, &m, 20, &mut out);
        for r in 0..rows {
            let mut acc = 0i64;
            for b in 0..blocks {
                let mut p = 0i32;
                for c in 0..64 {
                    p += w[r * cols + b * 64 + c] as i32 * x[b * 64 + c] as i32;
                }
                acc = acc.wrapping_add((p as i64).wrapping_mul(m[r * blocks + b] as i64));
            }
            assert_eq!(out[r], rnd(acc, 20), "row {r}");
        }
    }

    #[test]
    fn blocked_sdot_equals_scalar() {
        // The sdot two-limb path must equal the scalar definition EXACTLY,
        // including the xh==128 (x==32767) overflow-correction edge.
        let pool = Pool::new(8);
        let mut s = 0xDEAD_BEEFu64;
        let (rows, cols) = (131, 1024);
        let blocks = cols / 64;
        let w: Vec<i8> = (0..rows * cols).map(|_| xorshift(&mut s) as i8).collect();
        // Force many extreme activations (32767/-32768) to exercise limbs.
        let x: Vec<i16> = (0..cols)
            .map(|_| match xorshift(&mut s) % 8 {
                0 => 32767,
                1 => -32768,
                2 => 32766,
                _ => xorshift(&mut s) as i16,
            })
            .collect();
        let m: Vec<i32> = (0..rows * blocks).map(|_| (xorshift(&mut s) % (1 << 24)) as i32 + 1).collect();
        let mut out = vec![0i64; rows];
        gemv_w8_x16_blocked_dot(&pool, &w, &x, rows, cols, &m, 20, &mut out);
        for r in 0..rows {
            let mut acc = 0i64;
            for b in 0..blocks {
                let p = dot_w8_x16_scalar(
                    &w[r * cols + b * 64..r * cols + (b + 1) * 64],
                    &x[b * 64..(b + 1) * 64],
                );
                acc = acc.wrapping_add(p.wrapping_mul(m[r * blocks + b] as i64));
            }
            assert_eq!(out[r], rnd(acc, 20), "row {r}");
        }
    }

    #[test]
    fn blocked_group_equals_separate() {
        // The fused q/k/v-style group must equal running each job alone.
        let pool = Pool::new(8);
        let mut s = 99u64;
        let cols = 1024usize;
        let blocks = cols / 64;
        let x: Vec<i16> = (0..cols).map(|_| xorshift(&mut s) as i16).collect();
        let xb: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
        let specs = [2048usize, 1024, 1024]; // nh*dh, nkv*dh, nkv*dh
        let mut ws = vec![];
        let mut ms = vec![];
        let mut want = vec![];
        for &rows in &specs {
            let w: Vec<i8> = (0..rows * cols).map(|_| xorshift(&mut s) as i8).collect();
            let m: Vec<i32> =
                (0..rows * blocks).map(|_| (xorshift(&mut s) % (1 << 24)) as i32 + 1).collect();
            let mut sep = vec![0i64; rows];
            gemv_w8_x16_blocked(&pool, &w, &x, rows, cols, &m, 20, &mut sep);
            want.push(sep);
            ws.push(w.iter().map(|&v| v as u8).collect::<Vec<u8>>());
            ms.push(m);
        }
        let mut outs: Vec<Vec<i64>> = specs.iter().map(|&r| vec![0i64; r]).collect();
        {
            let mut jobs = vec![];
            for (i, &rows) in specs.iter().enumerate() {
                jobs.push(BlockedJob {
                    w: &ws[i],
                    x: &xb,
                    m: &ms[i],
                    shift: 20,
                    rows,
                    cols,
                    out: outs[i].as_mut_ptr(),
                });
            }
            gemv_blocked_group(&pool, &jobs);
        }
        for (g, w) in outs.iter().zip(&want) {
            assert_eq!(g, w);
        }
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
