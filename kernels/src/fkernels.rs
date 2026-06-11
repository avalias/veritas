//! FW-6 committed FLOAT kernels: "Qwen as published", deterministically.
//!
//! The committed semantics of a float dot are THIS module's scalar
//! definition: bf16 weights widened to f32 (exact), fused multiply-adds
//! into SIXTEEN virtual accumulator lanes per 64-element block (the lane
//! interleave NEON/GPU vectors use), a pinned pairwise combine, and a
//! sequential block chain. Float addition is NOT associative, so — unlike
//! the integer path — the REDUCTION SHAPE is part of the spec. Any
//! implementation (scalar, NEON, WGSL) must produce bit-identical f32
//! results, and the tests assert exactly that.
//!
//! No libm anywhere: only +, *, fused mul-add, division and sqrt — all
//! IEEE-754 correctly-rounded on every target — plus exp via a committed
//! polynomial (qwen::fmath::cexp).
#![allow(clippy::float_arithmetic)] // FW-6: floats ARE the committed semantics here

use crate::{Pool, SendPtrF};

/// bf16 (raw bits) → f32: exact widening (bf16 is a truncated f32).
#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// THE committed 64-element block dot (scalar definition).
///
/// Sixteen virtual accumulator lanes: lane (k, l) with k ∈ 0..4 (vector),
/// l ∈ 0..4 (component) accumulates elements j = 16·i + 4·k + l for
/// i ∈ 0..4, via fused multiply-add in sequence. Combine: vector adds
/// (a0+a1), (a2+a3), then their sum; horizontal (s0+s1) + (s2+s3).
/// This maps 1:1 onto 4×f32x4 NEON registers and 4×vec4<f32> in WGSL.
pub fn fdot_block_scalar(w: &[u16], x: &[f32]) -> f32 {
    debug_assert_eq!(w.len(), 64);
    debug_assert_eq!(x.len(), 64);
    let mut a = [[0f32; 4]; 4]; // a[k][l]
    for i in 0..4 {
        for k in 0..4 {
            for l in 0..4 {
                let j = 16 * i + 4 * k + l;
                a[k][l] = bf16_to_f32(w[j]).mul_add(x[j], a[k][l]);
            }
        }
    }
    // s[l] = (a0[l] + a1[l]) + (a2[l] + a3[l])
    let mut s = [0f32; 4];
    for l in 0..4 {
        s[l] = (a[0][l] + a[1][l]) + (a[2][l] + a[3][l]);
    }
    // horizontal: (s0 + s1) + (s2 + s3)
    (s[0] + s[1]) + (s[2] + s[3])
}

/// Committed row dot: sequential chain of block partials (cols % 64 == 0).
pub fn fdot_row_scalar(w: &[u16], x: &[f32]) -> f32 {
    debug_assert_eq!(w.len(), x.len());
    let mut acc = 0f32;
    for (wb, xb) in w.chunks_exact(64).zip(x.chunks_exact(64)) {
        acc += fdot_block_scalar(wb, xb);
    }
    acc
}

/// NEON implementation of the SAME tree. 4×f32x4 accumulators, vfmaq per
/// 16-lane stride, vector combine, pinned horizontal sum. Bit-identical to
/// `fdot_row_scalar` (asserted by tests).
#[cfg(target_arch = "aarch64")]
pub fn fdot_row(w: &[u16], x: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    debug_assert_eq!(w.len(), x.len());
    debug_assert!(w.len().is_multiple_of(64));
    let mut acc = 0f32;
    unsafe {
        let mut b = 0;
        while b < w.len() {
            let mut a0 = vdupq_n_f32(0.0);
            let mut a1 = vdupq_n_f32(0.0);
            let mut a2 = vdupq_n_f32(0.0);
            let mut a3 = vdupq_n_f32(0.0);
            for i in 0..4 {
                let base = b + 16 * i;
                // bf16 → f32: u16x8 load, widen with <<16, reinterpret.
                let wb = vld1q_u16(w.as_ptr().add(base));
                let wlo = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(wb), 16));
                let whi = vreinterpretq_f32_u32(vshll_high_n_u16(wb, 16));
                let wb2 = vld1q_u16(w.as_ptr().add(base + 8));
                let wlo2 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(wb2), 16));
                let whi2 = vreinterpretq_f32_u32(vshll_high_n_u16(wb2, 16));
                a0 = vfmaq_f32(a0, wlo, vld1q_f32(x.as_ptr().add(base)));
                a1 = vfmaq_f32(a1, whi, vld1q_f32(x.as_ptr().add(base + 4)));
                a2 = vfmaq_f32(a2, wlo2, vld1q_f32(x.as_ptr().add(base + 8)));
                a3 = vfmaq_f32(a3, whi2, vld1q_f32(x.as_ptr().add(base + 12)));
            }
            let s = vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3));
            // pinned horizontal: (s0+s1) + (s2+s3)
            let p = (vgetq_lane_f32(s, 0) + vgetq_lane_f32(s, 1))
                + (vgetq_lane_f32(s, 2) + vgetq_lane_f32(s, 3));
            acc += p;
            b += 64;
        }
    }
    acc
}

#[cfg(not(target_arch = "aarch64"))]
pub fn fdot_row(w: &[u16], x: &[f32]) -> f32 {
    fdot_row_scalar(w, x)
}

/// Row-parallel committed-float GEMV: out[r] = fdot_row(w_row_r, x).
/// Rows are independent values, each internally order-pinned, so ANY thread
/// layout is bit-stable (§9.1, float edition — the canonical tree is what
/// licenses this).
pub fn fgemv(pool: &Pool, w: &[u16], x: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
    debug_assert_eq!(out.len(), rows);
    debug_assert_eq!(w.len(), rows * cols);
    let out_addr = SendPtrF(out.as_mut_ptr());
    pool.run(rows, 16, &move |start, end| {
        let out_ptr = out_addr;
        for r in start..end {
            let d = fdot_row(&w[r * cols..(r + 1) * cols], x);
            // SAFETY: distinct rows.
            unsafe { *out_ptr.0.add(r) = d };
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xorshift(s: &mut u64) -> u64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        *s
    }

    fn rand_bf16(s: &mut u64) -> u16 {
        // Random-sign, moderate-exponent bf16s (realistic weight range).
        let v = (xorshift(s) % 2000) as f32 / 1000.0 - 1.0;
        (v.to_bits() >> 16) as u16
    }

    /// THE float-kernel test: NEON must equal the scalar definition
    /// BIT-FOR-BIT (f32 bits compared as u32 — no tolerance anywhere).
    #[test]
    fn fdot_neon_equals_scalar_bitwise() {
        let mut s = 0xF10A7u64;
        for trial in 0..200 {
            let cols = 64 * (1 + (xorshift(&mut s) % 48) as usize);
            let w: Vec<u16> = (0..cols).map(|_| rand_bf16(&mut s)).collect();
            let x: Vec<f32> =
                (0..cols).map(|_| (xorshift(&mut s) % 4000) as f32 / 100.0 - 20.0).collect();
            let a = fdot_row(&w, &x);
            let b = fdot_row_scalar(&w, &x);
            assert_eq!(a.to_bits(), b.to_bits(), "trial {trial} cols {cols}");
        }
    }

    /// Thread-count invariance: 1-thread pool == 8-thread pool, bitwise.
    #[test]
    fn fgemv_thread_invariant_bitwise() {
        let mut s = 0xBEEFu64;
        let (rows, cols) = (333, 1024);
        let w: Vec<u16> = (0..rows * cols).map(|_| rand_bf16(&mut s)).collect();
        let x: Vec<f32> =
            (0..cols).map(|_| (xorshift(&mut s) % 4000) as f32 / 100.0 - 20.0).collect();
        let p1 = Pool::new(1);
        let p8 = Pool::new(8);
        let mut o1 = vec![0f32; rows];
        let mut o8 = vec![0f32; rows];
        fgemv(&p1, &w, &x, rows, cols, &mut o1);
        fgemv(&p8, &w, &x, rows, cols, &mut o8);
        for r in 0..rows {
            assert_eq!(o1[r].to_bits(), o8[r].to_bits(), "row {r}");
            assert_eq!(o1[r].to_bits(), fdot_row_scalar(&w[r * cols..(r + 1) * cols], &x).to_bits());
        }
    }
}
