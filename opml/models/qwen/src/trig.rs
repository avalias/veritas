//! Pure-integer rotary table generation (SPEC §6.3 discipline, no libm).
//!
//! Pipeline, all in i128/u128 fixed point:
//!   π            — Machin's formula 16·atan(1/5) − 4·atan(1/239), Q96
//!   θ^(1/64)     — six nested integer square roots in Q32
//!   invfreq_i    — θ^(−i/64) by iterated Q32 multiplication + reciprocal
//!   angle p·f    — Q32 product, reduced mod 2π
//!   sin/cos      — Taylor in Q32 on [−π, π], 11 terms (|t|³/6 … converges
//!                  well below Q14 resolution)
//!   output       — Q1.14 i16 via vm::exec::rnd (THE rounding rule)
//!
//! The tables are canonical artifacts; tests pin anchor values and the
//! Pythagorean identity envelope.

use vm::exec::{rnd, sat16};

const Q32: u32 = 32;

/// π in Q96 via Machin: π = 16·atan(1/5) − 4·atan(1/239).
/// atan(1/x) = Σ_{k≥0} (−1)^k / ((2k+1)·x^(2k+1)), exact integer series.
fn pi_q96() -> i128 {
    fn atan_inv_q96(x: i128) -> i128 {
        let one = 1i128 << 96;
        let mut term = one / x; // 1/x in Q96
        let mut sum = term;
        let mut k = 1i128;
        loop {
            term /= x * x; // now 1/x^(2k+1)
            if term == 0 {
                return sum;
            }
            let contrib = term / (2 * k + 1);
            if k % 2 == 1 {
                sum -= contrib;
            } else {
                sum += contrib;
            }
            k += 1;
        }
    }
    16 * atan_inv_q96(5) - 4 * atan_inv_q96(239)
}

/// Floor square root (u128 Newton).
fn isqrt(n: u128) -> u128 {
    if n == 0 {
        return 0;
    }
    let bits = 128 - n.leading_zeros();
    let mut x = 1u128 << bits.div_ceil(2);
    loop {
        let y = (x + n / x) >> 1;
        if y >= x {
            return x;
        }
        x = y;
    }
}

/// sqrt of a Q32 value, in Q32: sqrt(v/2^32)·2^32 = isqrt(v·2^32).
fn sqrt_q32(v: u128) -> u128 {
    isqrt(v << Q32)
}

/// sin and cos of t (Q32, |t| ≤ π) by Taylor series in i128 Q32.
fn sincos_q32(t: i128) -> (i128, i128) {
    // sin: t − t³/3! + t⁵/5! − …   cos: 1 − t²/2! + t⁴/4! − …
    let one = 1i128 << Q32;
    let t2 = (t * t) >> Q32;
    let (mut sin, mut cos) = (t, one);
    let mut term_s = t; // t^(2k+1)/(2k+1)!
    let mut term_c = one; // t^(2k)/(2k)!
    let mut k = 1i128;
    while k <= 11 {
        term_c = ((term_c * t2) >> Q32) / ((2 * k - 1) * (2 * k));
        term_s = ((term_s * t2) >> Q32) / ((2 * k) * (2 * k + 1));
        if k % 2 == 1 {
            cos -= term_c;
            sin -= term_s;
        } else {
            cos += term_c;
            sin += term_s;
        }
        if term_c == 0 && term_s == 0 {
            break;
        }
        k += 1;
    }
    (sin, cos)
}

/// Rotary tables for `max_seq` positions × `pairs` frequency pairs:
/// (cos, sin, −sin), each Q1.14 i16, row-major [pos][pair].
/// invfreq_i = theta^(−i/pairs), the RoPE convention with d = 2·pairs.
pub fn rope_tables(theta: u64, pairs: usize, max_seq: usize) -> (Vec<i16>, Vec<i16>, Vec<i16>) {
    let pi = (pi_q96() >> (96 - Q32 as i32)) as u128; // π in Q32
    let two_pi = 2 * pi;

    // r = theta^(1/64) via nested square roots (pairs must be 64 = 2^6).
    assert_eq!(pairs, 64, "table generation pinned to head_dim 128");
    let mut r = (theta as u128) << Q32; // Q32
    for _ in 0..6 {
        r = sqrt_q32(r);
    }
    // invfreq_i = 2^64 / r^i (Q32 reciprocal of Q32 power).
    let mut invfreq = Vec::with_capacity(pairs);
    let mut r_pow = 1u128 << Q32; // r^0
    for _ in 0..pairs {
        invfreq.push(((1u128 << (2 * Q32)) / r_pow) as i128);
        r_pow = (r_pow * r) >> Q32;
    }

    let (mut cos_t, mut sin_t, mut nsin_t) = (
        Vec::with_capacity(max_seq * pairs),
        Vec::with_capacity(max_seq * pairs),
        Vec::with_capacity(max_seq * pairs),
    );
    for pos in 0..max_seq {
        for &f in &invfreq {
            // angle = pos·f mod 2π, shifted to [−π, π].
            let mut a = ((pos as i128) * f) % (two_pi as i128);
            if a > pi as i128 {
                a -= two_pi as i128;
            }
            let (s, c) = sincos_q32(a);
            // Q32 → Q14 with round-half-even; |values| ≤ 1.0 ⇒ sat16 is a
            // no-op except for exact 1.0 (16384, in range).
            let cq = sat16(rnd(c as i64, 18));
            let sq = sat16(rnd(s as i64, 18));
            cos_t.push(cq);
            sin_t.push(sq);
            nsin_t.push(sq.saturating_neg());
        }
    }
    (cos_t, sin_t, nsin_t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pi_anchor() {
        // π·2^32 = 13493037704.5… — floor within 1 ulp either side.
        let pi32 = (pi_q96() >> 64) as i64;
        assert!((pi32 - 13493037705).abs() <= 1, "pi Q32 = {pi32}");
    }

    #[test]
    fn rope_anchors_and_identity() {
        let (cos, sin, nsin) = rope_tables(1_000_000, 64, 96);
        // pos 0: angle 0 for every pair.
        for i in 0..64 {
            assert_eq!(cos[i], 16384, "cos(0) = 1.0");
            assert_eq!(sin[i], 0);
            assert_eq!(nsin[i], 0);
        }
        // pos 1, pair 0: invfreq = 1 ⇒ angle 1 rad: cos ≈ 0.5403, sin ≈ 0.8415.
        let (c10, s10) = (cos[64] as i64, sin[64] as i64);
        assert!((c10 - 8852).abs() <= 2, "cos(1) Q14 got {c10}");
        assert!((s10 - 13788).abs() <= 2, "sin(1) Q14 got {s10}");
        // Pythagorean identity within quantization noise everywhere.
        for i in 0..cos.len() {
            let (c, s) = (cos[i] as i64, sin[i] as i64);
            let norm = c * c + s * s;
            assert!(
                (norm - (1i64 << 28)).abs() < (1 << 16),
                "unit circle at {i}: {norm}"
            );
            assert_eq!(nsin[i] as i64, -(sin[i] as i64));
        }
        // High pair index ⇒ tiny frequency ⇒ slow rotation: pos 1 pair 63
        // angle ≈ theta^(-63/64) ≈ 1.24e-6·... cos ≈ 1.0.
        assert_eq!(cos[64 + 63], 16384);
    }
}
