//! Soft-float fp32 in PURE INTEGER arithmetic — the FW-6 one-step verifier's
//! arithmetic core, and the Move twin's reference (SPEC FW-6).
//!
//! IEEE-754 binary32, round-to-nearest-even, gradual underflow (subnormals),
//! signed zeros, infinities. NaNs canonicalize to quiet 0x7FC00000 — a
//! committed rule (hardware NaN payloads differ across vendors; honest
//! traces are NaN-free, see fmodel docs, so canonicalization only ever
//! decides adversarial junk deterministically).
//!
//! Everything here is u32/u64 bit manipulation — no float types in the
//! implementation (the workspace float ban stays intact); only the TESTS
//! compare against hardware ops, asserting bit equality over millions of
//! random and targeted patterns. The Move port (dispute/sources/softfloat
//! .move) is held to these answers by generated vectors.

const QNAN: u32 = 0x7FC0_0000;
const INF_EXP: u32 = 255;

#[inline]
fn sign_of(b: u32) -> u32 {
    b >> 31
}

#[inline]
fn exp_of(b: u32) -> u32 {
    (b >> 23) & 0xFF
}

#[inline]
fn frac_of(b: u32) -> u32 {
    b & 0x7F_FFFF
}

#[inline]
fn is_nan(b: u32) -> bool {
    exp_of(b) == INF_EXP && frac_of(b) != 0
}

#[inline]
fn is_inf(b: u32) -> bool {
    exp_of(b) == INF_EXP && frac_of(b) == 0
}

#[inline]
fn is_zero(b: u32) -> bool {
    b & 0x7FFF_FFFF == 0
}

#[inline]
fn pack(sign: u32, exp: u32, frac: u32) -> u32 {
    (sign << 31) | (exp << 23) | frac
}

/// Significand with implicit bit (24-bit for normals) and UNBIASED-ish
/// exponent such that value = sig × 2^(exp − 127 − 23). Subnormals are
/// normalized into the same form (exp may go ≤ 0).
#[inline]
fn norm_sig(exp_raw: u32, frac: u32) -> (i32, u32) {
    if exp_raw == 0 {
        // subnormal: value = frac × 2^(1 − 127 − 23); normalize.
        let shift = frac.leading_zeros() - 8; // bring bit 23 up
        (1 - shift as i32, frac << shift)
    } else {
        (exp_raw as i32, frac | 0x80_0000)
    }
}

/// Round-pack: sign, biased exponent `e` (i32, may be ≤ 0 or ≥ 255), and a
/// significand `sig` scaled so that the VALUE is sig × 2^(e − 127 − 47),
/// with sig ∈ [2^47, 2^48) for normal-range inputs (callers normalize).
/// Returns the rounded ieee bits. Round-to-nearest-even on the discarded
/// low bits; subnormal results re-round at the denormalized position;
/// overflow → ±inf.
fn round_pack(sign: u32, e: i32, sig: u64) -> u32 {
    debug_assert!((1u64 << 47..1 << 48).contains(&sig));
    if e >= INF_EXP as i32 {
        return pack(sign, INF_EXP, 0);
    }
    // Discard count: 24 for normals; more when the result is subnormal.
    let (e_out, shift) = if e <= 0 { (0u32, 24 + (1 - e) as u32) } else { (e as u32, 24) };
    if shift >= 64 {
        return pack(sign, 0, 0); // underflow to zero (sticky can't round up from 0 past half)
    }
    let kept = sig >> shift;
    let rem = sig & ((1u64 << shift) - 1);
    let half = 1u64 << (shift - 1);
    let round_up = rem > half || (rem == half && (kept & 1) == 1);
    // Integer packing with carry ripple: the normal-path `kept` CONTAINS the
    // implicit bit (2^23), which supplies the +1 step from exponent e−1 —
    // so pack (e−1) and let the addition carry. Subnormal `kept` has no
    // implicit bit and packs against exponent field 0. Rounding carries
    // propagate into the exponent (incl. subnormal → smallest normal,
    // and 254 → inf via the check below) automatically.
    let _ = e_out;
    let body = if e <= 0 {
        kept as u32 + round_up as u32
    } else {
        (((e - 1) as u32) << 23) + kept as u32 + round_up as u32
    };
    if body >= (INF_EXP << 23) {
        return pack(sign, INF_EXP, 0);
    }
    (sign << 31) | body
}

/// fp32 multiply, RN.
pub fn fmul(a: u32, b: u32) -> u32 {
    if is_nan(a) || is_nan(b) {
        return QNAN;
    }
    let ss = sign_of(a) ^ sign_of(b);
    if is_inf(a) || is_inf(b) {
        if is_zero(a) || is_zero(b) {
            return QNAN; // inf × 0
        }
        return pack(ss, INF_EXP, 0);
    }
    if is_zero(a) || is_zero(b) {
        return pack(ss, 0, 0);
    }
    let (ea, ma) = norm_sig(exp_of(a), frac_of(a));
    let (eb, mb) = norm_sig(exp_of(b), frac_of(b));
    // round_pack's frame: value = sig·2^(E−127−47) with sig ∈ [2^47, 2^48).
    // p = ma·mb at combined scale 2^(ea+eb−254−46) ⇒ E = ea+eb−126 unshifted.
    let mut e = ea + eb - 126;
    let mut p = (ma as u64) * (mb as u64); // ∈ [2^46, 2^48)
    if p < 1 << 47 {
        p <<= 1;
        e -= 1;
    }
    round_pack(ss, e, p)
}

/// fp32 add, RN. (Subtraction = add with flipped sign bit.)
pub fn fadd(a: u32, b: u32) -> u32 {
    if is_nan(a) || is_nan(b) {
        return QNAN;
    }
    if is_inf(a) {
        if is_inf(b) && sign_of(a) != sign_of(b) {
            return QNAN; // inf − inf
        }
        return a;
    }
    if is_inf(b) {
        return b;
    }
    if is_zero(a) && is_zero(b) {
        // IEEE: (+0)+(−0) = +0 under RN; (−0)+(−0) = −0.
        return pack(sign_of(a) & sign_of(b), 0, 0);
    }
    if is_zero(a) {
        return b;
    }
    if is_zero(b) {
        return a;
    }
    let (ea, ma) = norm_sig(exp_of(a), frac_of(a));
    let (eb, mb) = norm_sig(exp_of(b), frac_of(b));
    // Work at 2^(e − 127 − 47) scale: significands << 24, plus 3 guard
    // bits of headroom for alignment sticky (shift cap keeps it exact).
    let (sa, sb) = (sign_of(a), sign_of(b));
    let (mut e, hi, hi_s, mut lo, lo_s) = if (ea, ma) >= (eb, mb) {
        (ea, (ma as u64) << 24, sa, (mb as u64) << 24, sb)
    } else {
        (eb, (mb as u64) << 24, sb, (ma as u64) << 24, sa)
    };
    let d = e - if (ea, ma) >= (eb, mb) { eb } else { ea };
    // Align lo down by d with sticky (cap: beyond 50 bits it is pure sticky).
    let d = d.min(50) as u32;
    let sticky = if d == 0 { 0 } else { u64::from(lo & ((1u64 << d) - 1) != 0) };
    lo = (lo >> d) | sticky;
    let mut sig;
    let sign;
    if hi_s == lo_s {
        sig = hi + lo;
        sign = hi_s;
    } else if hi > lo {
        sig = hi - lo;
        sign = hi_s;
    } else if lo > hi {
        sig = lo - hi;
        sign = lo_s;
    } else {
        return pack(0, 0, 0); // exact cancellation → +0 (RN rule)
    };
    // Normalize to [2^47, 2^48).
    while sig >= 1 << 48 {
        let st = sig & 1;
        sig = (sig >> 1) | st;
        e += 1;
    }
    while sig < 1 << 47 {
        sig <<= 1;
        e -= 1;
    }
    round_pack(sign, e, sig)
}

/// fp32 fused multiply-add a·b + c, RN — ONE rounding at the end (the
/// committed DOTF lane op). Exact 48-bit product, 128-bit alignment.
pub fn ffma(a: u32, b: u32, c: u32) -> u32 {
    if is_nan(a) || is_nan(b) || is_nan(c) {
        return QNAN;
    }
    // Product specials.
    if is_inf(a) || is_inf(b) {
        if is_zero(a) || is_zero(b) {
            return QNAN;
        }
        let ps = sign_of(a) ^ sign_of(b);
        if is_inf(c) && sign_of(c) != ps {
            return QNAN;
        }
        return pack(ps, INF_EXP, 0);
    }
    if is_inf(c) {
        return c;
    }
    if is_zero(a) || is_zero(b) {
        // 0·b + c = c (with the +0/−0 rule when c is zero too).
        let ps = sign_of(a) ^ sign_of(b);
        if is_zero(c) {
            return pack(ps & sign_of(c), 0, 0);
        }
        return c;
    }
    let ps = sign_of(a) ^ sign_of(b);
    let (ea, ma) = norm_sig(exp_of(a), frac_of(a));
    let (eb, mb) = norm_sig(exp_of(b), frac_of(b));
    let mut ep = ea + eb - 126; // same frame as fmul (see comment there)
    let mut p = (ma as u64) * (mb as u64); // [2^46, 2^48)
    if p < 1 << 47 {
        p <<= 1;
        ep -= 1;
    }
    if is_zero(c) {
        return round_pack(ps, ep, p);
    }
    let (ec0, mc0) = norm_sig(exp_of(c), frac_of(c));
    let cs = sign_of(c);
    // Bring c to the product's 47-fraction scale: value = sig × 2^(e−127−47).
    let mc = (mc0 as u64) << 24;
    let ec = ec0;
    // 128-bit workspace with 32 guard bits below the 47-scale.
    let (mut e, hi, hi_s, mut lo, lo_s) = if (ep, p) >= (ec, mc) {
        (ep, (p as u128) << 32, ps, (mc as u128) << 32, cs)
    } else {
        (ec, (mc as u128) << 32, cs, (p as u128) << 32, ps)
    };
    let d = e - if (ep, p) >= (ec, mc) { ec } else { ep };
    let d = d.min(100) as u32;
    let sticky = if d == 0 { 0 } else { u128::from(lo & ((1u128 << d) - 1) != 0) };
    lo = (lo >> d) | sticky;
    let mut sig;
    let sign;
    if hi_s == lo_s {
        sig = hi + lo;
        sign = hi_s;
    } else if hi > lo {
        sig = hi - lo;
        sign = hi_s;
    } else if lo > hi {
        sig = lo - hi;
        sign = lo_s;
    } else {
        return pack(0, 0, 0);
    };
    // Normalize into [2^79, 2^80) (47+32 = 79 fraction scale).
    while sig >= 1 << 80 {
        let st = sig & 1;
        sig = (sig >> 1) | st;
        e += 1;
    }
    while sig < 1 << 79 {
        sig <<= 1;
        e -= 1;
    }
    // Fold the 32 guard bits into a sticky-preserving 48-bit significand.
    let low32 = (sig & 0xFFFF_FFFF) as u64;
    let top = (sig >> 32) as u64; // [2^47, 2^48)
    let folded = top | u64::from(low32 != 0);
    round_pack(sign, e, folded)
}

#[cfg(test)]
#[allow(clippy::float_arithmetic)] // tests compare against HARDWARE floats
#[allow(clippy::needless_range_loop)] // spec-literal lane indices in the tree
mod tests {
    use super::*;

    fn xorshift(s: &mut u64) -> u64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        *s
    }

    /// Random bit pattern with boosted odds of interesting exponents
    /// (zeros, subnormals, near-overflow, inf/nan).
    fn rand_f32_bits(s: &mut u64) -> u32 {
        let r = xorshift(s);
        let mut b = r as u32;
        match (r >> 32) % 8 {
            0 => b &= 0x807F_FFFF,                  // zero/subnormal exponent
            1 => b = (b & 0x807F_FFFF) | 0x7F80_0000, // inf/nan
            2 => b = (b & 0x807F_FFFF) | 0x7F00_0000, // huge
            3 => b = (b & 0x807F_FFFF) | 0x0080_0000, // tiny normal
            _ => {}
        }
        b
    }

    /// Canonicalize hardware results the way the committed semantics do:
    /// any NaN → QNAN (payloads are vendor-specific; we pin one).
    fn canon(bits: u32) -> u32 {
        if is_nan(bits) {
            QNAN
        } else {
            bits
        }
    }

    #[test]
    fn fmul_matches_hardware_bitwise() {
        let mut s = 0xF00Du64;
        for i in 0..2_000_000u64 {
            let (a, b) = (rand_f32_bits(&mut s), rand_f32_bits(&mut s));
            let want = canon((f32::from_bits(a) * f32::from_bits(b)).to_bits());
            let got = fmul(a, b);
            assert_eq!(got, want, "i={i} a={a:08x} b={b:08x}");
        }
    }

    #[test]
    fn fadd_matches_hardware_bitwise() {
        let mut s = 0xADD5u64;
        for i in 0..2_000_000u64 {
            let (a, b) = (rand_f32_bits(&mut s), rand_f32_bits(&mut s));
            let want = canon((f32::from_bits(a) + f32::from_bits(b)).to_bits());
            let got = fadd(a, b);
            assert_eq!(got, want, "i={i} a={a:08x} b={b:08x}");
        }
    }

    #[test]
    fn ffma_matches_hardware_bitwise() {
        let mut s = 0xF3AAu64;
        for i in 0..2_000_000u64 {
            let a = rand_f32_bits(&mut s);
            let b = rand_f32_bits(&mut s);
            let c = rand_f32_bits(&mut s);
            let want =
                canon(f32::from_bits(a).mul_add(f32::from_bits(b), f32::from_bits(c)).to_bits());
            let got = ffma(a, b, c);
            assert_eq!(got, want, "i={i} a={a:08x} b={b:08x} c={c:08x}");
        }
    }

    /// The committed 64-block dot (fkernels tree), replayed entirely in
    /// softfloat, must equal the hardware kernel bit-for-bit — this is the
    /// exact computation the Move one-step verifier performs for a DOTF op.
    #[test]
    fn committed_block_dot_in_softfloat() {
        let mut s = 0xD07Fu64;
        for _ in 0..2_000 {
            let w: Vec<u32> = (0..64)
                .map(|_| {
                    // bf16-pattern weights (low 16 bits zero), like the real kernel.
                    rand_f32_bits(&mut s) & 0xFFFF_0000
                })
                .collect();
            let x: Vec<u32> = (0..64).map(|_| rand_f32_bits(&mut s) & 0x7FFF_FFFF).collect();
            // skip non-finite draws — honest traces are finite
            if w.iter().chain(&x).any(|&v| exp_of(v) == INF_EXP) {
                continue;
            }
            // Hardware: the fkernels scalar tree.
            let mut a = [[0f32; 4]; 4];
            for i in 0..4 {
                for k in 0..4 {
                    for l in 0..4 {
                        let j = 16 * i + 4 * k + l;
                        a[k][l] =
                            f32::from_bits(w[j]).mul_add(f32::from_bits(x[j]), a[k][l]);
                    }
                }
            }
            let mut hsum = [0f32; 4];
            for l in 0..4 {
                hsum[l] = (a[0][l] + a[1][l]) + (a[2][l] + a[3][l]);
            }
            let hw = ((hsum[0] + hsum[1]) + (hsum[2] + hsum[3])).to_bits();
            // Softfloat: same tree, integer-only.
            let mut sa = [[0u32; 4]; 4];
            for i in 0..4 {
                for k in 0..4 {
                    for l in 0..4 {
                        let j = 16 * i + 4 * k + l;
                        sa[k][l] = ffma(w[j], x[j], sa[k][l]);
                    }
                }
            }
            let mut ss_ = [0u32; 4];
            for l in 0..4 {
                ss_[l] = fadd(fadd(sa[0][l], sa[1][l]), fadd(sa[2][l], sa[3][l]));
            }
            let sf = fadd(fadd(ss_[0], ss_[1]), fadd(ss_[2], ss_[3]));
            assert_eq!(sf, canon(hw));
        }
    }
}
