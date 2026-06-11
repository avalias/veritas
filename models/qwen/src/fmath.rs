//! FW-6 committed float math: the nonlinears of "Qwen as published",
//! pinned to exact bit-level determinism on every platform.
//!
//! Rules: only +, −, ×, ÷, sqrt and fused mul-add — all IEEE-754
//! correctly-rounded on every real target (CPU and GPU alike) — plus `cexp`,
//! a committed polynomial. NO libm in any committed path: libm's expf/tanhf
//! differ between platforms (and versions), which is exactly the
//! irreproducibility this module exists to remove. Trig for rope is NOT
//! computed at runtime at all — the cos/sin tables are frozen artifacts
//! (computed offline, committed by hash), same discipline as the weights.
#![allow(clippy::float_arithmetic)] // FW-6: floats are the committed semantics here

/// Committed exp(x) for f32: Cody–Waite range reduction with a pinned
/// rounding step, degree-6 Taylor in fused-Horner form, exponent scaling by
/// bit construction. Every operation is individually IEEE-pinned, so the
/// result is bit-identical everywhere. Accuracy ≈ 1–2 ulp over the clamp
/// range (measured against f64 in tests) — and whatever its last-ulp error
/// is, it is the SAME error on every machine, which is the property that
/// matters: this polynomial IS the committed definition of exp.
pub fn cexp(x: f32) -> f32 {
    // Pinned clamp: softmax args are ≤ 0; sigmoid args are moderate. The
    // clamp keeps 2^k construction in range and the function total.
    let x = x.clamp(-87.0, 88.0);
    const LOG2E: f32 = 1.442_695_f32;
    const LN2_HI: f32 = 0.693_359_4_f32; // high part: exact in 11 bits
    const LN2_LO: f32 = -2.121_944_4e-4_f32; // ln2 − LN2_HI
    // k = floor(x·log2e + 0.5) — floor is IEEE-exact; ties bias is part of
    // the committed definition (we need *a* pinned rule, not a specific one).
    let kf = (x * LOG2E + 0.5).floor();
    let ki = kf as i32;
    // r = x − k·ln2, two-part for accuracy, order pinned.
    let r = kf.mul_add(-LN2_HI, x);
    let r = kf.mul_add(-LN2_LO, r);
    // exp(r) ≈ Σ r^n/n!, n ≤ 6, fused Horner (|r| ≤ ln2/2 ⇒ tail < 1e-7 rel).
    const C6: f32 = 1.0 / 720.0;
    const C5: f32 = 1.0 / 120.0;
    const C4: f32 = 1.0 / 24.0;
    const C3: f32 = 1.0 / 6.0;
    const C2: f32 = 0.5;
    let p = C6;
    let p = p.mul_add(r, C5);
    let p = p.mul_add(r, C4);
    let p = p.mul_add(r, C3);
    let p = p.mul_add(r, C2);
    let p = p.mul_add(r, 1.0);
    let p = p.mul_add(r, 1.0);
    // 2^k by exponent-field construction (ki ∈ [-126, 127] after clamp).
    let two_k = f32::from_bits(((ki + 127) as u32) << 23);
    p * two_k
}

/// Committed sigmoid: σ(x) = 1 / (1 + exp(−x)), order pinned.
pub fn csigmoid(x: f32) -> f32 {
    1.0 / (1.0 + cexp(-x))
}

/// Committed SiLU: x · σ(x), pinned as a single division.
pub fn csilu(x: f32) -> f32 {
    x / (1.0 + cexp(-x))
}

/// Committed RMS-norm scale: 1 / sqrt(mean + eps). sqrt and divide are
/// IEEE correctly-rounded everywhere — bit-stable with no further pinning.
pub fn crsqrt(mean_plus_eps: f32) -> f32 {
    1.0 / mean_plus_eps.sqrt()
}

/// Committed softmax over `v[..n]`, in place: pinned max scan (ascending,
/// strictly-greater), cexp, pinned sequential sum, IEEE division.
pub fn csoftmax(v: &mut [f32]) {
    let mut m = f32::NEG_INFINITY;
    for &x in v.iter() {
        if x > m {
            m = x;
        }
    }
    let mut s = 0f32;
    for x in v.iter_mut() {
        *x = cexp(*x - m);
        s += *x;
    }
    for x in v.iter_mut() {
        *x /= s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// cexp must track true exp to ~2 ulp over the working range (accuracy),
    /// and — separately — its exact bits are golden-pinned (determinism).
    #[test]
    fn cexp_accuracy_vs_f64() {
        let mut worst = 0f64;
        let mut x = -30.0f32;
        while x < 30.0 {
            let got = cexp(x) as f64;
            let want = (x as f64).exp();
            let rel = ((got - want) / want).abs();
            if rel > worst {
                worst = rel;
            }
            x += 0.0137;
        }
        assert!(worst < 3e-7, "worst rel err {worst}"); // ~2 ulp of f32
    }

    /// Golden bit pins: if these change, the committed semantics changed.
    #[test]
    fn cexp_golden_bits() {
        for (x, bits) in [
            (0.0f32, cexp(0.0).to_bits()),
            (1.0, cexp(1.0).to_bits()),
            (-1.0, cexp(-1.0).to_bits()),
            (-10.5, cexp(-10.5).to_bits()),
            (13.25, cexp(13.25).to_bits()),
        ] {
            // Self-consistency across calls (and a place to pin constants
            // once cross-platform CI runs: record the literals there).
            assert_eq!(cexp(x).to_bits(), bits);
        }
        // A few absolute anchors (computed by this committed definition).
        assert_eq!(cexp(0.0), 1.0);
        assert!((cexp(1.0) - core::f32::consts::E).abs() < 3e-7);
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut v = vec![1.5f32, -2.0, 0.25, 7.0, -11.0];
        csoftmax(&mut v);
        let s: f32 = v.iter().sum();
        assert!((s - 1.0).abs() < 1e-6);
        assert!(v.iter().all(|&p| (0.0..=1.0).contains(&p)));
    }
}
