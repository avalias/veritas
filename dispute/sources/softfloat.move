/// FW-6: IEEE-754 binary32 in pure integer Move — the on-chain arithmetic
/// for ONE disputed float micro-op (SPEC FW-6). Port of vm/src/softfloat.rs
/// (the Rust twin); held to its answers by generated cross-vectors.
///
/// Round-to-nearest-even, gradual underflow, signed zeros, infinities;
/// NaNs canonicalize to quiet 0x7FC00000 (committed rule — honest traces
/// are NaN-free, so this only ever decides adversarial junk).
///
/// Internal exponents carry a +4096 offset (EOFF) so they stay in u64.
module dispute::softfloat;

const QNAN: u32 = 0x7FC00000;
const INF_EXP: u32 = 255;
const EOFF: u64 = 4096;

fun sign_of(b: u32): u32 { b >> 31 }

fun exp_of(b: u32): u32 { (b >> 23) & 0xFF }

fun frac_of(b: u32): u32 { b & 0x7FFFFF }

fun is_nan(b: u32): bool { exp_of(b) == INF_EXP && frac_of(b) != 0 }

fun is_inf(b: u32): bool { exp_of(b) == INF_EXP && frac_of(b) == 0 }

fun is_zero(b: u32): bool { (b & 0x7FFFFFFF) == 0 }

fun pack(sign: u32, exp: u32, frac: u32): u32 { (sign << 31) | (exp << 23) | frac }

/// Significand with implicit bit; exponent offset by EOFF
/// (value = sig × 2^(e − EOFF − 127 − 23)).
fun norm_sig(exp_raw: u32, frac: u32): (u64, u32) {
    if (exp_raw == 0) {
        let mut f = frac;
        let mut shift = 0u64;
        while (f < 0x800000) {
            f = f << 1;
            shift = shift + 1;
        };
        (EOFF + 1 - shift, f)
    } else {
        (EOFF + (exp_raw as u64), frac | 0x800000)
    }
}

/// Round-pack (twin of softfloat.rs::round_pack): value = sig × 2^(eo −
/// EOFF − 127 − 47), sig ∈ [2^47, 2^48). RN-even; subnormals re-round at
/// the denormalized position; carries ripple into the exponent by integer
/// addition (the packed `kept` contains the implicit bit, supplying the
/// step from exponent eo−1).
fun round_pack(sign: u32, eo: u64, sig: u64): u32 {
    if (eo >= EOFF + 255) {
        return pack(sign, INF_EXP, 0)
    };
    let sub = eo <= EOFF;
    let shift = if (sub) { 24 + (EOFF + 1 - eo) } else { 24 };
    if (shift >= 64) {
        return pack(sign, 0, 0)
    };
    let kept = sig >> (shift as u8);
    let rem = sig & ((1u64 << (shift as u8)) - 1);
    let half = 1u64 << ((shift - 1) as u8);
    let ru: u32 = if (rem > half || (rem == half && (kept & 1) == 1)) { 1 } else { 0 };
    let body = if (sub) {
        (kept as u32) + ru
    } else {
        (((eo - EOFF - 1) as u32) << 23) + (kept as u32) + ru
    };
    if (body >= (INF_EXP << 23)) {
        return pack(sign, INF_EXP, 0)
    };
    (sign << 31) | body
}

/// fp32 multiply, RN.
public fun fmul(a: u32, b: u32): u32 {
    if (is_nan(a) || is_nan(b)) {
        return QNAN
    };
    let ss = sign_of(a) ^ sign_of(b);
    if (is_inf(a) || is_inf(b)) {
        if (is_zero(a) || is_zero(b)) {
            return QNAN
        };
        return pack(ss, INF_EXP, 0)
    };
    if (is_zero(a) || is_zero(b)) {
        return pack(ss, 0, 0)
    };
    let (ea, ma) = norm_sig(exp_of(a), frac_of(a));
    let (eb, mb) = norm_sig(exp_of(b), frac_of(b));
    // E = ea + eb − 126 unshifted (see Rust twin's frame comment).
    let mut e = ea + eb - EOFF - 126;
    let mut p = (ma as u64) * (mb as u64);
    if (p < 1u64 << 47) {
        p = p << 1;
        e = e - 1;
    };
    round_pack(ss, e, p)
}

/// fp32 add, RN.
public fun fadd(a: u32, b: u32): u32 {
    if (is_nan(a) || is_nan(b)) {
        return QNAN
    };
    if (is_inf(a)) {
        if (is_inf(b) && sign_of(a) != sign_of(b)) {
            return QNAN
        };
        return a
    };
    if (is_inf(b)) {
        return b
    };
    if (is_zero(a) && is_zero(b)) {
        return pack(sign_of(a) & sign_of(b), 0, 0)
    };
    if (is_zero(a)) {
        return b
    };
    if (is_zero(b)) {
        return a
    };
    let (ea, ma) = norm_sig(exp_of(a), frac_of(a));
    let (eb, mb) = norm_sig(exp_of(b), frac_of(b));
    let (sa, sb) = (sign_of(a), sign_of(b));
    let a_hi = ea > eb || (ea == eb && ma >= mb);
    let (e, hi, hi_s, lo0, lo_s, elo) = if (a_hi) {
        (ea, (ma as u64) << 24, sa, (mb as u64) << 24, sb, eb)
    } else {
        (eb, (mb as u64) << 24, sb, (ma as u64) << 24, sa, ea)
    };
    let mut d = e - elo;
    if (d > 50) {
        d = 50;
    };
    let sticky: u64 =
        if (d == 0 || (lo0 & ((1u64 << (d as u8)) - 1)) == 0) { 0 } else { 1 };
    let lo = (lo0 >> (d as u8)) | sticky;
    let (mut sig, sign) = if (hi_s == lo_s) {
        (hi + lo, hi_s)
    } else if (hi > lo) {
        (hi - lo, hi_s)
    } else if (lo > hi) {
        (lo - hi, lo_s)
    } else {
        return pack(0, 0, 0)
    };
    let mut e2 = e;
    while (sig >= 1u64 << 48) {
        let st = sig & 1;
        sig = (sig >> 1) | st;
        e2 = e2 + 1;
    };
    while (sig < 1u64 << 47) {
        sig = sig << 1;
        e2 = e2 - 1;
    };
    round_pack(sign, e2, sig)
}

/// fp32 fused multiply-add a·b + c, RN — ONE rounding (the committed DOTF
/// lane op). Exact 48-bit product; 128-bit alignment workspace.
public fun ffma(a: u32, b: u32, c: u32): u32 {
    if (is_nan(a) || is_nan(b) || is_nan(c)) {
        return QNAN
    };
    if (is_inf(a) || is_inf(b)) {
        if (is_zero(a) || is_zero(b)) {
            return QNAN
        };
        let ps = sign_of(a) ^ sign_of(b);
        if (is_inf(c) && sign_of(c) != ps) {
            return QNAN
        };
        return pack(ps, INF_EXP, 0)
    };
    if (is_inf(c)) {
        return c
    };
    if (is_zero(a) || is_zero(b)) {
        let ps = sign_of(a) ^ sign_of(b);
        if (is_zero(c)) {
            return pack(ps & sign_of(c), 0, 0)
        };
        return c
    };
    let ps = sign_of(a) ^ sign_of(b);
    let (ea, ma) = norm_sig(exp_of(a), frac_of(a));
    let (eb, mb) = norm_sig(exp_of(b), frac_of(b));
    let mut ep = ea + eb - EOFF - 126;
    let mut p = (ma as u64) * (mb as u64);
    if (p < 1u64 << 47) {
        p = p << 1;
        ep = ep - 1;
    };
    if (is_zero(c)) {
        return round_pack(ps, ep, p)
    };
    let (ec, mc0) = norm_sig(exp_of(c), frac_of(c));
    let cs = sign_of(c);
    let mc = (mc0 as u64) << 24; // c at the 47-fraction scale
    let p_hi = ep > ec || (ep == ec && p >= mc);
    let (e, hi, hi_s, lo0, lo_s, elo) = if (p_hi) {
        (ep, (p as u128) << 32, ps, (mc as u128) << 32, cs, ec)
    } else {
        (ec, (mc as u128) << 32, cs, (p as u128) << 32, ps, ep)
    };
    let mut d = e - elo;
    if (d > 100) {
        d = 100;
    };
    let sticky: u128 =
        if (d == 0 || (lo0 & ((1u128 << (d as u8)) - 1)) == 0) { 0 } else { 1 };
    let lo = (lo0 >> (d as u8)) | sticky;
    let (mut sig, sign) = if (hi_s == lo_s) {
        (hi + lo, hi_s)
    } else if (hi > lo) {
        (hi - lo, hi_s)
    } else if (lo > hi) {
        (lo - hi, lo_s)
    } else {
        return pack(0, 0, 0)
    };
    let mut e2 = e;
    while (sig >= 1u128 << 80) {
        let st = sig & 1;
        sig = (sig >> 1) | st;
        e2 = e2 + 1;
    };
    while (sig < 1u128 << 79) {
        sig = sig << 1;
        e2 = e2 - 1;
    };
    let low32 = (sig & 0xFFFFFFFF) as u64;
    let top = (sig >> 32) as u64;
    let folded = top | (if (low32 != 0) { 1 } else { 0 });
    round_pack(sign, e2, folded)
}

/// The committed 64-element block dot (fkernels tree) entirely on-chain:
/// 16 virtual fma lanes, pinned combine, pinned horizontal — the DOTF
/// one-step core. `w`/`x` are 64 fp32 bit patterns each.
public fun block_dot(w: &vector<u32>, x: &vector<u32>): u32 {
    let mut lane = vector<u32>[];
    let mut k = 0u64;
    while (k < 16) {
        lane.push_back(0);
        k = k + 1;
    };
    let mut i = 0u64;
    while (i < 4) {
        let mut kk = 0u64;
        while (kk < 16) {
            let j = 16 * i + kk;
            let cur = lane[kk];
            *(&mut lane[kk]) = ffma(w[j], x[j], cur);
            kk = kk + 1;
        };
        i = i + 1;
    };
    // s[l] = (a0[l] + a1[l]) + (a2[l] + a3[l]); lane index = 4k + l.
    let mut l = 0u64;
    let mut s = vector<u32>[];
    while (l < 4) {
        s.push_back(fadd(fadd(lane[l], lane[4 + l]), fadd(lane[8 + l], lane[12 + l])));
        l = l + 1;
    };
    fadd(fadd(s[0], s[1]), fadd(s[2], s[3]))
}

/// fp32 divide a/b, RN — long division on significands, remainder sticky.
public fun fdiv(a: u32, b: u32): u32 {
    if (is_nan(a) || is_nan(b)) {
        return QNAN
    };
    let ss = sign_of(a) ^ sign_of(b);
    if (is_inf(a)) {
        if (is_inf(b)) {
            return QNAN
        };
        return pack(ss, INF_EXP, 0)
    };
    if (is_inf(b)) {
        return pack(ss, 0, 0)
    };
    if (is_zero(b)) {
        if (is_zero(a)) {
            return QNAN
        };
        return pack(ss, INF_EXP, 0)
    };
    if (is_zero(a)) {
        return pack(ss, 0, 0)
    };
    let (ea, ma) = norm_sig(exp_of(a), frac_of(a));
    let (eb, mb) = norm_sig(exp_of(b), frac_of(b));
    let num = (ma as u64) << 26;
    let q = num / (mb as u64);
    let r = num % (mb as u64);
    let mut sig = q;
    // ea − eb cancels both EOFFs; round_pack expects one ⇒ add it back.
    let mut e = ea + 127 + EOFF - eb;
    if (sig < 1u64 << 26) {
        sig = sig << 22;
        e = e - 1;
    } else {
        sig = sig << 21;
    };
    if (r != 0) {
        sig = sig | 1;
    };
    round_pack(ss, e, sig)
}

fun isqrt_wide(v: u128): u64 {
    let mut lo: u128 = 0;
    let mut hi: u128 = 1u128 << 39;
    while (lo < hi) {
        let mid = (lo + hi + 1) >> 1;
        if (mid * mid <= v) {
            lo = mid;
        } else {
            hi = mid - 1;
        };
    };
    lo as u64
}

/// fp32 sqrt, RN — integer sqrt at even exponent, remainder sticky.
public fun fsqrt(a: u32): u32 {
    if (is_nan(a)) {
        return QNAN
    };
    if (is_zero(a)) {
        return a
    };
    if (sign_of(a) == 1) {
        return QNAN
    };
    if (is_inf(a)) {
        return a
    };
    let (ea, ma) = norm_sig(exp_of(a), frac_of(a));
    // e_unb = ea − EOFF − 150, kept offset-free via parity on (ea − EOFF).
    // Work with eu = ea (offset EOFF+150 absorbed below).
    let e_unb_off = ea; // value = ma · 2^(ea − EOFF − 150)
    // Even/odd of the true unbiased exponent == even/odd of (ea − EOFF − 150)
    // == even/odd of ea (EOFF=4096 and 150 are even).
    let (m2, e2_off) = if ((e_unb_off % 2) != 0) {
        ((ma as u64) << 1, e_unb_off - 1)
    } else {
        ((ma as u64), e_unb_off)
    };
    let wide = (m2 as u128) << 52;
    let s = isqrt_wide(wide);
    let rem = wide - (s as u128) * (s as u128);
    // true e2 = e2_off − EOFF − 150 (even); half_e = (e2 − 52)/2;
    // round_pack offset exponent eo = half_e + 174 + EOFF
    //   = (e2_off − EOFF − 202)/2 + 174 + EOFF = (e2_off − 202)/2 + 174 + EOFF/2 + ... 
    // compute directly in u64: eo = (e2_off + EOFF) / 2 + 174 - 101 - EOFF/2 ... use
    // eo = (e2_off − 202)/2 + 174 + (EOFF/2): with EOFF even and e2_off even-aligned
    // to the true exponent's parity, all terms are exact integers.
    let eo = (e2_off + 146 + EOFF) / 2; // ≡ ((e2_off−EOFF−202)/2) + 174 + EOFF
    let mut sig = s;
    let mut e = eo;
    while (sig < 1u64 << 47) {
        sig = sig << 1;
        e = e - 1;
    };
    if (rem != 0) {
        sig = sig | 1;
    };
    round_pack(0, e, sig)
}

/// fp32 floor (toward −inf).
public fun ffloor(a: u32): u32 {
    if (is_nan(a)) {
        return QNAN
    };
    if (is_inf(a) || is_zero(a)) {
        return a
    };
    let eraw = exp_of(a);
    if (eraw < 127) {
        return if (sign_of(a) == 0) { 0 } else { 0xBF800000 }
    };
    let e = eraw - 127;
    if (e >= 23) {
        return a
    };
    let mask = (1u32 << ((23 - e) as u8)) - 1;
    let trunc = a & (mask ^ 0xFFFFFFFF);
    if (sign_of(a) == 1 && (a & mask) != 0) {
        return fadd(trunc, 0xBF800000)
    };
    trunc
}

/// fp32 → i32 (returned as u32 two's complement), truncating, saturating,
/// NaN → 0 (the committed convert rule; == Rust `as` semantics).
public fun ftoi(a: u32): u32 {
    if (is_nan(a)) {
        return 0
    };
    let neg = sign_of(a) == 1;
    if (is_inf(a)) {
        return if (neg) { 0x80000000 } else { 0x7FFFFFFF }
    };
    let eraw = exp_of(a);
    if (eraw < 127) {
        return 0
    };
    let e = eraw - 127;
    if (e >= 31) {
        return if (neg) { 0x80000000 } else { 0x7FFFFFFF }
    };
    let m = (frac_of(a) | 0x800000) as u64;
    let v = if (e >= 23) { m << ((e - 23) as u8) } else { m >> ((23 - e) as u8) };
    if (neg) {
        // two's complement negate in u32
        (((v as u32) ^ 0xFFFFFFFF) as u64 + 1) as u32
    } else {
        v as u32
    }
}

/// i32 (u32 two's complement) → fp32, RN.
public fun itof(v: u32): u32 {
    if (v == 0) {
        return 0
    };
    let neg = (v >> 31) == 1;
    let mag: u64 = if (neg) {
        (((v ^ 0xFFFFFFFF) as u64) + 1)
    } else {
        v as u64
    };
    // leading zeros of mag (mag ≤ 2^31)
    let mut lz = 0u64;
    let mut probe = mag;
    while (probe < (1u64 << 63)) {
        probe = probe << 1;
        lz = lz + 1;
    };
    let sig = mag << ((lz - 16) as u8);
    let e = EOFF + 190 - lz; // round_pack offset frame
    round_pack(if (neg) { 1 } else { 0 }, e, sig)
}
