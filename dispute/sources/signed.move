/// i64 arithmetic on u64 two's-complement bit patterns (SPEC §9.3).
/// Move has no signed integers; every formula here is normative and
/// cross-tested against the Rust implementation (vm/src/exec.rs) by the
/// generated vector suite — a semantic gap here is a soundness bug.
module dispute::signed;

const MASK64: u128 = 0xFFFF_FFFF_FFFF_FFFF;
const ALL_ONES: u64 = 0xFFFF_FFFF_FFFF_FFFF;
const SIGN_BIT: u64 = 1 << 63;

/// Wrapping add (mod 2^64) via u128 — Move aborts on native overflow.
public fun wadd(a: u64, b: u64): u64 {
    ((((a as u128) + (b as u128)) & MASK64) as u64)
}

/// Wrapping mul (mod 2^64); operands < 2^64 so the u128 product never aborts.
public fun wmul(a: u64, b: u64): u64 {
    ((((a as u128) * (b as u128)) & MASK64) as u64)
}

public fun wsub(a: u64, b: u64): u64 {
    wadd(a, neg(b))
}

/// Two's-complement negation; neg(0) = 0, neg(i64::MIN) = i64::MIN.
public fun neg(x: u64): u64 {
    wadd(x ^ ALL_ONES, 1)
}

public fun is_neg(x: u64): bool {
    x >= SIGN_BIT
}

/// Signed less-than via the sign-bit offset trick (SPEC §9.3).
public fun slt(a: u64, b: u64): bool {
    (a ^ SIGN_BIT) < (b ^ SIGN_BIT)
}

public fun sgt(a: u64, b: u64): bool {
    slt(b, a)
}

public fun sext8(b: u8): u64 {
    if (b >= 0x80) { (b as u64) | 0xFFFF_FFFF_FFFF_FF00 } else { (b as u64) }
}

/// Input: raw 16-bit pattern in the low bits.
public fun sext16(v: u64): u64 {
    if (v >= 0x8000) { v | 0xFFFF_FFFF_FFFF_0000 } else { v }
}

/// Input: raw 32-bit pattern in the low bits.
public fun sext32(v: u64): u64 {
    if (v >= 0x8000_0000) { v | 0xFFFF_FFFF_0000_0000 } else { v }
}

/// Arithmetic shift right, s in [0, 63] (caller enforces T4).
public fun sar(x: u64, s: u8): u64 {
    if (s == 0) { return x };
    if (is_neg(x)) {
        (x >> s) | (ALL_ONES << ((64 - (s as u16)) as u8)) // 64−s ∈ [1,63]
    } else {
        x >> s
    }
}

/// THE rounding rule (SPEC §5.1): arithmetic >> s, round-half-to-even.
public fun rnd(x: u64, s: u8): u64 {
    if (s == 0) { return x };
    let q = sar(x, s);
    // r = x − q·2^s ∈ [0, 2^s) — exact via wrapping arithmetic.
    let r = wsub(x, ((((q as u128) << s) & MASK64) as u64));
    let half = 1u64 << ((s - 1) as u8);
    if (r > half) {
        wadd(q, 1)
    } else if (r == half) {
        wadd(q, q & 1) // half rounds to even
    } else {
        q
    }
}

/// Truncating division toward zero; divisor is a POSITIVE i64 (T5 ensures).
public fun sdiv(a: u64, dv: u64): u64 {
    // |i64::MIN| wraps to itself = 2^63 as u64 — the magnitude is correct.
    let ma = if (is_neg(a)) { neg(a) } else { a };
    let q = ma / dv;
    if (is_neg(a)) { neg(q) } else { q }
}

/// Saturating i8 store value (bit pattern). Stores saturate, never wrap.
public fun sat8(x: u64): u8 {
    if (slt(x, neg(128))) {
        0x80 // −128
    } else if (sgt(x, 127)) {
        0x7F // 127
    } else {
        ((x & 0xFF) as u8)
    }
}

/// Saturating i16 value as a 16-bit pattern in the low bits.
public fun sat16(x: u64): u64 {
    if (slt(x, neg(32768))) {
        0x8000 // −32768
    } else if (sgt(x, 32767)) {
        0x7FFF // 32767
    } else {
        x & 0xFFFF
    }
}
