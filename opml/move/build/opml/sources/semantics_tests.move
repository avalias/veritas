/// In-Move semantics tests: the signed-arithmetic layer checked against
/// INDEPENDENT formulations (magnitude arithmetic on small u64s), plus the
/// SPEC §5.1 boundary tables. The Rust↔Move binding vectors live in
/// generated_vectors.move (emitted by client/gen_move_vectors).
#[test_only]
module opml::semantics_tests;

use opml::signed as sg;

/// Brief pitfall list: "test the mapping against Rust exhaustively for
/// MAC8". All 65 536 (a, b) i8 pairs: sext8/wmul/wadd two's-complement
/// result must equal the independently computed signed product (magnitudes
/// + explicit sign, no sext/wmul involved). Split into 16 blocks of 4 096
/// cases to stay inside the per-test gas budget — coverage stays
/// exhaustive: ∪ blocks = the full 256×256 grid.
fun mac8_block(a_lo: u64, a_hi: u64) {
    let mut a = a_lo;
    while (a < a_hi) {
        let mut b = 0u64;
        while (b < 256) {
            let got = sg::wmul(sg::sext8((a as u8)), sg::sext8((b as u8)));
            // Independent path: i8 value = x − 256 if x ≥ 128.
            let (ma, a_neg) = if (a >= 128) { (256 - a, true) } else { (a, false) };
            let (mb, b_neg) = if (b >= 128) { (256 - b, true) } else { (b, false) };
            let mag = ma * mb; // ≤ 16384, exact in u64
            let want = if (a_neg != b_neg) { sg::neg(mag) } else { mag };
            assert!(got == want, a * 256 + b);
            b = b + 1;
        };
        a = a + 1;
    };
}

#[test] fun mac8_exhaustive_00() { mac8_block(0, 16) }
#[test] fun mac8_exhaustive_01() { mac8_block(16, 32) }
#[test] fun mac8_exhaustive_02() { mac8_block(32, 48) }
#[test] fun mac8_exhaustive_03() { mac8_block(48, 64) }
#[test] fun mac8_exhaustive_04() { mac8_block(64, 80) }
#[test] fun mac8_exhaustive_05() { mac8_block(80, 96) }
#[test] fun mac8_exhaustive_06() { mac8_block(96, 112) }
#[test] fun mac8_exhaustive_07() { mac8_block(112, 128) }
#[test] fun mac8_exhaustive_08() { mac8_block(128, 144) }
#[test] fun mac8_exhaustive_09() { mac8_block(144, 160) }
#[test] fun mac8_exhaustive_10() { mac8_block(160, 176) }
#[test] fun mac8_exhaustive_11() { mac8_block(176, 192) }
#[test] fun mac8_exhaustive_12() { mac8_block(192, 208) }
#[test] fun mac8_exhaustive_13() { mac8_block(208, 224) }
#[test] fun mac8_exhaustive_14() { mac8_block(224, 240) }
#[test] fun mac8_exhaustive_15() { mac8_block(240, 256) }

/// SPEC §5.1 rounding boundary table, verbatim (conformance C-1).
#[test]
fun rnd_spec_boundary_table() {
    // (x, s, expected) with i64 values as two's-complement u64 patterns.
    assert!(sg::rnd(3, 1) == 2, 0);
    assert!(sg::rnd(5, 1) == 2, 1);
    assert!(sg::rnd(sg::neg(3), 1) == sg::neg(2), 2);
    assert!(sg::rnd(sg::neg(5), 1) == sg::neg(2), 3);
    assert!(sg::rnd(7, 2) == 2, 4);
    assert!(sg::rnd(sg::neg(7), 2) == sg::neg(2), 5);
    assert!(sg::rnd(sg::neg(1), 63) == 0, 6);
    assert!(sg::rnd(0, 17) == 0, 7);
    assert!(sg::rnd(0x7FFF_FFFF_FFFF_FFFF, 1) == (1u64 << 62), 8);
    assert!(sg::rnd(1u64 << 63, 1) == sg::neg(1u64 << 62), 9);
    assert!(sg::rnd(1u64 << 63, 63) == sg::neg(1), 10);
    assert!(sg::rnd(0x7FFF_FFFF_FFFF_FFFF, 63) == 1, 11);
    assert!(sg::rnd(42, 0) == 42, 12);
    assert!(sg::rnd(sg::neg(42), 0) == sg::neg(42), 13);
}

/// Saturation edges (conformance C-2): stores saturate, never wrap.
#[test]
fun saturation_edges() {
    assert!(sg::sat8(sg::neg(129)) == 0x80, 0); // −129 → −128
    assert!(sg::sat8(sg::neg(128)) == 0x80, 1);
    assert!(sg::sat8(127) == 0x7F, 2);
    assert!(sg::sat8(128) == 0x7F, 3); // 128 → 127
    assert!(sg::sat8(1u64 << 63) == 0x80, 4); // i64::MIN
    assert!(sg::sat8(0x7FFF_FFFF_FFFF_FFFF) == 0x7F, 5); // i64::MAX
    assert!(sg::sat16(sg::neg(32769)) == 0x8000, 6);
    assert!(sg::sat16(32768) == 0x7FFF, 7);
    assert!(sg::sat16(1u64 << 63) == 0x8000, 8);
    assert!(sg::sat16(sg::neg(5)) == 0xFFFB, 9); // in-range passthrough
}

/// Truncating division (conformance C-3): toward zero, never floor.
#[test]
fun sdiv_cases() {
    assert!(sg::sdiv(7, 2) == 3, 0);
    assert!(sg::sdiv(sg::neg(7), 2) == sg::neg(3), 1); // −3, NOT −4
    assert!(sg::sdiv(0, 5) == 0, 2);
    assert!(sg::sdiv(sg::neg(1), 2) == 0, 3);
    assert!(sg::sdiv(1u64 << 63, 1) == (1u64 << 63), 4); // i64::MIN / 1
    assert!(sg::sdiv(1u64 << 63, 2) == sg::neg(1u64 << 62), 5);
    assert!(sg::sdiv(0x7FFF_FFFF_FFFF_FFFF, 1) == 0x7FFF_FFFF_FFFF_FFFF, 6);
}

/// Signed compare and shift sanity at the extremes.
#[test]
fun slt_sar_extremes() {
    let min = 1u64 << 63;
    let max = 0x7FFF_FFFF_FFFF_FFFF;
    assert!(sg::slt(min, sg::neg(1)), 0);
    assert!(sg::slt(sg::neg(1), 0), 1);
    assert!(sg::slt(0, 1), 2);
    assert!(sg::slt(1, max), 3);
    assert!(!sg::slt(max, min), 4);
    assert!(sg::sar(min, 63) == sg::neg(1), 5);
    assert!(sg::sar(max, 63) == 0, 6);
    assert!(sg::sar(sg::neg(8), 2) == sg::neg(2), 7);
    assert!(sg::sar(sg::neg(1), 1) == sg::neg(1), 8); // −1 >> s stays −1
    assert!(sg::sext16(0x8000) == sg::neg(32768), 9);
    assert!(sg::sext32(0x8000_0000) == sg::neg(0x8000_0000), 10);
}
