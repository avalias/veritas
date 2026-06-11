//! Conformance tests C-1..C-6 + C-13 (SPEC §11), trap semantics (SPEC §4.4),
//! and codec roundtrips. These tests ARE the spec's enforcement arm — a
//! failure here is a consensus bug, not a style issue.

use vm::exec::{rnd, sat16, sat8, sext8, trunc_div, Machine, StepError, StepOutcome};
use vm::fixtures::{golden_machine, XorShift64};
use vm::hash::Hash;
use vm::isa::{Instr, Opcode, Operand};
use vm::merkle::{fold_proof, verify_inclusion, zero_page_subtrees, MerkleTree};
use vm::state::{CommittedMemory, Registers, HALTED, TRAPPED};
use vm::trace::{per_step_roots, trace_digest};
use vm::PAGE_SIZE;

// ---------------------------------------------------------------------------
// Pinned goldens — values printed by `cargo run -p vm --bin gen_goldens`.
// A change here is a deliberate consensus break (SPEC version bump).
// ---------------------------------------------------------------------------

const GOLDEN_Z: [&str; 5] = [
    "bf6f798d58af8aeb1f485fbfb8359bbebcd1a7f45188086d61d7c046055878b4",
    "5a830f5b44704190af76b564c3efcf972281939ca8c49217250df80e137e9a3b",
    "f7200364b87523d7772ac99e2ff7d01c6889f719162ea533a2d79874cf497ad5",
    "78bfc13dd605d67d8d72b41fb15e7c260a56b6e37523caf72078118254e3f4b4",
    "033f09a950acd0c15e44744a90c5717027d9a650e608ab17458058b6a22923d1",
];
const GOLDEN_ZERO_STATE_ROOT: &str =
    "1706f2ac780a40d0d3a4cc159c465a76155f03140980d8f6e866b9711dfacfd2";
const GOLDEN_TRACE_DIGEST: &str =
    "95da763b9d265fe1888bff55d36a9dfed56ac1405396d61dc43d800f7da6a0ae";
const GOLDEN_TRACE_STEPS: u64 = 41;
const GOLDEN_FINAL_ROOT: &str =
    "1f35d07706a53e0315ce209cfb73931692e55104768acbbedac03a558ecb79fc";

fn hex(h: &Hash) -> String {
    h.iter().map(|b| format!("{b:02x}")).collect()
}

// ---------------------------------------------------------------------------
// C-1: the rounding rule
// ---------------------------------------------------------------------------

/// Independent formulation of round-half-even via i128 Euclidean division.
fn rnd_ref(x: i64, s: u8) -> i64 {
    if s == 0 {
        return x;
    }
    let den = 1i128 << s;
    let v = x as i128;
    let q = v.div_euclid(den);
    let r = v.rem_euclid(den);
    let half = den / 2;
    let q = if r > half {
        q + 1
    } else if r == half {
        q + (q & 1)
    } else {
        q
    };
    q as i64
}

#[test]
fn c1_rnd_boundary_table() {
    // (x, s, expected) — exactly SPEC §5.1's table.
    let table: &[(i64, u8, i64)] = &[
        (3, 1, 2),
        (5, 1, 2),
        (-3, 1, -2),
        (-5, 1, -2),
        (7, 2, 2),
        (-7, 2, -2),
        (-1, 63, 0),
        (0, 17, 0),
        (i64::MAX, 1, 1 << 62), // 2^62 − 0.5 → even
        (i64::MIN, 1, -(1 << 62)),
        (i64::MIN, 63, -1),
        (i64::MAX, 63, 1), // (2^63−1)/2^63 → just under 1, r > half
        (42, 0, 42),
        (-42, 0, -42),
    ];
    for &(x, s, want) in table {
        assert_eq!(rnd(x, s), want, "rnd({x}, {s})");
    }
}

#[test]
fn c1_rnd_matches_reference() {
    let mut rng = XorShift64::new(1);
    for _ in 0..200_000 {
        let x = rng.next_u64() as i64;
        let s = (rng.next_u64() % 64) as u8;
        assert_eq!(rnd(x, s), rnd_ref(x, s), "rnd({x}, {s})");
    }
    // Half-point sweep: x = q·2^s ± half for small q, all s.
    for s in 1..=10u8 {
        let half = 1i64 << (s - 1);
        for q in -64..=64i64 {
            let x = (q << s) + half;
            assert_eq!(rnd(x, s), rnd_ref(x, s), "half point q={q} s={s}");
        }
    }
}

// ---------------------------------------------------------------------------
// C-2: saturation edges
// ---------------------------------------------------------------------------

#[test]
fn c2_saturation_edges() {
    assert_eq!(sat8(-129), -128);
    assert_eq!(sat8(-128), -128);
    assert_eq!(sat8(127), 127);
    assert_eq!(sat8(128), 127);
    assert_eq!(sat8(i64::MIN), -128);
    assert_eq!(sat8(i64::MAX), 127);
    assert_eq!(sat16(-32769), -32768);
    assert_eq!(sat16(32768), 32767);
    assert_eq!(sat16(i64::MIN), -32768);
    assert_eq!(sat16(i64::MAX), 32767);
    assert_eq!(sat16(-5), -5);
}

// ---------------------------------------------------------------------------
// C-3: truncating division
// ---------------------------------------------------------------------------

#[test]
fn c3_trunc_div_cases() {
    assert_eq!(trunc_div(7, 2), 3);
    assert_eq!(trunc_div(-7, 2), -3); // toward zero, NOT floor
    assert_eq!(trunc_div(0, 5), 0);
    assert_eq!(trunc_div(-1, 2), 0);
    assert_eq!(trunc_div(i64::MIN, 1), i64::MIN);
    assert_eq!(trunc_div(i64::MIN, 2), -(1 << 62));
    assert_eq!(trunc_div(i64::MAX, 1), i64::MAX);
}

// ---------------------------------------------------------------------------
// C-4: Merkle tree
// ---------------------------------------------------------------------------

#[test]
fn c4_zero_subtree_goldens() {
    let z = zero_page_subtrees(4);
    for (l, want) in GOLDEN_Z.iter().enumerate() {
        assert_eq!(hex(&z[l]), *want, "Z_{l}");
    }
}

#[test]
fn c4_inclusion_proofs_roundtrip_and_tamper() {
    let mut rng = XorShift64::new(7);
    let depth = 6u8;
    let leaves: Vec<Hash> = (0..40)
        .map(|_| {
            let mut h = [0u8; 32];
            rng.fill(&mut h);
            h
        })
        .collect();
    let pad = [0xAAu8; 32];
    let tree = MerkleTree::from_leaf_hashes(depth, leaves.clone(), pad);
    for i in 0..(1u64 << depth) {
        let leaf = tree.leaf_hash(i);
        let sibs = tree.prove(i);
        assert!(verify_inclusion(&tree.root(), leaf, i, &sibs), "idx {i}");
        // Wrong index must fail (position binding via fold order) — except
        // when the sibling slot holds the *identical* leaf (the pad region):
        // there "leaf at i^1" is a true statement and rightly verifies.
        if tree.leaf_hash(i ^ 1) != leaf {
            assert!(!verify_inclusion(&tree.root(), leaf, i ^ 1, &sibs));
        }
        // Tampered leaf must fail.
        let mut bad = leaf;
        bad[0] ^= 1;
        assert!(!verify_inclusion(&tree.root(), bad, i, &sibs));
        // Tampered sibling must fail.
        let mut bad_sibs = sibs.clone();
        bad_sibs[3][0] ^= 1;
        assert!(!verify_inclusion(&tree.root(), leaf, i, &bad_sibs));
    }
}

#[test]
fn c4_incremental_update_equals_rebuild() {
    let mut rng = XorShift64::new(11);
    let mut mem = CommittedMemory::new_zero(6);
    assert_eq!(mem.root(), mem.recompute_root_full());
    for _ in 0..200 {
        // Aligned writes of size 1/2/4 at random addresses.
        let size = [1usize, 2, 4][(rng.next_u64() % 3) as usize];
        let addr = (rng.next_u64() % (mem.mem_bytes() / size as u64)) * size as u64;
        let mut bytes = vec![0u8; size];
        rng.fill(&mut bytes);
        mem.write(addr, &bytes);
        assert_eq!(mem.root(), mem.recompute_root_full());
        assert_eq!(mem.read(addr, size), &bytes[..]);
    }
    // fold_proof recomputes the same root the tree maintains.
    let page9 = *mem.page(9);
    let sibs = mem.prove_page(9);
    assert_eq!(
        fold_proof(vm::hash::page_leaf_hash(&page9), 9, &sibs),
        mem.root()
    );
}

// ---------------------------------------------------------------------------
// C-5: per-op semantic vectors
// ---------------------------------------------------------------------------

/// Tiny machine: d=8 (LUT16's full index range needs 196 KiB), program given.
fn mk(prog: Vec<Instr>) -> Machine {
    Machine::new(8, 4, prog)
}

fn step_ok(m: &mut Machine) -> StepOutcome {
    m.step().expect("not terminal")
}

#[test]
fn c5_mac8_mac16_sign() {
    let mut m = mk(vec![
        Instr { a: Operand::at(0), b: Operand::at(1), ..Instr::op(Opcode::Mac8) },
        Instr { a: Operand::at(2), b: Operand::at(4), ..Instr::op(Opcode::Mac16) },
    ]);
    m.mem.write(0, &[(-3i8) as u8]);
    m.mem.write(1, &[5u8]);
    m.mem.write(2, &(-300i16).to_le_bytes());
    m.mem.write(4, &(100i16).to_le_bytes());
    step_ok(&mut m);
    assert_eq!(m.regs.acc, -15);
    assert_eq!((m.regs.pc, m.regs.step), (1, 1));
    step_ok(&mut m);
    assert_eq!(m.regs.acc, -15 - 30_000);
}

#[test]
fn c5_loads_and_ldc_sign_extend() {
    let mut m = mk(vec![
        Instr { a: Operand::at(0), ..Instr::op(Opcode::Ld8) },
        Instr { a: Operand::at(4), ..Instr::op(Opcode::Ld32) },
        Instr { imm: (-5i32) as u32, ..Instr::op(Opcode::Ldc) },
    ]);
    m.mem.write(0, &[0x80]); // -128
    m.mem.write(4, &(-7i32).to_le_bytes());
    step_ok(&mut m);
    assert_eq!(m.regs.acc, -128);
    step_ok(&mut m);
    assert_eq!(m.regs.acc, -7);
    step_ok(&mut m);
    assert_eq!(m.regs.acc, -5);
}

#[test]
fn c5_add32_mul32_wrap_and_div32() {
    let mut m = mk(vec![
        Instr { a: Operand::at(0), ..Instr::op(Opcode::Add32) },
        Instr { a: Operand::at(4), ..Instr::op(Opcode::Mul32) },
        Instr { a: Operand::at(8), ..Instr::op(Opcode::Div32) },
    ]);
    m.mem.write(0, &(-100i32).to_le_bytes());
    m.mem.write(4, &(2i32).to_le_bytes());
    m.mem.write(8, &(7i32).to_le_bytes());
    step_ok(&mut m);
    assert_eq!(m.regs.acc, -100);
    // Wrapping is normative semantics: force an i64 overflow via MUL32.
    m.regs.acc = i64::MAX;
    step_ok(&mut m);
    assert_eq!(m.regs.acc, i64::MAX.wrapping_mul(2)); // = -2
    m.regs.acc = -23;
    step_ok(&mut m);
    assert_eq!(m.regs.acc, -3); // trunc toward zero
}

#[test]
fn c5_shift_clamp_store() {
    let mut m = mk(vec![
        Instr { s: 2, ..Instr::op(Opcode::ShiftRndn) },
        Instr { w: Operand::at(64), ..Instr::op(Opcode::Clamp8) },
        Instr { w: Operand::at(66), ..Instr::op(Opcode::Clamp16) },
        Instr { k: 0, w: Operand::at(68), ..Instr::op(Opcode::St32) },
    ]);
    m.regs.acc = -1399; // -1399/4 = -349.75 → -350
    step_ok(&mut m);
    assert_eq!(m.regs.acc, -350);
    step_ok(&mut m);
    assert_eq!(m.mem.read_u8(64) as i8, -128); // saturated
    assert_eq!(m.regs.acc, -350, "CLAMP8 must not change acc");
    step_ok(&mut m);
    assert_eq!(m.mem.read_u16(66) as i16, -350);
    step_ok(&mut m);
    assert_eq!(m.mem.read_u32(68), (-350i64) as u32); // low32 truncation
}

#[test]
fn c5_st32_selectors() {
    let mut prog = vec![];
    for k in 0..6u8 {
        prog.push(Instr { k, w: Operand::at(64 + 4 * k as u64), ..Instr::op(Opcode::St32) });
    }
    let mut m = mk(prog);
    m.regs.acc = -2;
    m.regs.aux = 0x1_2345_6789; // low32 = 0x2345_6789
    m.regs.idx = [10, 11, 12, 13];
    for _ in 0..6 {
        step_ok(&mut m);
    }
    assert_eq!(m.mem.read_u32(64), (-2i32) as u32);
    assert_eq!(m.mem.read_u32(68), 0x2345_6789);
    for j in 0..4u64 {
        assert_eq!(m.mem.read_u32(72 + 4 * j), 10 + j as u32);
    }
}

#[test]
fn c5_lut16_edges() {
    // Table base 1024; index = sat16(acc) + 32768.
    let mut m = mk(vec![
        Instr { a: Operand::at(1024), ..Instr::op(Opcode::Lut16) },
        Instr { a: Operand::at(1024), ..Instr::op(Opcode::Lut16) },
    ]);
    m.mem.write(1024, &(-1234i16).to_le_bytes()); // entry for t = -32768
    m.mem.write(1024 + 131070, &(4321i16).to_le_bytes()); // entry for t = +32767
    m.regs.acc = i64::MIN; // saturates to -32768 → index 0
    step_ok(&mut m);
    assert_eq!(m.regs.acc, -1234);
    m.regs.acc = 9_999_999; // saturates to +32767 → last entry
    step_ok(&mut m);
    assert_eq!(m.regs.acc, 4321);
}

#[test]
fn c5_ldidx_argmax_tie_break() {
    let mut m = mk(vec![
        Instr { k: 2, a: Operand::at(0), ..Instr::op(Opcode::Ldidx) },
        // Scan three i32s at 64,68,72 with idx3 as the position register.
        Instr { k: 3, a: { let mut o = Operand::at(64); o.stride[3] = 4; o }, ..Instr::op(Opcode::ArgmaxStep) },
        Instr { k: 3, target: 1, imm: 3, ..Instr::op(Opcode::Loop) },
    ]);
    m.mem.write(0, &77u32.to_le_bytes());
    // values: 5, 9, 9 — max 9 first at index 1; tie at index 2 must lose.
    m.mem.write(64, &5i32.to_le_bytes());
    m.mem.write(68, &9i32.to_le_bytes());
    m.mem.write(72, &9i32.to_le_bytes());
    m.regs.acc = i64::MIN;
    step_ok(&mut m); // LDIDX
    assert_eq!(m.regs.idx[2], 77);
    for _ in 0..6 {
        step_ok(&mut m); // 3 × (ARGMAX, LOOP)
    }
    assert_eq!(m.regs.acc, 9);
    assert_eq!(m.regs.aux, 1, "ties must break to the lowest index");
    assert_eq!(m.regs.idx[3], 0, "LOOP auto-resets its counter");
}

#[test]
fn c5_argmax_off_global_index() {
    // Chunked-head pattern (SPEC §5.2): scan with imm = chunk·N records a
    // GLOBAL row index while idx[k] stays chunk-local.
    let mut m = mk(vec![
        Instr { k: 1, imm: 512, a: { let mut o = Operand::at(64); o.stride[1] = 4; o }, ..Instr::op(Opcode::ArgmaxOff) },
        Instr { k: 1, target: 0, imm: 3, ..Instr::op(Opcode::Loop) },
    ]);
    m.mem.write(64, &5i32.to_le_bytes());
    m.mem.write(68, &9i32.to_le_bytes());
    m.mem.write(72, &9i32.to_le_bytes()); // tie loses to the earlier 9
    m.regs.acc = i64::MIN;
    for _ in 0..6 {
        step_ok(&mut m);
    }
    assert_eq!(m.regs.acc, 9);
    assert_eq!(m.regs.aux, 512 + 1, "aux = imm + winning idx");
    // No-update path leaves aux untouched.
    let mut m = mk(vec![Instr { k: 0, imm: 99, a: Operand::at(64), ..Instr::op(Opcode::ArgmaxOff) }]);
    m.mem.write(64, &(-3i32).to_le_bytes());
    m.regs.acc = 100;
    m.regs.aux = 7;
    step_ok(&mut m);
    assert_eq!((m.regs.acc, m.regs.aux), (100, 7));
    // T6: k out of range traps exactly like ARGMAX_STEP.
    assert_traps(mk(vec![Instr { k: 4, a: Operand::at(0), ..Instr::op(Opcode::ArgmaxOff) }]));
}

#[test]
fn c5_control_flow() {
    let mut m = mk(vec![
        Instr { target: 3, ..Instr::op(Opcode::Jmp) },
        Instr::op(Opcode::Halt), // skipped
        Instr::op(Opcode::Halt), // skipped
        Instr { a: Operand::at(0), imm: 42, target: 6, ..Instr::op(Opcode::Jeq) }, // taken
        Instr::op(Opcode::Halt), // skipped
        Instr::op(Opcode::Halt), // skipped
        Instr { a: Operand::at(0), imm: 43, target: 0, ..Instr::op(Opcode::Jeq) }, // not taken
        Instr::op(Opcode::Halt),
    ]);
    m.mem.write(0, &42u32.to_le_bytes());
    assert_eq!(step_ok(&mut m), StepOutcome::Ran);
    assert_eq!(m.regs.pc, 3);
    step_ok(&mut m);
    assert_eq!(m.regs.pc, 6);
    step_ok(&mut m);
    assert_eq!(m.regs.pc, 7);
    let pc_before = m.regs.pc;
    assert_eq!(step_ok(&mut m), StepOutcome::Halted);
    assert_eq!(m.regs.pc, pc_before, "HALT leaves pc unchanged");
    assert_eq!(m.regs.halted, HALTED);
    assert_eq!(m.step(), Err(StepError::AlreadyTerminal));
}

#[test]
fn c5_loop_iteration_count() {
    // Body = slot 0 (LDC), LOOP at slot 1: body must run exactly imm times.
    let mut m = mk(vec![
        Instr { imm: 1, ..Instr::op(Opcode::Ldc) },
        Instr { k: 0, target: 0, imm: 5, ..Instr::op(Opcode::Loop) },
        Instr::op(Opcode::Halt),
    ]);
    let mut body_runs = 0;
    loop {
        match step_ok(&mut m) {
            StepOutcome::Halted => break,
            _ => {
                if m.regs.pc == 1 {
                    body_runs += 1; // just executed slot 0
                }
            }
        }
    }
    assert_eq!(body_runs, 5);
    assert_eq!(m.regs.idx[0], 0, "auto-reset on exit");
}

// ---------------------------------------------------------------------------
// Trap semantics (SPEC §4.4): frozen state, step+1, memory untouched
// ---------------------------------------------------------------------------

fn assert_traps(mut m: Machine) {
    let before = m.regs;
    let root_before = m.mem.root();
    let out = m.step().expect("not terminal");
    assert_eq!(out, StepOutcome::Trapped);
    assert_eq!(m.regs.halted, TRAPPED);
    assert_eq!(m.regs.step, before.step + 1);
    assert_eq!(m.regs.pc, before.pc, "pc frozen on trap");
    assert_eq!(m.regs.acc, before.acc);
    assert_eq!(m.regs.aux, before.aux);
    assert_eq!(m.regs.idx, before.idx);
    assert_eq!(m.mem.root(), root_before, "memory frozen on trap");
    assert_eq!(m.step(), Err(StepError::AlreadyTerminal));
}

#[test]
fn traps_t1_t2() {
    // T1: pc beyond 2^p. p=4 ⇒ JMP 16 then fetch traps.
    let mut m = mk(vec![Instr { target: 16, ..Instr::op(Opcode::Jmp) }]);
    step_ok(&mut m);
    assert_traps(m);
    // T2: padding (zero instruction) after the program end.
    let mut m = mk(vec![Instr { imm: 3, ..Instr::op(Opcode::Ldc) }]);
    step_ok(&mut m);
    assert_traps(m);
}

#[test]
fn traps_t3_bounds_and_alignment() {
    let mem_bytes = (1u64 << 8) * PAGE_SIZE as u64;
    // Misaligned LD32.
    assert_traps(mk(vec![Instr { a: Operand::at(2), ..Instr::op(Opcode::Ld32) }]));
    // Misaligned MAC16 (odd address).
    assert_traps(mk(vec![Instr { a: Operand::at(3), b: Operand::at(4), ..Instr::op(Opcode::Mac16) }]));
    // Out-of-bounds MAC8 via second operand.
    assert_traps(mk(vec![Instr { a: Operand::at(0), b: Operand::at(mem_bytes), ..Instr::op(Opcode::Mac8) }]));
    // ea computed through idx·stride wrapping must still bounds-check.
    let mut m = mk(vec![Instr {
        a: { let mut o = Operand::at(0); o.stride[0] = u32::MAX; o },
        ..Instr::op(Opcode::Ld32)
    }]);
    m.regs.idx[0] = u32::MAX; // ea = (2^32-1)^2 mod 2^64 — far out of bounds
    assert_traps(m);
}

#[test]
fn traps_t4_t5_t6() {
    // T4: shift amount > 63.
    assert_traps(mk(vec![Instr { s: 64, ..Instr::op(Opcode::ShiftRndn) }]));
    // T5: divisor 0 and negative.
    for dv in [0i32, -3] {
        let mut m = mk(vec![Instr { a: Operand::at(0), ..Instr::op(Opcode::Div32) }]);
        m.mem.write(0, &dv.to_le_bytes());
        m.regs.acc = 100;
        assert_traps(m);
    }
    // T6: selector out of range.
    assert_traps(mk(vec![Instr { k: 6, w: Operand::at(0), ..Instr::op(Opcode::St32) }]));
    assert_traps(mk(vec![Instr { k: 4, a: Operand::at(0), ..Instr::op(Opcode::Ldidx) }]));
    assert_traps(mk(vec![Instr { k: 4, a: Operand::at(0), ..Instr::op(Opcode::ArgmaxStep) }]));
    assert_traps(mk(vec![Instr { k: 4, target: 0, imm: 1, ..Instr::op(Opcode::Loop) }]));
}

#[test]
fn traps_t7_dot() {
    let mem_bytes = (1u64 << 8) * PAGE_SIZE as u64;
    let dot = |imm: u32, a: u64, b: u64| Instr {
        imm,
        a: Operand::at(a),
        b: Operand::at(b),
        ..Instr::op(Opcode::Dot8)
    };
    assert_traps(mk(vec![dot(0, 0, 64)])); // zero lanes
    assert_traps(mk(vec![dot(65, 0, 64)])); // over cap
    // DOT16's cap is 32: imm = 33 is valid for DOT8 but traps for DOT16.
    let mut m = mk(vec![dot(33, 0, 64)]);
    m.program[0].opcode = Opcode::Dot16 as u8;
    assert_traps(m);
    assert_traps(mk(vec![dot(64, 32, 64)])); // 32 is not 64-aligned
    assert_traps(mk(vec![dot(64, 0, mem_bytes)])); // line out of bounds
    // Positive control: last valid line is fine.
    let mut m = mk(vec![dot(64, mem_bytes - 64, mem_bytes - 64)]);
    assert_eq!(step_ok(&mut m), StepOutcome::Ran);
}

// ---------------------------------------------------------------------------
// C-13: DOT ≡ MAC chain
// ---------------------------------------------------------------------------

#[test]
fn c13_dot8_equals_mac8_chain() {
    let mut rng = XorShift64::new(13);
    for trial in 0..50 {
        let lanes = 1 + (rng.next_u64() % 64) as u32;
        let mut line_a = [0u8; 64];
        let mut line_b = [0u8; 64];
        rng.fill(&mut line_a);
        rng.fill(&mut line_b);
        let acc0 = rng.next_u64() as i64;

        // Machine 1: one DOT8.
        let mut m1 = mk(vec![Instr {
            imm: lanes,
            a: Operand::at(0),
            b: Operand::at(64),
            ..Instr::op(Opcode::Dot8)
        }]);
        m1.mem.write(0, &line_a);
        m1.mem.write(64, &line_b);
        m1.regs.acc = acc0;
        step_ok(&mut m1);

        // Machine 2: `lanes` scalar MAC8s.
        let prog: Vec<Instr> = (0..lanes as u64)
            .map(|j| Instr { a: Operand::at(j), b: Operand::at(64 + j), ..Instr::op(Opcode::Mac8) })
            .collect();
        let mut m2 = Machine::new(8, 7, prog);
        m2.mem.write(0, &line_a);
        m2.mem.write(64, &line_b);
        m2.regs.acc = acc0;
        for _ in 0..lanes {
            step_ok(&mut m2);
        }

        // Direct sum as the third witness.
        let mut want = acc0;
        for j in 0..lanes as usize {
            want = want.wrapping_add(sext8(line_a[j]).wrapping_mul(sext8(line_b[j])));
        }
        assert_eq!(m1.regs.acc, want, "trial {trial}");
        assert_eq!(m2.regs.acc, want, "trial {trial}");
    }
}

#[test]
fn c13_dot16_equals_direct_sum_and_aliasing() {
    let mut rng = XorShift64::new(17);
    for _ in 0..50 {
        let lanes = 1 + (rng.next_u64() % 32) as u32;
        let mut line = [0u8; 64];
        rng.fill(&mut line);
        // A = B aliasing (legal: sum of squares, SPEC §6.4).
        let mut m = mk(vec![Instr {
            imm: lanes,
            a: Operand::at(128),
            b: Operand::at(128),
            ..Instr::op(Opcode::Dot16)
        }]);
        m.mem.write(128, &line);
        step_ok(&mut m);
        let mut want = 0i64;
        for j in 0..lanes as usize {
            let v = i16::from_le_bytes([line[2 * j], line[2 * j + 1]]) as i64;
            want = want.wrapping_add(v.wrapping_mul(v));
        }
        assert_eq!(m.regs.acc, want);
        assert!(m.regs.acc >= 0, "sum of squares");
    }
}

// ---------------------------------------------------------------------------
// Wide ops (SPEC 0.4.0): LD16, DOT8X16, DOTBM
// ---------------------------------------------------------------------------

#[test]
fn ld16_sign_extends() {
    for (bytes, want) in [([0x34u8, 0x12], 0x1234i64), ([0x00, 0x80], -32768), ([0xff, 0xff], -1)]
    {
        let mut m = mk(vec![Instr { a: Operand::at(10), ..Instr::op(Opcode::Ld16) }]);
        m.mem.write(10, &bytes);
        m.regs.acc = 777; // overwritten, not accumulated
        step_ok(&mut m);
        assert_eq!(m.regs.acc, want);
    }
    // T3: misaligned.
    assert_traps(mk(vec![Instr { a: Operand::at(11), ..Instr::op(Opcode::Ld16) }]));
}

#[test]
fn c13_dot8x16_equals_direct_sum() {
    let mut rng = XorShift64::new(23);
    for trial in 0..50 {
        let lanes = 1 + (rng.next_u64() % 64) as u32;
        let mut wline = [0u8; 64];
        let mut xline = [0u8; 128];
        rng.fill(&mut wline);
        rng.fill(&mut xline);
        let acc0 = rng.next_u64() as i64;
        let mut m = mk(vec![Instr {
            imm: lanes,
            a: Operand::at(0),
            b: Operand::at(128),
            ..Instr::op(Opcode::Dot8x16)
        }]);
        m.mem.write(0, &wline);
        m.mem.write(128, &xline);
        m.regs.acc = acc0;
        step_ok(&mut m);
        let mut want = acc0;
        for j in 0..lanes as usize {
            let wv = sext8(wline[j]);
            let xv = i16::from_le_bytes([xline[2 * j], xline[2 * j + 1]]) as i64;
            want = want.wrapping_add(wv.wrapping_mul(xv));
        }
        assert_eq!(m.regs.acc, want, "trial {trial}");
    }
}

#[test]
fn c13_dotbm_equals_dot_times_multiplier() {
    let mut rng = XorShift64::new(29);
    for trial in 0..50 {
        let lanes = 1 + (rng.next_u64() % 64) as u32;
        let mut wline = [0u8; 64];
        let mut xline = [0u8; 128];
        rng.fill(&mut wline);
        rng.fill(&mut xline);
        let mult = rng.next_u64() as i32;
        let acc0 = rng.next_u64() as i64;

        // Machine 1: DOTBM (W slot reads the multiplier cell).
        let mut m1 = mk(vec![Instr {
            imm: lanes,
            a: Operand::at(0),
            b: Operand::at(128),
            w: Operand::at(256),
            ..Instr::op(Opcode::Dotbm)
        }]);
        m1.mem.write(0, &wline);
        m1.mem.write(128, &xline);
        m1.mem.write(256, &mult.to_le_bytes());
        m1.regs.acc = acc0;
        step_ok(&mut m1);

        // Witness: fresh-partial dot, then acc += p · m.
        let mut p = 0i64;
        for j in 0..lanes as usize {
            let wv = sext8(wline[j]);
            let xv = i16::from_le_bytes([xline[2 * j], xline[2 * j + 1]]) as i64;
            p = p.wrapping_add(wv.wrapping_mul(xv));
        }
        let want = acc0.wrapping_add(p.wrapping_mul(mult as i64));
        assert_eq!(m1.regs.acc, want, "trial {trial}");
    }
}

#[test]
fn traps_t7_wide() {
    let mem_bytes = (1u64 << 8) * PAGE_SIZE as u64;
    let wide = |imm: u32, a: u64, b: u64| Instr {
        imm,
        a: Operand::at(a),
        b: Operand::at(b),
        w: Operand::at(256),
        ..Instr::op(Opcode::Dot8x16)
    };
    assert_traps(mk(vec![wide(0, 0, 128)])); // zero lanes
    assert_traps(mk(vec![wide(65, 0, 128)])); // over cap
    assert_traps(mk(vec![wide(64, 32, 128)])); // A not 64-aligned
    assert_traps(mk(vec![wide(64, 0, 64)])); // B 64- but not 128-aligned
    assert_traps(mk(vec![wide(64, 0, mem_bytes - 64)])); // B line out of bounds
    // DOTBM extra: unaligned multiplier cell traps (T3 on the W READ).
    let mut m = mk(vec![Instr { w: Operand::at(257), ..wide(64, 0, 128) }]);
    m.program[0].opcode = Opcode::Dotbm as u8;
    assert_traps(m);
    // Positive control: last valid wide line.
    let mut m = mk(vec![wide(64, mem_bytes - 64, mem_bytes - 128)]);
    assert_eq!(step_ok(&mut m), StepOutcome::Ran);
}


// ---------------------------------------------------------------------------
// Float ops (SPEC FW-6): FDOT, FOP
// ---------------------------------------------------------------------------

#[test]
fn fdot_accumulates_committed_block() {
    use vm::softfloat::{block_dot_bf16, fadd};
    let mut rng = XorShift64::new(0xF0D7);
    for trial in 0..30 {
        let mut w16 = [0u16; 64];
        let mut x32 = [0u32; 64];
        for j in 0..64 {
            // finite, moderate values: bf16 weights, f32 activations
            w16[j] = ((rng.next_u64() as u32 & 0x3F7F_FFFF) >> 16) as u16;
            x32[j] = rng.next_u64() as u32 & 0x3F7F_FFFF;
        }
        let cell0 = rng.next_u64() as u32 & 0x3F7F_FFFF;
        let mut m = mk(vec![Instr {
            imm: 64,
            a: Operand::at(0),
            b: Operand::at(256),
            w: Operand::at(512),
            ..Instr::op(Opcode::Fdot)
        }]);
        let wb: Vec<u8> = w16.iter().flat_map(|v| v.to_le_bytes()).collect();
        let xb: Vec<u8> = x32.iter().flat_map(|v| v.to_le_bytes()).collect();
        m.mem.write(0, &wb);
        m.mem.write(256, &xb);
        m.mem.write(512, &cell0.to_le_bytes());
        step_ok(&mut m);
        let want = fadd(cell0, block_dot_bf16(&w16, &x32));
        assert_eq!(m.mem.read_u32(512), want, "trial {trial}");
    }
}

#[test]
fn fop_all_selectors_and_traps() {
    use vm::softfloat as sf;
    let a_bits = 0x4048_F5C3u32; // 3.14
    let b_bits = 0x4015_5555u32; // 2.333
    let c_bits = 0xBF80_0000u32; // -1.0
    for k in 0u8..8 {
        let mut m = mk(vec![Instr {
            k,
            a: Operand::at(0),
            b: Operand::at(4),
            w: Operand::at(8),
            ..Instr::op(Opcode::Fop)
        }]);
        m.mem.write(0, &a_bits.to_le_bytes());
        m.mem.write(4, &b_bits.to_le_bytes());
        m.mem.write(8, &c_bits.to_le_bytes());
        step_ok(&mut m);
        let want = match k {
            0 => sf::fadd(a_bits, b_bits),
            1 => sf::fmul(a_bits, b_bits),
            2 => sf::ffma(a_bits, b_bits, c_bits),
            3 => sf::fdiv(a_bits, b_bits),
            4 => sf::fsqrt(a_bits),
            5 => sf::ffloor(a_bits),
            6 => sf::ftoi(a_bits) as u32,
            _ => sf::itof(a_bits as i32),
        };
        assert_eq!(m.mem.read_u32(8), want, "k={k}");
    }
    // T6: selector out of range; T7: FDOT wrong imm; T3: misalignment.
    assert_traps(mk(vec![Instr { k: 8, a: Operand::at(0), b: Operand::at(4), w: Operand::at(8), ..Instr::op(Opcode::Fop) }]));
    assert_traps(mk(vec![Instr { imm: 32, a: Operand::at(0), b: Operand::at(256), w: Operand::at(512), ..Instr::op(Opcode::Fdot) }]));
    assert_traps(mk(vec![Instr { imm: 64, a: Operand::at(64), b: Operand::at(256), w: Operand::at(512), ..Instr::op(Opcode::Fdot) }]));
    assert_traps(mk(vec![Instr { imm: 64, a: Operand::at(0), b: Operand::at(128), w: Operand::at(512), ..Instr::op(Opcode::Fdot) }]));
}

// ---------------------------------------------------------------------------
// Codec roundtrips
// ---------------------------------------------------------------------------

#[test]
fn instr_codec_roundtrip_fuzz() {
    fn rand_operand(rng: &mut XorShift64) -> Operand {
        let mut stride = [0u32; 4];
        for s in stride.iter_mut() {
            *s = rng.next_u64() as u32;
        }
        Operand { base: rng.next_u64(), stride }
    }
    let mut rng = XorShift64::new(19);
    for _ in 0..10_000 {
        let i = Instr {
            opcode: rng.next_u64() as u8, // raw — including invalid opcodes
            k: rng.next_u64() as u8,
            s: rng.next_u64() as u8,
            imm: rng.next_u64() as u32,
            target: rng.next_u64() as u32,
            a: rand_operand(&mut rng),
            b: rand_operand(&mut rng),
            w: rand_operand(&mut rng),
        };
        assert_eq!(Instr::decode(&i.encode()), i);
    }
    assert_eq!(Instr::zero().encode(), [0u8; 96]);
}

#[test]
fn registers_codec_roundtrip_fuzz() {
    let mut rng = XorShift64::new(23);
    for _ in 0..10_000 {
        let r = Registers {
            pc: rng.next_u64() as u32,
            halted: (rng.next_u64() % 3) as u8,
            step: rng.next_u64(),
            acc: rng.next_u64() as i64,
            aux: rng.next_u64() as i64,
            idx: [
                rng.next_u64() as u32,
                rng.next_u64() as u32,
                rng.next_u64() as u32,
                rng.next_u64() as u32,
            ],
        };
        assert_eq!(Registers::decode(&r.encode()), r);
    }
    assert_eq!(Registers::default().encode(), [0u8; 45]);
}

// ---------------------------------------------------------------------------
// C-6: golden trace digest (Invariant 1)
// ---------------------------------------------------------------------------

#[test]
fn c6_zero_machine_state_root_golden() {
    let m = Machine::new(2, 2, vec![]);
    assert_eq!(hex(&m.state_root()), GOLDEN_ZERO_STATE_ROOT);
}

#[test]
fn c6_golden_trace_digest() {
    let mut m = golden_machine();
    let (digest, result) = trace_digest(&mut m, 10_000).expect("terminates");
    assert_eq!(result.outcome, StepOutcome::Halted, "golden run must HALT cleanly");
    assert_eq!(result.steps, GOLDEN_TRACE_STEPS);
    assert_eq!(hex(&digest), GOLDEN_TRACE_DIGEST);
    assert_eq!(hex(&result.final_root), GOLDEN_FINAL_ROOT);
}

#[test]
fn c6_trace_digest_reproducible_and_consistent() {
    // Same machine, two constructions → identical digests (determinism),
    // and the streaming digest equals H(concat per-step roots).
    let mut m1 = golden_machine();
    let mut m2 = golden_machine();
    assert_eq!(m1.state_root(), m2.state_root(), "genesis roots match");
    let (d1, r1) = trace_digest(&mut m1, 10_000).unwrap();
    let (roots, r2) = per_step_roots(&mut m2, 10_000).unwrap();
    assert_eq!(r1.steps, r2.steps);
    assert_eq!(roots.len() as u64, r1.steps + 1, "root_0 ..= root_N");
    use sha3::{Digest, Sha3_256};
    let mut h = Sha3_256::new();
    for r in &roots {
        h.update(r);
    }
    let concat: Hash = h.finalize().into();
    assert_eq!(d1, concat);
    // Every state root is distinct step to step (step counter advances).
    for w in roots.windows(2) {
        assert_ne!(w[0], w[1]);
    }
}
