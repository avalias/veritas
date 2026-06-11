//! Emits dispute/tests/generated_vectors.move — the Rust↔Move equivalence
//! suite (brief Phase 2.3, "the most important test in the project").
//!
//! Two layers:
//!  1. Signed-arithmetic vectors: random + boundary inputs with expected
//!     outputs computed by THE RUST IMPLEMENTATION (vm::exec), emitted as
//!     u64 two's-complement bit patterns. `sui move test` then holds
//!     dispute::signed to the same answers.
//!  2. Full verify_step vectors: real StepProofs built by vm::onestep
//!     against real machines, with the honest post-root (and tampered
//!     variants) — exercising V1–V9 end to end inside the Move VM.
//!
//! Usage: cargo run -p client --bin gen_move_vectors   (writes the file)

use std::fmt::Write as _;
use vm::exec::{rnd, sat16, sat8, sext16, sext32, sext8, trunc_div, Machine};
use vm::fixtures::XorShift64;
use vm::hash::Hash;
use vm::isa::{Instr, Opcode, Operand};
use vm::onestep::{build_step_proof, JudgeParams, ProgramTree, StepProof};
use vm::PAGE_SIZE;

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn u64s(vals: &[u64]) -> String {
    vals.iter().map(|v| format!("{v}")).collect::<Vec<_>>().join(", ")
}

fn u8s(vals: &[u8]) -> String {
    vals.iter().map(|v| format!("{v}")).collect::<Vec<_>>().join(", ")
}

// ---------------------------------------------------------------------------
// Layer 1: signed-arithmetic vectors
// ---------------------------------------------------------------------------

/// Interesting i64 values + randoms — every boundary the helpers care about.
fn interesting_i64(rng: &mut XorShift64, n: usize) -> Vec<i64> {
    let mut v: Vec<i64> = vec![
        0, 1, -1, 2, -2, 127, 128, -128, -129, 255, 256, 32767, 32768, -32768, -32769, 65535,
        65536, i32::MAX as i64, i32::MIN as i64, i64::MAX, i64::MIN, i64::MAX - 1, i64::MIN + 1,
    ];
    while v.len() < n {
        v.push(match rng.next_u64() % 4 {
            0 => rng.next_u64() as i64,
            1 => (rng.next_u64() % 65536) as i64 - 32768,
            2 => (rng.next_u64() as i64) >> (rng.next_u64() % 56),
            _ => 1i64 << (rng.next_u64() % 63),
        });
    }
    v
}

fn emit_signed_vectors(out: &mut String, rng: &mut XorShift64) {
    let xs = interesting_i64(rng, 160);
    let ys = interesting_i64(rng, 160);

    // rnd(x, s) — THE rounding rule.
    let ss: Vec<u8> = xs.iter().map(|_| (rng.next_u64() % 64) as u8).collect();
    let want: Vec<u64> = xs.iter().zip(&ss).map(|(&x, &s)| rnd(x, s) as u64).collect();
    write!(
        out,
        r#"
#[test]
fun rust_rnd_vectors() {{
    let xs: vector<u64> = vector[{}];
    let ss: vector<u8> = vector[{}];
    let want: vector<u64> = vector[{}];
    let mut i = 0u64;
    while (i < xs.length()) {{
        assert!(sg::rnd(xs[i], ss[i]) == want[i], i);
        i = i + 1;
    }};
}}
"#,
        u64s(&xs.iter().map(|&x| x as u64).collect::<Vec<_>>()),
        u8s(&ss),
        u64s(&want)
    )
    .unwrap();

    // wadd / wmul / slt / sar / sdiv / sat / sext — one combined sweep.
    let mut wadd_w = vec![];
    let mut wmul_w = vec![];
    let mut slt_w = vec![];
    let mut sar_s = vec![];
    let mut sar_w = vec![];
    let mut sdiv_d = vec![];
    let mut sdiv_w = vec![];
    let mut sat8_w = vec![];
    let mut sat16_w = vec![];
    for (i, (&x, &y)) in xs.iter().zip(&ys).enumerate() {
        wadd_w.push(x.wrapping_add(y) as u64);
        wmul_w.push(x.wrapping_mul(y) as u64);
        slt_w.push(u8::from(x < y));
        let s = (i % 64) as u8;
        sar_s.push(s);
        sar_w.push((x >> s) as u64);
        let d = (y.unsigned_abs() % 1_000_000 + 1) as i64; // positive divisor
        sdiv_d.push(d as u64);
        sdiv_w.push(trunc_div(x, d) as u64);
        sat8_w.push(sat8(x) as u8);
        sat16_w.push(sat16(x) as u16 as u64);
    }
    write!(
        out,
        r#"
#[test]
fun rust_signed_core_vectors() {{
    let xs: vector<u64> = vector[{xs}];
    let ys: vector<u64> = vector[{ys}];
    let wadd_w: vector<u64> = vector[{wadd}];
    let wmul_w: vector<u64> = vector[{wmul}];
    let slt_w: vector<u8> = vector[{slt}];
    let sar_s: vector<u8> = vector[{sars}];
    let sar_w: vector<u64> = vector[{sarw}];
    let sdiv_d: vector<u64> = vector[{sdivd}];
    let sdiv_w: vector<u64> = vector[{sdivw}];
    let sat8_w: vector<u8> = vector[{sat8w}];
    let sat16_w: vector<u64> = vector[{sat16w}];
    let mut i = 0u64;
    while (i < xs.length()) {{
        assert!(sg::wadd(xs[i], ys[i]) == wadd_w[i], i);
        assert!(sg::wmul(xs[i], ys[i]) == wmul_w[i], 1000 + i);
        assert!(sg::slt(xs[i], ys[i]) == (slt_w[i] == 1), 2000 + i);
        assert!(sg::sar(xs[i], sar_s[i]) == sar_w[i], 3000 + i);
        assert!(sg::sdiv(xs[i], sdiv_d[i]) == sdiv_w[i], 4000 + i);
        assert!(sg::sat8(xs[i]) == sat8_w[i], 5000 + i);
        assert!(sg::sat16(xs[i]) == sat16_w[i], 6000 + i);
        i = i + 1;
    }};
}}
"#,
        xs = u64s(&xs.iter().map(|&x| x as u64).collect::<Vec<_>>()),
        ys = u64s(&ys.iter().map(|&y| y as u64).collect::<Vec<_>>()),
        wadd = u64s(&wadd_w),
        wmul = u64s(&wmul_w),
        slt = u8s(&slt_w),
        sars = u8s(&sar_s),
        sarw = u64s(&sar_w),
        sdivd = u64s(&sdiv_d),
        sdivw = u64s(&sdiv_w),
        sat8w = u8s(&sat8_w),
        sat16w = u64s(&sat16_w),
    )
    .unwrap();

    // sext8/16/32 — exhaustive over u8, boundary over the rest.
    let sext8_w: Vec<u64> = (0..=255u8).map(|b| sext8(b) as u64).collect();
    let s16_in: Vec<u64> = vec![0, 1, 0x7FFF, 0x8000, 0x8001, 0xFFFF];
    let s16_w: Vec<u64> = s16_in.iter().map(|&v| sext16(v as u16) as u64).collect();
    let s32_in: Vec<u64> = vec![0, 1, 0x7FFF_FFFF, 0x8000_0000, 0xFFFF_FFFF];
    let s32_w: Vec<u64> = s32_in.iter().map(|&v| sext32(v as u32) as u64).collect();
    write!(
        out,
        r#"
#[test]
fun rust_sext_vectors() {{
    let s8w: vector<u64> = vector[{s8w}];
    let mut b = 0u64;
    while (b < 256) {{
        assert!(sg::sext8((b as u8)) == s8w[b], b);
        b = b + 1;
    }};
    let s16i: vector<u64> = vector[{s16i}];
    let s16w: vector<u64> = vector[{s16w}];
    let s32i: vector<u64> = vector[{s32i}];
    let s32w: vector<u64> = vector[{s32w}];
    let mut i = 0u64;
    while (i < s16i.length()) {{
        assert!(sg::sext16(s16i[i]) == s16w[i], 100 + i);
        i = i + 1;
    }};
    i = 0;
    while (i < s32i.length()) {{
        assert!(sg::sext32(s32i[i]) == s32w[i], 200 + i);
        i = i + 1;
    }};
}}
"#,
        s8w = u64s(&sext8_w),
        s16i = u64s(&s16_in),
        s16w = u64s(&s16_w),
        s32i = u64s(&s32_in),
        s32w = u64s(&s32_w),
    )
    .unwrap();
}

// ---------------------------------------------------------------------------
// Layer 2: verify_step end-to-end vectors
// ---------------------------------------------------------------------------

const D: u8 = 8;
const P: u8 = 2;

/// Deterministic machine: seeded bytes in the low pages, given program.
fn machine(prog: Vec<Instr>, seed: u64) -> Machine {
    let mut m = Machine::new(D, P, prog);
    let mut rng = XorShift64::new(seed);
    for page in 0..4u64 {
        let mut buf = [0u8; PAGE_SIZE];
        rng.fill(&mut buf);
        m.mem.set_page(page, buf);
    }
    m
}

struct VsCase {
    name: &'static str,
    judge: JudgeParams,
    pre_root: Hash,
    claimed: Hash,
    proof: StepProof,
    expect: u8, // 0 resolver, 1 challenger
}

fn build_case(
    name: &'static str,
    instr: Instr,
    seed: u64,
    setup: impl Fn(&mut Machine),
    lie: bool,
) -> VsCase {
    let prog = vec![instr];
    let tree = ProgramTree::new(&prog, P);
    let judge = JudgeParams { d: D, p: P, program_root: tree.root() };
    let mut prover = machine(prog.clone(), seed);
    setup(&mut prover);
    let mut runner = machine(prog, seed);
    setup(&mut runner);
    assert_eq!(prover.state_root(), runner.state_root());
    let pre_root = prover.state_root();
    let proof = build_step_proof(&prover, &tree);
    let claimed = match runner.step() {
        Ok(_) => {
            let mut c = runner.state_root();
            if lie {
                c[3] ^= 0x80; // any wrong post-root
            }
            c
        }
        Err(_) => pre_root, // terminal pre-state: any claim loses (V2)
    };
    VsCase {
        name,
        judge,
        pre_root,
        claimed,
        proof,
        expect: if lie || prover.regs.halted != 0 { 1 } else { 0 },
    }
}

fn emit_opening(out: &mut String, page: &Option<vm::onestep::PageOpening>) {
    match page {
        None => writeln!(out, "        x\"\", vector[],").unwrap(),
        Some(o) => {
            let sibs = o
                .siblings
                .iter()
                .map(|s| format!("x\"{}\"", hex(s)))
                .collect::<Vec<_>>()
                .join(", ");
            writeln!(out, "        x\"{}\", vector[{}],", hex(&o.page), sibs).unwrap();
        }
    }
}

fn emit_case(out: &mut String, c: &VsCase) {
    let instr_sibs = c
        .proof
        .instr_siblings
        .iter()
        .map(|s| format!("x\"{}\"", hex(s)))
        .collect::<Vec<_>>()
        .join(", ");
    write!(
        out,
        r#"
#[test]
fun {}() {{
    let v = interp::verify_step(
        &x"{}",
        &x"{}",
        {}, {},
        &x"{}",
        x"{}",
        x"{}",
        x"{}",
        vector[{}],
"#,
        c.name,
        hex(&c.pre_root),
        hex(&c.claimed),
        c.judge.d,
        c.judge.p,
        hex(&c.judge.program_root),
        hex(&c.proof.regs),
        hex(&c.proof.mem_root),
        hex(&c.proof.instr),
        instr_sibs,
    )
    .unwrap();
    emit_opening(out, &c.proof.open_a);
    emit_opening(out, &c.proof.open_b);
    emit_opening(out, &c.proof.open_w);
    writeln!(out, "    );\n    assert!(v == {}, 0);\n}}", c.expect).unwrap();
}

fn cases() -> Vec<VsCase> {
    let dot8 = Instr {
        imm: 64,
        a: Operand::at(0),
        b: Operand::at(64),
        ..Instr::op(Opcode::Dot8)
    };
    let clamp8 = Instr { w: Operand::at(200), ..Instr::op(Opcode::Clamp8) };
    let div0 = Instr { a: Operand::at(512), ..Instr::op(Opcode::Div32) };
    let shift = Instr { s: 7, ..Instr::op(Opcode::ShiftRndn) };
    let lut = Instr { a: Operand::at(0), ..Instr::op(Opcode::Lut16) };
    let st32 = Instr { k: 3, w: Operand::at(400), ..Instr::op(Opcode::St32) };
    let amx_off = Instr { k: 2, imm: 768, a: Operand::at(300), ..Instr::op(Opcode::ArgmaxOff) };
    let lp = Instr { k: 1, target: 0, imm: 9, ..Instr::op(Opcode::Loop) };
    let halt = Instr::op(Opcode::Halt);
    let ld16 = Instr { a: Operand::at(102), ..Instr::op(Opcode::Ld16) };
    let d816 = Instr {
        imm: 64,
        a: Operand::at(0),
        b: Operand::at(128),
        ..Instr::op(Opcode::Dot8x16)
    };
    let dotbm = Instr { w: Operand::at(260), ..d816 };
    let dotbm = Instr { opcode: Opcode::Dotbm as u8, ..dotbm };
    // B 64- but not 128-aligned ⇒ T7-wide trap (honest claim of the trap).
    let d816_badb = Instr { b: Operand::at(64), ..d816 };
    // FW-6 float ops: FDOT (block dot accumulate) + FOP selectors.
    let fdot = Instr {
        imm: 64,
        a: Operand::at(0),     // 128-aligned bf16 line
        b: Operand::at(256),   // 256-aligned f32 line
        w: Operand::at(512),   // f32 accumulator cell
        ..Instr::op(Opcode::Fdot)
    };
    let ffma_op = Instr {
        k: 2,
        a: Operand::at(16),
        b: Operand::at(20),
        w: Operand::at(24),
        ..Instr::op(Opcode::Fop)
    };
    let fsqrt_op = Instr { k: 4, a: Operand::at(16), w: Operand::at(28), ..Instr::op(Opcode::Fop) };
    let fdiv_op = Instr { k: 3, a: Operand::at(16), b: Operand::at(20), w: Operand::at(32), ..Instr::op(Opcode::Fop) };

    vec![
        build_case("vs_dot8_honest", dot8, 11, |m| m.regs.acc = -12345, false),
        build_case("vs_dot8_false_claim", dot8, 11, |m| m.regs.acc = -12345, true),
        build_case("vs_ld16_honest", ld16, 53, |m| m.regs.acc = 4242, false),
        build_case("vs_dot8x16_honest", d816, 59, |m| m.regs.acc = -987_654, false),
        build_case("vs_dot8x16_false_claim", d816, 59, |m| m.regs.acc = -987_654, true),
        build_case("vs_dotbm_w_read_honest", dotbm, 61, |m| m.regs.acc = 31_337, false),
        build_case("vs_dotbm_false_claim", dotbm, 61, |m| m.regs.acc = 31_337, true),
        build_case("vs_dot8x16_wide_misalign_traps", d816_badb, 67, |_| {}, false),
        build_case("vs_fdot_honest", fdot, 71, |_| {}, false),
        build_case("vs_fdot_false_claim", fdot, 71, |_| {}, true),
        build_case("vs_fop_fma_honest", ffma_op, 73, |_| {}, false),
        build_case("vs_fop_sqrt_honest", fsqrt_op, 79, |_| {}, false),
        build_case("vs_fop_div_false_claim", fdiv_op, 83, |_| {}, true),
        build_case(
            "vs_clamp8_write_honest",
            clamp8,
            13,
            |m| m.regs.acc = 99_999, // saturates to 127
            false,
        ),
        build_case(
            "vs_div_by_zero_traps_honest",
            div0,
            17,
            |m| {
                m.regs.acc = 5000;
                m.mem.write(512, &0i32.to_le_bytes()); // divisor 0 ⇒ T5
            },
            false,
        ),
        build_case("vs_shift_register_only", shift, 19, |m| m.regs.acc = -777_777, false),
        build_case(
            "vs_lut16_saturated_index",
            lut,
            23,
            |m| m.regs.acc = i64::MIN, // sat16 → −32768 → table index 0
            false,
        ),
        build_case(
            "vs_st32_from_idx",
            st32,
            29,
            |m| m.regs.idx[1] = 0xDEAD_BEEF,
            false,
        ),
        build_case("vs_loop_taken", lp, 31, |m| m.regs.idx[1] = 3, false),
        build_case(
            "vs_argmax_off_updates_aux",
            amx_off,
            47,
            |m| {
                m.regs.acc = i64::MIN; // any read value wins
                m.regs.idx[2] = 5; // aux must become 768 + 5
            },
            false,
        ),
        build_case("vs_halt_honest", halt, 37, |m| m.regs.acc = 42, false),
        build_case(
            "vs_halted_pre_state_is_fraud",
            halt,
            41,
            |m| m.regs.halted = 1, // terminal: any successor claim loses
            false,
        ),
        build_case(
            "vs_pc_out_of_tree_traps",
            halt,
            43,
            |m| m.regs.pc = 7, // ≥ 2^p = 4 ⇒ T1 trap, no instr proof needed
            false,
        ),
    ]
}

/// FW-6: softfloat cross-vectors — the Move fp32 ops held to the Rust
/// twin's answers (which are themselves held to HARDWARE bit-for-bit by
/// 6M-case fuzz in vm::softfloat). Includes specials, subnormals, and the
/// committed 64-block dot.
#[allow(clippy::needless_range_loop)] // spec-literal lane indices
fn emit_softfloat_vectors() {
    use vm::softfloat::{fadd, fdiv, ffloor, ffma, fgt, fmul, fsqrt, ftoi, itof};
    let mut s = 0x50F7_F107u64;
    let mut rng = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let mut rand_bits = move || {
        let r = rng();
        let mut b = r as u32;
        match (r >> 32) % 8 {
            0 => b &= 0x807F_FFFF,                    // zero/subnormal
            1 => b = (b & 0x807F_FFFF) | 0x7F80_0000, // inf/nan
            2 => b = (b & 0x807F_FFFF) | 0x7F00_0000, // huge
            3 => b = (b & 0x807F_FFFF) | 0x0080_0000, // tiny normal
            _ => {}
        }
        b
    };
    let mut out = String::new();
    out.push_str(
        "/// GENERATED by `cargo run -p client --bin gen_move_vectors` — DO NOT EDIT.\n\
         /// FW-6 softfloat cross-vectors: expected bits computed by the Rust twin\n\
         /// (vm::softfloat), which is fuzz-proven bit-identical to IEEE hardware.\n\
         #[test_only]\n\
         module dispute::softfloat_vectors;\n\n\
         use dispute::softfloat as sf;\n",
    );
    // Pairwise ops.
    let n = 400usize;
    let mut a_v = Vec::new();
    let mut b_v = Vec::new();
    let mut c_v = Vec::new();
    let mut mul_w = Vec::new();
    let mut add_w = Vec::new();
    let mut fma_w = Vec::new();
    // Edge pins first.
    let edges = [
        0x0000_0000u32,
        0x8000_0000,
        0x7F80_0000,
        0xFF80_0000,
        0x7FC0_0000,
        0x0000_0001,
        0x8000_0001,
        0x007F_FFFF,
        0x0080_0000,
        0x7F7F_FFFF,
        0x3F80_0000,
        0xBF80_0000,
    ];
    for &a in &edges {
        for &b in &edges {
            a_v.push(a);
            b_v.push(b);
            c_v.push(edges[(a ^ b) as usize % edges.len()]);
        }
    }
    for _ in 0..n {
        a_v.push(rand_bits());
        b_v.push(rand_bits());
        c_v.push(rand_bits());
    }
    let mut div_w = Vec::new();
    let mut sqrt_w = Vec::new();
    let mut floor_w = Vec::new();
    let mut ftoi_w = Vec::new();
    let mut fgt_w = Vec::new();
    let mut itof_in = Vec::new();
    let mut itof_w = Vec::new();
    for i in 0..a_v.len() {
        mul_w.push(fmul(a_v[i], b_v[i]));
        add_w.push(fadd(a_v[i], b_v[i]));
        fma_w.push(ffma(a_v[i], b_v[i], c_v[i]));
        div_w.push(fdiv(a_v[i], b_v[i]));
        sqrt_w.push(fsqrt(a_v[i]));
        floor_w.push(ffloor(a_v[i]));
        ftoi_w.push(ftoi(a_v[i]) as u32);
        fgt_w.push(fgt(a_v[i], b_v[i]));
        let v = (a_v[i] ^ b_v[i]) as i32;
        itof_in.push(v as u32);
        itof_w.push(itof(v));
    }
    let u32s = |v: &[u32]| v.iter().map(|x| format!("{x}")).collect::<Vec<_>>().join(", ");
    write!(
        out,
        r#"
#[test]
fun rust_softfloat_vectors() {{
    let a: vector<u32> = vector[{}];
    let b: vector<u32> = vector[{}];
    let c: vector<u32> = vector[{}];
    let mulw: vector<u32> = vector[{}];
    let addw: vector<u32> = vector[{}];
    let fmaw: vector<u32> = vector[{}];
    let mut i = 0u64;
    while (i < a.length()) {{
        assert!(sf::fmul(a[i], b[i]) == mulw[i], i);
        assert!(sf::fadd(a[i], b[i]) == addw[i], 10000 + i);
        assert!(sf::ffma(a[i], b[i], c[i]) == fmaw[i], 20000 + i);
        i = i + 1;
    }};
}}

#[test]
fun rust_softfloat_unary_div_cvt_vectors() {{
    let a: vector<u32> = vector[{a2}];
    let b: vector<u32> = vector[{b2}];
    let divw: vector<u32> = vector[{divw}];
    let sqrtw: vector<u32> = vector[{sqrtw}];
    let floorw: vector<u32> = vector[{floorw}];
    let ftoiw: vector<u32> = vector[{ftoiw}];
    let itofi: vector<u32> = vector[{itofi}];
    let itofw: vector<u32> = vector[{itofw}];
    let fgtw: vector<u32> = vector[{fgtw}];
    let mut i = 0u64;
    while (i < a.length()) {{
        assert!(sf::fdiv(a[i], b[i]) == divw[i], i);
        assert!(sf::fsqrt(a[i]) == sqrtw[i], 10000 + i);
        assert!(sf::ffloor(a[i]) == floorw[i], 20000 + i);
        assert!(sf::ftoi(a[i]) == ftoiw[i], 30000 + i);
        assert!(sf::itof(itofi[i]) == itofw[i], 40000 + i);
        assert!(sf::fgt(a[i], b[i]) == fgtw[i], 50000 + i);
        i = i + 1;
    }};
}}
"#,
        u32s(&a_v),
        u32s(&b_v),
        u32s(&c_v),
        u32s(&mul_w),
        u32s(&add_w),
        u32s(&fma_w),
        a2 = u32s(&a_v),
        b2 = u32s(&b_v),
        divw = u32s(&div_w),
        sqrtw = u32s(&sqrt_w),
        floorw = u32s(&floor_w),
        ftoiw = u32s(&ftoi_w),
        itofi = u32s(&itof_in),
        itofw = u32s(&itof_w),
        fgtw = u32s(&fgt_w),
    )
    .unwrap();
    // Committed block dots (finite inputs — honest-trace domain).
    for case in 0..3 {
        let mut w = Vec::new();
        let mut x = Vec::new();
        while w.len() < 64 {
            let wb = rand_bits() & 0xFFFF_0000; // bf16-pattern weight
            let xb = rand_bits() & 0x7FFF_FFFF;
            if (wb >> 23) & 0xFF == 0xFF || (xb >> 23) & 0xFF == 0xFF {
                continue; // skip inf/nan draws
            }
            w.push(wb);
            x.push(xb);
        }
        let mut lanes = [0u32; 16];
        for i in 0..4 {
            for kk in 0..16 {
                let j = 16 * i + kk;
                lanes[kk] = ffma(w[j], x[j], lanes[kk]);
            }
        }
        let mut sline = [0u32; 4];
        for l in 0..4 {
            sline[l] = fadd(fadd(lanes[l], lanes[4 + l]), fadd(lanes[8 + l], lanes[12 + l]));
        }
        let want = fadd(fadd(sline[0], sline[1]), fadd(sline[2], sline[3]));
        write!(
            out,
            r#"
#[test]
fun committed_block_dot_{case}() {{
    let w: vector<u32> = vector[{}];
    let x: vector<u32> = vector[{}];
    assert!(sf::block_dot(&w, &x) == {want}, 0);
}}
"#,
            u32s(&w),
            u32s(&x),
        )
        .unwrap();
    }
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../dispute/tests/softfloat_vectors.move");
    std::fs::write(path, &out).expect("write softfloat vectors");
    println!("wrote {path} ({} KiB)", out.len() / 1024);
}

fn main() {
    emit_softfloat_vectors();
    let mut out = String::new();
    out.push_str(
        "/// GENERATED by `cargo run -p client --bin gen_move_vectors` — DO NOT EDIT.\n\
         /// Rust↔Move equivalence suite: expected values computed by vm::exec /\n\
         /// vm::onestep (the Rust twins). A failure here is a semantic gap between\n\
         /// the off-chain and on-chain interpreters — a critical soundness bug.\n\
         #[test_only]\n\
         module dispute::generated_vectors;\n\n\
         use dispute::interp;\n\
         use dispute::signed as sg;\n",
    );
    let mut rng = XorShift64::new(0x6E2A_70E5);
    emit_signed_vectors(&mut out, &mut rng);
    for c in cases() {
        emit_case(&mut out, &c);
    }
    // One tampered-opening case: must ABORT (E_BAD_OPENING = 4), not decide.
    let mut t = cases().remove(0);
    if let Some(o) = &mut t.proof.open_a {
        t.proof.open_a = Some(vm::onestep::PageOpening {
            page: {
                let mut p = o.page.clone();
                p[100] ^= 1;
                p
            },
            siblings: o.siblings.clone(),
        });
    }
    t.name = "vs_tampered_opening_aborts";
    out.push_str("\n#[expected_failure(abort_code = 4, location = dispute::interp)]");
    emit_case(&mut out, &t);

    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../dispute/tests/generated_vectors.move");
    std::fs::write(path, &out).expect("write generated vectors");
    println!("wrote {path} ({} KiB)", out.len() / 1024);
}
