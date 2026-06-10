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

    vec![
        build_case("vs_dot8_honest", dot8, 11, |m| m.regs.acc = -12345, false),
        build_case("vs_dot8_false_claim", dot8, 11, |m| m.regs.acc = -12345, true),
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

fn main() {
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
