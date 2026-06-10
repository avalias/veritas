//! The soundness test (brief Phase 2.3, run locally against the Rust twin):
//! for randomized machine states — valid, trapping, halted, adversarial —
//! the one-step verifier's verdict must agree with `Machine::step`, honest
//! proofs of honest claims must win, and tampered anything must lose or
//! abort. When the Move contract lands, it cross-tests against the same
//! vectors.

#![allow(clippy::manual_is_multiple_of)] // `% n == 0` reads fine in test RNG plumbing

use vm::exec::{Machine, StepOutcome};
use vm::fixtures::XorShift64;
use vm::isa::{Instr, Operand};
use vm::onestep::{build_step_proof, verify_step, JudgeParams, ProgramTree, StepProof, Verdict};
use vm::state::Registers;
use vm::PAGE_SIZE;

const D: u8 = 8; // 256 KiB — covers full LUT16 index range
const P: u8 = 2; // 4 program slots

/// Random-but-plausible instruction: real opcodes with fields that land in
/// valid ranges often enough to exercise every op, and invalid ranges often
/// enough to exercise every trap.
fn rand_instr(rng: &mut XorShift64) -> Instr {
    let mem_bytes = (1u64 << D) * PAGE_SIZE as u64;
    let mut i = Instr {
        // 1..=0x15 mostly (valid), occasionally junk (T2).
        opcode: if rng.next_u64() % 8 == 0 {
            rng.next_u64() as u8
        } else {
            (1 + rng.next_u64() % 0x15) as u8
        },
        k: (rng.next_u64() % 8) as u8,
        s: (rng.next_u64() % 80) as u8,
        imm: match rng.next_u64() % 4 {
            0 => rng.next_u64() as u32,         // junk (DOT trap, JEQ miss…)
            1 => 1 + (rng.next_u64() % 64) as u32, // valid DOT lanes
            _ => (rng.next_u64() % 8) as u32,   // small (LOOP trips, LDC)
        },
        target: (rng.next_u64() % 6) as u32,
        a: rand_operand(rng, mem_bytes),
        b: rand_operand(rng, mem_bytes),
        w: rand_operand(rng, mem_bytes),
    };
    // Keep LUT16 bases mostly in range so it doesn't always trap.
    if i.opcode == 0x0C {
        i.a.base = (rng.next_u64() % 2) * 65536;
        i.a.stride = [0; 4];
    }
    i
}

fn rand_operand(rng: &mut XorShift64, mem_bytes: u64) -> Operand {
    let aligned = rng.next_u64() % 8 != 0; // mostly aligned/in-bounds
    let base = if aligned {
        (rng.next_u64() % (mem_bytes / 64)) * 64
    } else {
        rng.next_u64() % (2 * mem_bytes) // sometimes OOB/misaligned
    };
    let mut stride = [0u32; 4];
    if rng.next_u64() % 4 == 0 {
        stride[(rng.next_u64() % 4) as usize] = [1u32, 2, 4, 64][(rng.next_u64() % 4) as usize];
    }
    Operand { base, stride }
}

/// Build a machine in a randomized state (memory + registers).
fn rand_machine(rng: &mut XorShift64, program: Vec<Instr>) -> Machine {
    let mut m = Machine::new(D, P, program);
    // Random bytes in the low pages (operands mostly land here) and around
    // the LUT16 window at 65536.
    for page in [0u64, 1, 2, 3, 64, 65, 191] {
        let mut buf = [0u8; PAGE_SIZE];
        rng.fill(&mut buf);
        m.mem.set_page(page, buf);
    }
    m.regs = Registers {
        pc: match rng.next_u64() % 8 {
            0 => 1,                          // padding slot (T2 via zero instr)
            1 => 4 + (rng.next_u64() % 4) as u32, // ≥ 2^p (T1/V3)
            _ => 0,                          // the real instruction
        },
        halted: if rng.next_u64() % 16 == 0 { 1 } else { 0 },
        step: rng.next_u64() % 1_000_000,
        acc: match rng.next_u64() % 3 {
            0 => rng.next_u64() as i64,           // wild (LUT16 saturation…)
            1 => (rng.next_u64() % 65536) as i64 - 32768, // LUT16 domain
            _ => (rng.next_u64() % 4096) as i64 - 2048,
        },
        aux: rng.next_u64() as i64,
        idx: [
            (rng.next_u64() % 8) as u32,
            (rng.next_u64() % 8) as u32,
            (rng.next_u64() % 8) as u32,
            (rng.next_u64() % 8) as u32,
        ],
    };
    m
}

#[test]
fn verifier_matches_machine_on_random_states() {
    let mut rng = XorShift64::new(0xA11CE);
    let mut counts = [0u64; 3]; // ran, trapped, terminal-pre
    for trial in 0..4000 {
        let program = vec![rand_instr(&mut rng)];
        let tree = ProgramTree::new(&program, P);
        let judge = JudgeParams { d: D, p: P, program_root: tree.root() };

        // Two identical machines: one to prove from, one to step.
        let seed_state = rng.next_u64() | 1;
        let mut rng_a = XorShift64::new(seed_state);
        let mut rng_b = XorShift64::new(seed_state);
        let prover = rand_machine(&mut rng_a, program.clone());
        let mut runner = rand_machine(&mut rng_b, program.clone());
        assert_eq!(prover.state_root(), runner.state_root());

        let pre_root = prover.state_root();
        let proof = build_step_proof(&prover, &tree);

        match runner.step() {
            Err(_) => {
                // Terminal pre-state: ANY claimed successor is fraud (V2).
                counts[2] += 1;
                let v = verify_step(&pre_root, &pre_root, &judge, &proof).unwrap();
                assert_eq!(v, Verdict::ChallengerWins, "trial {trial}: halted pre");
            }
            Ok(outcome) => {
                counts[if outcome == StepOutcome::Trapped { 1 } else { 0 }] += 1;
                let true_post = runner.state_root();
                // Honest claim with honest proof ⇒ resolver wins.
                let v = verify_step(&pre_root, &true_post, &judge, &proof)
                    .unwrap_or_else(|e| panic!("trial {trial}: honest proof aborted: {e:?}"));
                assert_eq!(v, Verdict::ResolverWins, "trial {trial}: {outcome:?}");
                // Any other claimed post-root ⇒ challenger wins.
                let mut bad = true_post;
                bad[7] ^= 0x40;
                let v = verify_step(&pre_root, &bad, &judge, &proof).unwrap();
                assert_eq!(v, Verdict::ChallengerWins, "trial {trial}");
            }
        }
    }
    // The generator must actually exercise all three regimes.
    assert!(counts.iter().all(|&c| c > 100), "regime coverage {counts:?}");
}

#[test]
fn tampered_proofs_abort_not_win() {
    let mut rng = XorShift64::new(0xBEEF);
    let mut tamper_hits = 0;
    for _ in 0..400 {
        let program = vec![rand_instr(&mut rng)];
        let tree = ProgramTree::new(&program, P);
        let judge = JudgeParams { d: D, p: P, program_root: tree.root() };
        let seed_state = rng.next_u64() | 1;
        let mut rng_a = XorShift64::new(seed_state);
        let mut rng_b = XorShift64::new(seed_state);
        let prover = rand_machine(&mut rng_a, program.clone());
        let mut runner = rand_machine(&mut rng_b, program);
        let pre_root = prover.state_root();
        let proof = build_step_proof(&prover, &tree);
        if runner.step().is_err() {
            continue;
        }
        let true_post = runner.state_root();

        // Tamper 1: lie about the registers (breaks V1).
        let mut p1 = proof.clone();
        p1.regs[13] ^= 1; // acc low byte
        assert!(verify_step(&pre_root, &true_post, &judge, &p1).is_err());

        // Tamper 2: substitute a different instruction (breaks V4) —
        // only meaningful when the instruction was actually fetched.
        if !proof.instr_siblings.is_empty() {
            let mut p2 = proof.clone();
            p2.instr[4] ^= 0xFF;
            assert!(verify_step(&pre_root, &true_post, &judge, &p2).is_err());
            tamper_hits += 1;
        }

        // Tamper 3: corrupt an opened page (breaks the opening fold).
        if let Some(o) = proof.open_a.clone() {
            let mut p3 = proof.clone();
            let mut bad = o;
            bad.page[17] ^= 1;
            p3.open_a = Some(bad);
            assert!(verify_step(&pre_root, &true_post, &judge, &p3).is_err());
            tamper_hits += 1;
        }
    }
    assert!(tamper_hits > 100, "tamper paths under-exercised: {tamper_hits}");
}

/// A dishonest resolver cannot construct ANY proof that validates a false
/// post-root: the verifier recomputes the post-state from openings bound to
/// the pre-root, so the only free variable is the claim itself.
#[test]
fn false_claims_lose_regardless_of_proof() {
    let mut rng = XorShift64::new(0xF4A5D);
    for _ in 0..500 {
        let program = vec![rand_instr(&mut rng)];
        let tree = ProgramTree::new(&program, P);
        let judge = JudgeParams { d: D, p: P, program_root: tree.root() };
        let seed_state = rng.next_u64() | 1;
        let mut rng_a = XorShift64::new(seed_state);
        let mut rng_b = XorShift64::new(seed_state);
        let prover = rand_machine(&mut rng_a, program.clone());
        let mut runner = rand_machine(&mut rng_b, program);
        if runner.regs.halted != 0 {
            continue;
        }
        let pre_root = prover.state_root();
        let proof = build_step_proof(&prover, &tree);
        runner.step().unwrap();
        // Claim a root that is NOT the true successor.
        let mut lie = runner.state_root();
        lie[0] ^= 0x01;
        let v = verify_step(&pre_root, &lie, &judge, &proof).unwrap();
        assert_eq!(v, Verdict::ChallengerWins);
    }
}

#[test]
fn step_proof_size_is_bounded() {
    // SPEC §8.4 calldata estimate: ≤ 2 page openings + instruction proof.
    let mut rng = XorShift64::new(0x517E);
    for _ in 0..200 {
        let program = vec![rand_instr(&mut rng)];
        let tree = ProgramTree::new(&program, P);
        let mut rng_a = XorShift64::new(rng.next_u64() | 1);
        let prover = rand_machine(&mut rng_a, program);
        let proof: StepProof = build_step_proof(&prover, &tree);
        let openings = [&proof.open_a, &proof.open_b, &proof.open_w];
        let bytes: usize = openings
            .iter()
            .filter_map(|o| o.as_ref())
            .map(|o| o.page.len() + 32 * o.siblings.len())
            .sum::<usize>()
            + 96
            + 32 * proof.instr_siblings.len()
            + 45
            + 32;
        assert!(bytes <= 2 * (1024 + 32 * D as usize) + 96 + 32 * P as usize + 77 + 64);
        assert!(proof.open_b.is_none() || proof.open_w.is_none(), "never B-read AND write");
    }
}
