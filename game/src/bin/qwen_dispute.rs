//! A REAL fraud game at Qwen scale (brief Phase 3.6, local chain): a faulty
//! resolver corrupts one weight byte mid-execution of the 29.5M-step Qwen
//! judgment; the bisection isolates exactly that step; the one-step verifier
//! convicts. The full-trace machinery of the toy game is impossible here
//! (29.5M roots), so parties use the two-level scheme (SPEC §7.4): a cursor
//! machine pinned at the agreed `lo` (which never decreases) plus
//! clone-and-advance for midpoint queries — O(N) total stepping per party.
//!
//!   cargo run -p game --release --bin qwen_dispute -- [n_prompt] [n_gen]

use compiler::qwen::compile_qwen;
use game::actors::Claim;
use game::chain::{MockChain, Params, Role, Status};
use qwen::config::QwenConfig;
use qwen::image::{genesis_image, Tables};
use qwen::layout::{QwenLayout, MEM_DEPTH};
use qwen::quant::{quantize, Calib, FloatModel, FloatState};
use qwen::tensors::SafeTensors;
use std::time::Instant;
use vm::exec::Machine;
use vm::hash::Hash;
use vm::isa::Instr;
use vm::onestep::{build_step_proof, JudgeParams, ProgramTree, StepProof};

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/qwen/artifacts");

/// One corrupted byte in a never-rewritten (weight) page at a chosen step —
/// the minimal "I computed the model wrong" fraud.
#[derive(Clone, Copy)]
struct Fault {
    step: u64,
    addr: u64,
    bit: u8,
}

struct QwenParty {
    #[allow(dead_code)] // role label for readability/debug
    name: &'static str,
    fault: Option<Fault>,
    /// Pinned at the dispute's agreed `lo` — advance-only.
    cursor: Machine,
    /// Steps executed across the whole game (the honest-work measure).
    stepped: u64,
    cloned: u32,
}

impl QwenParty {
    fn new(name: &'static str, d: u8, p: u8, prog: Vec<Instr>, image: &[u8], fault: Option<Fault>) -> Self {
        let mut m = Machine::new(d, p, prog);
        for (i, page) in image.chunks_exact(vm::PAGE_SIZE).enumerate() {
            if page.iter().any(|&b| b != 0) {
                m.mem.set_page(i as u64, page.try_into().unwrap());
            }
        }
        Self { name, fault, cursor: m, stepped: 0, cloned: 0 }
    }

    fn advance(m: &mut Machine, fault: Option<Fault>, target: u64, stepped: &mut u64) {
        while m.regs.step < target && m.regs.halted == 0 {
            let s = m.regs.step;
            m.step().expect("not terminal");
            *stepped += 1;
            if let Some(f) = fault {
                if f.step == s {
                    let v = m.mem.read_u8(f.addr) ^ (1 << (f.bit % 8));
                    m.mem.write(f.addr, &[v]);
                }
            }
        }
    }

    /// Root at `step ≥ lo`: clone the cursor, advance the clone.
    fn root_at(&mut self, step: u64) -> Hash {
        assert!(step >= self.cursor.regs.step, "bisection lo regressed");
        self.cloned += 1;
        let mut c = self.cursor.clone();
        Self::advance(&mut c, self.fault, step, &mut self.stepped);
        c.state_root()
    }

    /// The chain agreed up to `lo`: move the pinned cursor forward.
    fn sync_lo(&mut self, lo: u64) {
        let f = self.fault;
        let stepped = &mut self.stepped;
        Self::advance(&mut self.cursor, f, lo, stepped);
    }

    /// Run a clone to terminal: (terminal step, terminal root, output bytes).
    fn terminal(&mut self, out_base: u64) -> (u64, Hash, Vec<u8>) {
        self.cloned += 1;
        let mut c = self.cursor.clone();
        Self::advance(&mut c, self.fault, u64::MAX, &mut self.stepped);
        (c.regs.step, c.state_root(), c.mem.read(out_base, 4).to_vec())
    }

    fn proof_at(&mut self, lo: u64, tree: &ProgramTree) -> StepProof {
        self.cloned += 1;
        let mut c = self.cursor.clone();
        Self::advance(&mut c, self.fault, lo, &mut self.stepped);
        build_step_proof(&c, tree)
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let n_prompt: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(2);
    let n_gen: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(1);

    println!("· loading + quantizing Qwen3-0.6B…");
    let cfg = QwenConfig::load(&format!("{DIR}/config.json"));
    let tables = Tables::generate(cfg.rope_theta, cfg.head_dim);
    let prompt: Vec<u32> = (0..n_prompt as u32).map(|i| 9707 + 13 * i).collect();
    let st = SafeTensors::load(&format!("{DIR}/model.safetensors"));
    let fm = FloatModel::load(&cfg, &st);
    drop(st);
    let mut calib = Calib::new(cfg.num_hidden_layers);
    let mut fs = FloatState::new(&cfg, qwen::layout::MAX_SEQ);
    let mut tok = prompt[0];
    for pos in 0..prompt.len() + 4 {
        let next = qwen::quant::float_forward(
            &fm, &mut fs, qwen::layout::MAX_SEQ, &tables.cos, &tables.sin, tok, &mut calib,
        );
        tok = if pos >= prompt.len() - 1 { next } else { prompt[pos + 1] };
    }
    let im = quantize(&fm, &calib);
    drop(fm);

    let lay = QwenLayout::new(&cfg);
    let image = genesis_image(&lay, &im, &tables, &prompt);
    let c = compile_qwen(&lay, &im, n_prompt, n_gen);
    let tree = ProgramTree::new(&c.program, c.p);
    let judge = JudgeParams { d: MEM_DEPTH, p: c.p, program_root: tree.root() };
    println!(
        "  program: {} instrs (p = {}), judgment = {} steps",
        c.program.len(),
        c.p,
        c.total_steps + 1
    );

    // The fraud: one bit of one weight byte (layer 14 gate matrix), flipped
    // mid-execution of position 1 — a never-rewritten page, so the
    // corruption provably reaches the final root.
    let fault = Fault {
        step: c.token_boundaries[0] + 1_234_567,
        addr: lay.layers[14].w_gate + 99_999,
        bit: 3,
    };
    println!(
        "· fault: flip bit {} of weight byte @{} at step {}",
        fault.bit, fault.addr, fault.step
    );

    let t0 = Instant::now();
    let mut resolver = QwenParty::new("resolver", MEM_DEPTH, c.p, c.program.clone(), &image, Some(fault));
    let mut challenger = QwenParty::new("challenger", MEM_DEPTH, c.p, c.program.clone(), &image, None);

    // Resolver asserts its (faulted) judgment on-chain.
    println!("· resolver computes + asserts its judgment…");
    let (n, root_n, output) = resolver.terminal(lay.tok);
    let genesis_root = challenger.cursor.state_root();
    let chain = MockChain::new(Params::default());
    let mut fact = chain.assert_fact(judge, genesis_root, lay.tok, Claim { n, root_n, output });

    // Honest challenger recomputes, sees a different root, challenges.
    println!("· challenger recomputes…");
    let (cn, c_root, _) = challenger.terminal(lay.tok);
    assert_eq!(cn, n, "static schedule ⇒ same step count");
    assert_ne!(c_root, root_n, "fault must reach the final root");
    chain.challenge(&mut fact).expect("challenge accepted");
    println!("· bisecting over {} steps…", n);

    let mut rounds = 0;
    let winner = loop {
        let d = fact.dispute.as_ref().unwrap().clone();
        if d.hi - d.lo == 1 {
            println!("  ⚖ atomic at step {} → {} (fault was at {})", d.lo, d.hi, fault.step);
            // Fault mutates state AFTER executing step f.step: roots agree
            // through index f.step and diverge from f.step+1.
            assert_eq!(d.lo, fault.step, "bisection isolated the corrupted transition");
            let proof = challenger.proof_at(d.lo, &tree);
            break chain.submit_proof(&mut fact, &proof).expect("valid proof");
        }
        let mid = d.lo + (d.hi - d.lo) / 2;
        chain.post_mid(&mut fact, resolver.root_at(mid)).expect("post_mid");
        let agree = challenger.root_at(mid) == fact.dispute.as_ref().unwrap().pending_mid.unwrap();
        chain.respond(&mut fact, agree).expect("respond");
        let d2 = fact.dispute.as_ref().unwrap();
        rounds += 1;
        println!(
            "  round {:>2}: mid {:>9} — challenger {} — interval [{}, {}] ({} steps)",
            rounds,
            mid,
            if agree { "AGREES   " } else { "DISAGREES" },
            d2.lo,
            d2.hi,
            d2.hi - d2.lo
        );
        // Both cursors may move up to the agreed lo.
        let lo = d2.lo;
        resolver.sync_lo(lo);
        challenger.sync_lo(lo);
    };

    assert_eq!(winner, Role::Challenger, "fraud must lose");
    assert_eq!(fact.status, Status::Rejected);
    println!("== verdict ==");
    println!("winner: {winner:?} (fact rejected, resolver slashed {})", fact.resolver_delta);
    println!(
        "rounds: {rounds}; resolver stepped {}M / cloned {}; challenger stepped {}M / cloned {}",
        resolver.stepped / 1_000_000,
        resolver.cloned,
        challenger.stepped / 1_000_000,
        challenger.cloned
    );
    println!("wall: {:.1?} (model load excluded)", t0.elapsed());
    println!("✓ Qwen-scale fraud game: one flipped weight bit at step {} isolated and convicted", fault.step);
}
