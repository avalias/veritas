//! Stage a REAL fraud-proof dispute on a live network (devnet), bisected
//! down to the atomic interval, and emit demos/prediction-market/web/dispute.json so the dApp
//! can let a user deliver the final `verify_step` kill-shot from their
//! wallet — convicting the liar and slashing the bond on-chain.
//!
//!   sui client switch --env devnet
//!   cargo run -p client --bin devnet_stage_dispute -- <package> <resolver> <challenger>
//!
//! A resolver asserts a fraudulent toy-judge trace (one flipped weight
//! bit) with a bond; the challenger bonds and the two bisect the disputed
//! interval to one micro-op. We stop there (status CHALLENGED, atomic) and
//! write out the Fact id + the one-step proof. Anyone can then call
//! verify_step with that proof to settle it.

use client::localnet::{created_of_type, hex_arg, hex_vec_arg, Cli};
use game::actors::Party;
use game::setup::ToySetup;
use game::trace::{Fault, FaultKind};
use serde_json::{json, Value};

const BOND: u64 = 1_000_000; // 0.001 SUI
const CLOCK: &str = "0x6";
const LONG: u64 = 86_400_000; // 24h windows so the user has time to convict

fn main() {
    let mut a = std::env::args().skip(1);
    let (pkg, resolver, challenger) = (
        a.next().expect("usage: devnet_stage_dispute <package> <resolver> <challenger>"),
        a.next().expect("resolver address"),
        a.next().expect("challenger address"),
    );
    let cli = Cli::new(pkg.clone());

    let s = ToySetup::new("Rain?", 4);
    let honest = Party::new(&s, None);
    let n = honest.trace.n;

    // resolver asserts a fraudulent trace (one flipped weight bit)
    let fault = Fault { step: n / 3, kind: FaultKind::FlipMemBit { addr: s.lay.wq[1] + 777, bit: 2 } };
    let dishonest = Party::new(&s, Some(fault));
    assert_ne!(dishonest.claim().root_n, honest.claim().root_n);

    eprintln!("⛓ resolver asserts fraud at step {} (n={n})…", fault.step);
    cli.switch(&resolver);
    let bond_r = cli.split_bond(BOND);
    let c = dishonest.claim();
    let assert_args = vec![
        toy_model::layout::MEM_DEPTH.to_string(),
        s.compiled.p.to_string(),
        hex_arg(&s.judge.program_root),
        hex_arg(&s.genesis_root),
        s.lay.output.to_string(),
        c.n.to_string(),
        hex_arg(&c.root_n),
        hex_arg(&c.output),
        bond_r,
        LONG.to_string(),
        LONG.to_string(),
        CLOCK.into(),
    ];
    let tx = cli.call(&resolver, "assert_fact", &assert_args).expect("assert_fact");
    let fact = created_of_type(&tx, "::dispute::Fact").expect("Fact id");
    eprintln!("   fact {fact}");

    cli.switch(&challenger);
    let bond_c = cli.split_bond(BOND);
    cli.call(&challenger, "challenge", &[fact.clone(), bond_c, CLOCK.into()]).expect("challenge");

    // bisect to the atomic interval, recording each round for the UI
    let mut rounds: Vec<Value> = vec![];
    let (lo, hi) = loop {
        let f = cli.object_fields(&fact);
        let lo: u64 = f["lo"].as_str().unwrap().parse().unwrap();
        let hi: u64 = f["hi"].as_str().unwrap().parse().unwrap();
        if hi - lo == 1 {
            break (lo, hi);
        }
        let mid = lo + (hi - lo) / 2;
        cli.call(&resolver, "post_mid", &[fact.clone(), hex_arg(&dishonest.root_at(mid)), CLOCK.into()])
            .expect("post_mid");
        let agree = honest.root_at(mid) == dishonest.root_at(mid);
        cli.call(&challenger, "respond", &[fact.clone(), agree.to_string(), CLOCK.into()]).expect("respond");
        rounds.push(json!({"mid": mid, "agree": agree, "interval": hi - lo}));
        eprintln!("   round {:>2}: mid {mid} — challenger {}", rounds.len(), if agree { "agrees" } else { "disagrees" });
    };
    eprintln!("   atomic interval [{lo}, {hi}] — staged. fault step = {}", fault.step);

    // the one-step proof the dApp will submit via verify_step
    let proof = honest.build_proof(&s, lo);
    let open = |o: &Option<vm::onestep::PageOpening>| -> (Value, Value) {
        match o {
            None => (json!("0x"), json!([])),
            Some(o) => (json!(hex_arg(&o.page)), json!(o.siblings.iter().map(|s| hex_arg(s)).collect::<Vec<_>>())),
        }
    };
    let (pa, sa) = open(&proof.open_a);
    let (pb, sb) = open(&proof.open_b);
    let (pw, sw) = open(&proof.open_w);
    let out = json!({
        "package": pkg,
        "fact": fact,
        "resolver": resolver,
        "challenger": challenger,
        "n": n,
        "fault_step": fault.step,
        "atomic": [lo, hi],
        "rounds": rounds,
        "bond_mist": BOND,
        "verify": {
            "regs": hex_arg(&proof.regs),
            "mem_root": hex_arg(&proof.mem_root),
            "instr": hex_arg(&proof.instr),
            "instr_sibs": proof.instr_siblings.iter().map(|s| hex_arg(s)).collect::<Vec<_>>(),
            "page_a": pa, "sibs_a": sa,
            "page_b": pb, "sibs_b": sb,
            "page_w": pw, "sibs_w": sw,
        }
    });
    let root = std::process::Command::new("git").args(["rev-parse", "--show-toplevel"]).output().unwrap();
    let root = String::from_utf8(root.stdout).unwrap().trim().to_string();
    let path = format!("{root}/demos/prediction-market/web/dispute.json");
    std::fs::write(&path, serde_json::to_string_pretty(&out).unwrap()).unwrap();
    println!("wrote {path}");
    println!("fact {fact} staged at atomic interval [{lo},{hi}] — deliver verify_step to convict");
    // hex_vec_arg kept for symmetry with the CLI driver
    let _ = hex_vec_arg(&proof.instr_siblings);
}
