//! Phase 2 finale: the fraud game on a REAL Sui localnet.
//!
//!   sui start --with-faucet --force-regenesis   (running, funded addrs)
//!   sui client publish dispute/ …               (package id)
//!   cargo run -p client --bin localnet_demo -- <package_id> <resolver> <challenger>
//!
//! Scenario A: honest assertion finalizes after the challenge window.
//! Scenario B: a resolver asserts a fraudulent trace (one flipped weight
//! bit); the challenger opens a dispute; ~17 bisection rounds of real
//! transactions; one verify_step transaction executes the disputed
//! micro-op inside the Move VM; the resolver's bond is slashed.

use client::localnet::{created_of_type, gas_used, hex_arg, hex_vec_arg, Cli};
use game::actors::Party;
use game::setup::ToySetup;
use game::trace::{Fault, FaultKind};
use vm::onestep::StepProof;

const BOND: u64 = 1_000_000; // 0.001 SUI in MIST — demo economics
const CLOCK: &str = "0x6";

fn main() {
    let mut a = std::env::args().skip(1);
    let (pkg, resolver, challenger) = (
        a.next().expect("usage: localnet_demo <package> <resolver> <challenger>"),
        a.next().expect("resolver address"),
        a.next().expect("challenger address"),
    );
    let cli = Cli::new(pkg);

    println!("⛓ building toy judge…");
    let s = ToySetup::new("Rain?", 4);
    let honest = Party::new(&s, None);
    let n = honest.trace.n;

    // ---- Scenario A: honest assertion, unchallenged ----------------------
    println!("⛓ scenario A: honest assert → finalize after window");
    cli.switch(&resolver);
    let bond_a = cli.split_bond(BOND);
    let tx = cli
        .call(&resolver, "assert_fact", &assert_args(&s, &honest, &bond_a, 3_000, 600_000))
        .expect("assert_fact A");
    let fact_a = created_of_type(&tx, "::dispute::Fact").expect("Fact id");
    println!("   fact {fact_a} asserted; sleeping past the 3s window…");
    std::thread::sleep(std::time::Duration::from_millis(4_500));
    cli.call(&resolver, "finalize", &[fact_a.clone(), CLOCK.into()]).expect("finalize");
    let st = cli.object_fields(&fact_a)["status"].as_u64().unwrap();
    assert_eq!(st, 3, "FINALIZED");
    println!("   ✓ finalized on-chain (status 3), bond returned");

    // ---- Scenario B: fraud, dispute, slash --------------------------------
    let fault = Fault { step: n / 3, kind: FaultKind::FlipMemBit { addr: s.lay.wq[1] + 777, bit: 2 } };
    println!("⛓ scenario B: resolver flips a weight bit at step {} and asserts", fault.step);
    let dishonest = Party::new(&s, Some(fault));
    assert_ne!(dishonest.claim().root_n, honest.claim().root_n);

    cli.switch(&resolver);
    let bond_r = cli.split_bond(BOND);
    let tx = cli
        .call(&resolver, "assert_fact", &assert_args(&s, &dishonest, &bond_r, 600_000, 600_000))
        .expect("assert_fact B");
    let fact = created_of_type(&tx, "::dispute::Fact").expect("Fact id");

    cli.switch(&challenger);
    let bond_c = cli.split_bond(BOND);
    cli.call(&challenger, "challenge", &[fact.clone(), bond_c, CLOCK.into()])
        .expect("challenge");
    println!("   challenger bonded; bisection over [0, {n}]:");

    let mut rounds = 0u32;
    let (lo, hi) = loop {
        let f = cli.object_fields(&fact);
        let lo: u64 = f["lo"].as_str().unwrap().parse().unwrap();
        let hi: u64 = f["hi"].as_str().unwrap().parse().unwrap();
        if hi - lo == 1 {
            break (lo, hi);
        }
        let mid = lo + (hi - lo) / 2;
        cli.call(&resolver, "post_mid", &[
            fact.clone(),
            hex_arg(&dishonest.root_at(mid)),
            CLOCK.into(),
        ])
        .expect("post_mid");
        let agree = honest.root_at(mid) == dishonest.root_at(mid);
        cli.call(&challenger, "respond", &[fact.clone(), agree.to_string(), CLOCK.into()])
            .expect("respond");
        rounds += 1;
        println!(
            "   round {rounds:>2}: mid {mid:>6} — challenger {} → interval {} steps",
            if agree { "agrees   " } else { "disagrees" },
            (hi - lo) / 2 + (hi - lo) % 2
        );
    };
    println!("   atomic interval [{lo}, {hi}] — submitting one-step proof…");

    let proof = honest.build_proof(&s, lo);
    let (args, calldata) = verify_args(&fact, &proof);
    let tx = cli.call(&challenger, "verify_step", &args).expect("verify_step");
    let g = gas_used(&tx);
    let f = cli.object_fields(&fact);
    let status = f["status"].as_u64().unwrap();
    assert_eq!(status, 4, "REJECTED — resolver slashed");
    println!("   ✓ on-chain verdict: fraud at step {lo} (= injected {})", fault.step);
    assert_eq!(lo, fault.step);

    println!("\n== gas & size report (localnet) ==");
    println!("bisection transactions: {} (post_mid+respond pairs) + 3 setup + 1 verify", rounds * 2);
    println!("verify_step calldata:   {} bytes", calldata);
    println!(
        "verify_step gas:        computation {} + storage {} − rebate {} = net {} MIST",
        g.computation,
        g.storage,
        g.rebate,
        g.computation as i64 + g.storage as i64 - g.rebate as i64
    );
    println!("pot paid to challenger: {:?}", tx["balanceChanges"]);

    // ---- Scenario C: stalling resolver loses by on-chain timeout ----------
    println!("\n⛓ scenario C: dishonest resolver asserts, gets challenged, then goes silent");
    cli.switch(&resolver);
    let bond_r2 = cli.split_bond(BOND);
    let tx = cli
        .call(&resolver, "assert_fact", &assert_args(&s, &dishonest, &bond_r2, 600_000, 2_000))
        .expect("assert_fact C");
    let fact_c = created_of_type(&tx, "::dispute::Fact").expect("Fact id");
    cli.switch(&challenger);
    let bond_c2 = cli.split_bond(BOND);
    cli.call(&challenger, "challenge", &[fact_c.clone(), bond_c2, CLOCK.into()])
        .expect("challenge C");
    println!("   resolver owes the first midpoint; sleeping past the 2s move deadline…");
    std::thread::sleep(std::time::Duration::from_millis(3_500));
    cli.call(&challenger, "claim_timeout", &[fact_c.clone(), CLOCK.into()])
        .expect("claim_timeout");
    let st = cli.object_fields(&fact_c)["status"].as_u64().unwrap();
    assert_eq!(st, 4, "REJECTED by timeout");
    println!("   ✓ staller slashed by the clock (status 4)");

    println!("\n⚖ one flipped bit → caught and slashed by a real chain. Phase 2 complete.");
}

fn assert_args(
    s: &ToySetup,
    p: &Party,
    bond_coin: &str,
    window_ms: u64,
    timeout_ms: u64,
) -> Vec<String> {
    let c = p.claim();
    vec![
        toy_mem_depth().to_string(),
        s.compiled.p.to_string(),
        hex_arg(&s.judge.program_root),
        hex_arg(&s.genesis_root),
        s.lay.output.to_string(),
        c.n.to_string(),
        hex_arg(&c.root_n),
        hex_arg(&c.output),
        bond_coin.to_string(),
        window_ms.to_string(),
        timeout_ms.to_string(),
        CLOCK.into(),
    ]
}

fn toy_mem_depth() -> u8 {
    toy_model::layout::MEM_DEPTH
}

/// Flatten a StepProof into CLI args; returns (args, payload byte count).
fn verify_args(fact: &str, proof: &StepProof) -> (Vec<String>, usize) {
    let open = |o: &Option<vm::onestep::PageOpening>| -> (String, String, usize) {
        match o {
            None => ("[]".into(), "[]".into(), 0),
            Some(o) => (
                hex_arg(&o.page),
                hex_vec_arg(&o.siblings),
                o.page.len() + 32 * o.siblings.len(),
            ),
        }
    };
    let (pa, sa, ba) = open(&proof.open_a);
    let (pb, sb, bb) = open(&proof.open_b);
    let (pw, sw, bw) = open(&proof.open_w);
    let calldata =
        45 + 32 + 96 + 32 * proof.instr_siblings.len() + ba + bb + bw;
    (
        vec![
            fact.to_string(),
            hex_arg(&proof.regs),
            hex_arg(&proof.mem_root),
            hex_arg(&proof.instr),
            hex_vec_arg(&proof.instr_siblings),
            pa,
            sa,
            pb,
            sb,
            pw,
            sw,
        ],
        calldata,
    )
}
