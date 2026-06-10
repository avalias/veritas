//! `cargo run --bin demo` (brief Phase 1.7): narrates one honest
//! resolution and one full fraud dispute — bisection rounds, the shrinking
//! interval, the final one-step verification, and the slash.

use game::actors::Party;
use game::chain::{MockChain, Params, Status};
use game::driver::{run_dispute, Policy};
use game::setup::ToySetup;
use game::trace::{Fault, FaultKind};
use toy_model::model::detokenize;

fn main() {
    let prompt = "Will it rain tomorrow? Evidence: dark clouds. Answer:";
    println!("┌─ fraud-provable inference demo ─ toy judge, MockChain ─┐");
    println!("│ prompt: {prompt:?}");
    let setup = ToySetup::new(prompt, 6);
    println!(
        "│ program: {} instructions (tree depth {}), memory 1 MiB",
        setup.compiled.program.len(),
        setup.compiled.p
    );

    // -- Act 1: honest resolution -----------------------------------------
    println!("├─ act 1: honest resolver, no challenger");
    let honest = Party::new(&setup, None);
    let claim = honest.claim();
    let ids: Vec<u32> = claim.output[4..]
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    println!(
        "│ executed {} micro-ops; judgment tokens {:?} = {:?} (random weights — gibberish is expected)",
        claim.n,
        ids,
        detokenize(&ids)
    );
    let mut chain = MockChain::new(Params::default());
    let mut fact = chain.assert_fact(setup.judge.clone(), setup.genesis_root, setup.lay.output, claim);
    chain.tick(chain.params.challenge_window);
    chain.finalize(&mut fact).unwrap();
    println!("│ challenge window passed → finalized. bond returned.");

    // -- Act 2: fraud + dispute -------------------------------------------
    let fault = Fault {
        step: honest.trace.n / 3,
        kind: FaultKind::FlipMemBit { addr: setup.lay.wq[1] + 1234, bit: 5 },
    };
    println!("├─ act 2: resolver corrupts one weight bit at step {} and asserts the result", fault.step);
    let dishonest = Party::new(&setup, Some(fault));
    let mut chain = MockChain::new(Params::default());
    let mut fact = chain.assert_fact(
        setup.judge.clone(),
        setup.genesis_root,
        setup.lay.output,
        dishonest.claim(),
    );
    println!("│ honest challenger posts bond; bisection over [0, {}]:", dishonest.claim().n);
    let out = run_dispute(
        &mut chain,
        &mut fact,
        &setup,
        &dishonest,
        &honest,
        Policy::Honest,
        Policy::Honest,
        true,
    );
    println!(
        "│ verdict: {:?} after {} rounds — isolated step {} (injected fault step {})",
        out.winner,
        out.rounds,
        out.isolated_lo.unwrap(),
        fault.step
    );
    println!(
        "│ status {:?}; resolver bond delta {:+}, challenger {:+}",
        fact.status, fact.resolver_delta, fact.challenger_delta
    );
    assert_eq!(fact.status, Status::Rejected);
    assert_eq!(out.isolated_lo.unwrap(), fault.step);
    println!("└─ one corrupted bit, caught by one on-chain micro-op. ─┘");
}
