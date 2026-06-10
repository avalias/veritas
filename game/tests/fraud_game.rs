//! Phase 1 acceptance tests (brief Phase 1.6):
//!  - honest resolver, no challenger → finalizes after the window
//!  - dishonest resolver (random fault, 100 fuzz seeds) vs honest
//!    challenger → challenger always wins AND the isolated step equals the
//!    injected fault step
//!  - honest resolver vs lying challenger → resolver always wins
//!  - stalling party → loses by timeout
//!  - output tampering → slashed via the final-state challenge
//!  - golden trace digest of the full model+compiler+VM stack

use game::actors::Party;
use game::chain::{ChainError, FinalStateProof, MockChain, Params, Role, Status};
use game::driver::{run_dispute, Outcome, Policy};
use game::setup::ToySetup;
use game::trace::{replay_to, trace_every_k, Fault, FaultKind};
use std::sync::OnceLock;
use vm::fixtures::XorShift64;
use vm::onestep::PageOpening;
use vm::PAGE_SIZE;

/// One shared setup + honest trace for the whole suite (all deterministic).
fn setup() -> &'static (ToySetup, Party) {
    static S: OnceLock<(ToySetup, Party)> = OnceLock::new();
    S.get_or_init(|| {
        // Short prompt: the fuzz traces 100 full inferences; length here
        // multiplies the whole suite's runtime.
        let s = ToySetup::new("Rain?", 4);
        let honest = Party::new(&s, None);
        (s, honest)
    })
}

fn fresh_fact(chain: &MockChain, s: &ToySetup, p: &Party) -> game::chain::Fact {
    chain.assert_fact(s.judge.clone(), s.genesis_root, s.lay.output, p.claim())
}

#[test]
fn honest_resolver_unchallenged_finalizes_after_window() {
    let (s, honest) = setup();
    let mut chain = MockChain::new(Params::default());
    let mut fact = fresh_fact(&chain, s, honest);
    // Too early: the window must protect challengers.
    assert_eq!(chain.finalize(&mut fact), Err(ChainError::WindowStillOpen));
    chain.tick(chain.params.challenge_window);
    chain.finalize(&mut fact).unwrap();
    assert_eq!(fact.status, Status::Finalized);
    // And the window really closed: late challenges bounce.
    assert_eq!(chain.challenge(&mut fact), Err(ChainError::WrongStatus));
}

#[test]
fn dishonest_resolver_fuzz_100_seeds_challenger_always_wins() {
    let (s, honest) = setup();
    let n = honest.trace.n;
    let mut rng = XorShift64::new(0xFA017);
    // Weight-region byte addresses: corruption there provably persists to
    // the final root (weights are never rewritten).
    let w_lo = s.lay.emb;
    let w_hi = s.lay.head + (96 * 64) as u64;

    for seed in 0..100u64 {
        let step = rng.next_u64() % n; // any step, including 0 and N−1
        let kind = if seed % 2 == 0 {
            FaultKind::FlipMemBit {
                addr: w_lo + rng.next_u64() % (w_hi - w_lo),
                bit: (rng.next_u64() % 8) as u8,
            }
        } else {
            FaultKind::FlipAccBit((rng.next_u64() % 64) as u8)
        };
        let fault = Fault { step, kind };
        let dishonest = Party::new(s, Some(fault));
        assert_ne!(
            dishonest.root_at(step + 1),
            honest.root_at(step + 1),
            "seed {seed}: fault must corrupt the post-state immediately"
        );
        if dishonest.claim() == honest.claim() {
            // An acc-flip can wash out (e.g. immediately before LDC): the
            // "dishonest" claim is then simply correct — nothing to dispute.
            // Soundness is unaffected; skip (weight-flips never wash out).
            continue;
        }
        let mut chain = MockChain::new(Params::default());
        let mut fact = fresh_fact(&chain, s, &dishonest);
        let Outcome { winner, isolated_lo, rounds, by_timeout } = run_dispute(
            &mut chain,
            &mut fact,
            s,
            &dishonest,
            honest,
            Policy::Honest,
            Policy::Honest,
            false,
        );
        assert_eq!(winner, Role::Challenger, "seed {seed}");
        assert!(!by_timeout, "seed {seed}: must win on the merits");
        assert_eq!(fact.status, Status::Rejected, "seed {seed}");
        // The bisection isolated EXACTLY the corrupted transition. The
        // first divergent root is step+1 even if traces later reconverge,
        // because honest-agreement tracks the longest honest prefix.
        let lo = isolated_lo.expect("reached one-step verification");
        assert_eq!(lo, step, "seed {seed}: isolated wrong step");
        assert!(rounds as u64 >= 64u64.ilog2() as u64, "seed {seed}: rounds {rounds}");
        // Slashing: zero-sum bonds, challenger up, resolver down.
        assert_eq!(fact.resolver_delta, -(chain.params.bond as i64), "seed {seed}");
        assert_eq!(fact.challenger_delta, chain.params.bond as i64, "seed {seed}");
    }
}

#[test]
fn lying_challenger_always_loses() {
    let (s, honest) = setup();
    let n = honest.trace.n;
    let mut rng = XorShift64::new(0x11A2);
    for seed in 0..10u64 {
        // The challenger's worldview is corrupted; the resolver is honest.
        let fault = Fault {
            step: rng.next_u64() % n,
            kind: FaultKind::FlipAccBit((rng.next_u64() % 64) as u8),
        };
        let liar = Party::new(s, Some(fault));
        if liar.root_at(n) == honest.root_at(n) {
            continue; // fault washed out: liar has no disagreement to press
        }
        let mut chain = MockChain::new(Params::default());
        let mut fact = fresh_fact(&chain, s, honest);
        let out = run_dispute(
            &mut chain,
            &mut fact,
            s,
            honest,
            &liar,
            Policy::Honest,
            Policy::Honest,
            false,
        );
        assert_eq!(out.winner, Role::Resolver, "seed {seed}");
        assert_eq!(fact.status, Status::Finalized);
        assert_eq!(fact.resolver_delta, chain.params.bond as i64);
        assert_eq!(fact.challenger_delta, -(chain.params.bond as i64));
    }
}

#[test]
fn stalling_party_loses_by_timeout() {
    let (s, honest) = setup();
    let fault = Fault { step: 1000, kind: FaultKind::FlipAccBit(3) };
    let dishonest = Party::new(s, Some(fault));

    // Dishonest resolver stops posting mids after 3 rounds.
    let mut chain = MockChain::new(Params::default());
    let mut fact = fresh_fact(&chain, s, &dishonest);
    let out = run_dispute(
        &mut chain, &mut fact, s, &dishonest, honest,
        Policy::StallAfterRounds(3), Policy::Honest, false,
    );
    assert_eq!(out.winner, Role::Challenger);
    assert!(out.by_timeout);

    // Lying challenger goes silent mid-game against an honest resolver.
    let mut chain = MockChain::new(Params::default());
    let mut fact = fresh_fact(&chain, s, honest);
    let out = run_dispute(
        &mut chain, &mut fact, s, honest, &dishonest,
        Policy::Honest, Policy::StallAfterRounds(2), false,
    );
    assert_eq!(out.winner, Role::Resolver);
    assert!(out.by_timeout);

    // Resolver who never produces the final proof loses even after playing
    // the whole bisection (the proof obligation is the resolver's).
    let mut chain = MockChain::new(Params::default());
    let mut fact = fresh_fact(&chain, s, &dishonest);
    let out = run_dispute(
        &mut chain, &mut fact, s, &dishonest, &dishonest,
        Policy::StallAtProof, Policy::StallAtProof, false,
    );
    assert_eq!(out.winner, Role::Challenger);
    assert!(out.by_timeout);
}

#[test]
fn premature_timeout_claims_are_rejected() {
    let (s, honest) = setup();
    let chain = MockChain::new(Params::default());
    let mut fact = fresh_fact(&chain, s, honest);
    chain.challenge(&mut fact).unwrap();
    assert_eq!(
        chain.claim_timeout(&mut fact),
        Err(ChainError::DeadlineNotPassed),
        "clock games must not work"
    );
}

#[test]
fn output_tamper_slashed_via_final_state_challenge() {
    let (s, honest) = setup();
    let mut chain = MockChain::new(Params::default());
    // Correct trace, lying output field.
    let mut claim = honest.claim();
    claim.output[5] ^= 0x01; // flip a bit in the first token id
    let mut fact = chain.assert_fact(s.judge.clone(), s.genesis_root, s.lay.output, claim);
    chain.challenge_output(&mut fact).unwrap();
    // The resolver's own honest reveal now disproves its claimed output.
    let fsp = final_state_proof(s, honest);
    let winner = chain.reveal_final_state(&mut fact, &fsp).unwrap();
    assert_eq!(winner, Role::Challenger);
    assert_eq!(fact.status, Status::Rejected);

    // Control: honest claim survives a frivolous output challenge.
    let mut fact = fresh_fact(&chain, s, honest);
    chain.challenge_output(&mut fact).unwrap();
    let winner = chain.reveal_final_state(&mut fact, &fsp).unwrap();
    assert_eq!(winner, Role::Resolver);
    assert_eq!(fact.status, Status::Finalized);

    // Stonewalling the reveal also loses.
    let mut fact = fresh_fact(&chain, s, honest);
    chain.challenge_output(&mut fact).unwrap();
    chain.tick(chain.params.move_timeout + 1);
    assert_eq!(chain.claim_timeout(&mut fact).unwrap(), Role::Challenger);
}

fn final_state_proof(s: &ToySetup, p: &Party) -> FinalStateProof {
    let m = replay_to(s.machine(), p.fault, p.trace.n);
    let page_index = s.lay.output / PAGE_SIZE as u64;
    FinalStateProof {
        regs: m.regs.encode(),
        mem_root: m.mem.root(),
        pages: vec![(
            page_index,
            PageOpening {
                page: m.mem.page(page_index).to_vec(),
                siblings: m.mem.prove_page(page_index),
            },
        )],
    }
}

#[test]
fn checkpoint_trace_is_a_subsample_of_full_trace() {
    let (s, honest) = setup();
    let mut m = s.machine();
    let ck = trace_every_k(&mut m, 1000, 10_000_000);
    assert_eq!(ck.roots.first().unwrap().0, 0);
    assert_eq!(ck.roots.last().unwrap().0, honest.trace.n);
    for (step, root) in &ck.roots {
        assert_eq!(*root, honest.root_at(*step), "checkpoint at {step}");
    }
}

/// Cross-platform golden for the WHOLE stack (weights → compiler → VM →
/// trace). Pinned from the first run; CI re-derives it on x86 and ARM.
#[test]
fn golden_toy_judgment() {
    let (s, honest) = setup();
    const GOLDEN_N: u64 = 85937;
    const GOLDEN_ROOT_N: &str =
        "ce96af31892d58eca566551753a77b41cb2110cad999fb02b57c6b4355419f82";
    // Output region [n=4][ids 19, 66, 14, 33] — "5b.A" detokenized.
    const GOLDEN_OUTPUT: [u8; 20] = [
        4, 0, 0, 0, 19, 0, 0, 0, 66, 0, 0, 0, 14, 0, 0, 0, 33, 0, 0, 0,
    ];
    let hex: String = honest.claim().root_n.iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(honest.trace.n, GOLDEN_N);
    assert_eq!(hex, GOLDEN_ROOT_N);
    assert_eq!(honest.claim().output, GOLDEN_OUTPUT);
    assert_eq!(s.compiled.total_steps, GOLDEN_N);
}
