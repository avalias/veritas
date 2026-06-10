//! Dispute driver: plays both parties' strategies against the MockChain,
//! including deliberate stalling (for the timeout tests and the demo).

use crate::actors::Party;
use crate::chain::{Fact, MockChain, Role, Status};
use crate::setup::ToySetup;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    Honest,
    /// Stop making moves after this many bisection rounds.
    StallAfterRounds(u32),
    /// Play the bisection but never submit the final proof.
    StallAtProof,
}

pub struct Outcome {
    pub winner: Role,
    /// The atomic interval's lower end — the isolated step (when the game
    /// reached one-step verification).
    pub isolated_lo: Option<u64>,
    pub rounds: u32,
    pub by_timeout: bool,
}

fn stalls(p: Policy, rounds: u32, at_proof: bool) -> bool {
    match p {
        Policy::Honest => false,
        Policy::StallAfterRounds(r) => rounds >= r,
        Policy::StallAtProof => at_proof,
    }
}

/// Run a full dispute to settlement. `resolver`/`challenger` provide traces;
/// policies inject stalling. Narrates each round when `verbose`.
#[allow(clippy::too_many_arguments)] // test/demo orchestration, not API surface
pub fn run_dispute(
    chain: &mut MockChain,
    fact: &mut Fact,
    setup: &ToySetup,
    resolver: &Party,
    challenger: &Party,
    r_policy: Policy,
    c_policy: Policy,
    verbose: bool,
) -> Outcome {
    chain.challenge(fact).expect("challenge accepted");
    let mut isolated_lo = None;
    let mut by_timeout = false;
    loop {
        match fact.status {
            Status::Finalized | Status::Rejected => {
                let d = fact.dispute.as_ref().unwrap();
                return Outcome {
                    winner: if fact.status == Status::Finalized {
                        Role::Resolver
                    } else {
                        Role::Challenger
                    },
                    isolated_lo,
                    rounds: d.rounds,
                    by_timeout,
                };
            }
            Status::Challenged => {}
            _ => unreachable!("driver only runs bisection disputes"),
        }
        let d = fact.dispute.as_ref().unwrap().clone();
        if d.hi - d.lo == 1 {
            isolated_lo = Some(d.lo);
            if verbose {
                println!(
                    "  ⚖ interval atomic: step {} → {}; one-step verification",
                    d.lo, d.hi
                );
            }
            if stalls(r_policy, d.rounds, true) {
                // Honest challengers may preempt; otherwise the clock wins.
                if c_policy == Policy::Honest {
                    let proof = challenger.build_proof(setup, d.lo);
                    let w = chain.submit_proof(fact, &proof).expect("valid proof");
                    if verbose {
                        println!("  → challenger submitted the proof; winner: {w:?}");
                    }
                } else {
                    chain.tick(chain.params.move_timeout + 1);
                    let w = chain.claim_timeout(fact).expect("timeout claimable");
                    by_timeout = true;
                    if verbose {
                        println!("  → both stalled; timeout, winner: {w:?}");
                    }
                }
            } else {
                let proof = resolver.build_proof(setup, d.lo);
                let w = chain.submit_proof(fact, &proof).expect("valid proof");
                if verbose {
                    println!("  → resolver submitted the proof; winner: {w:?}");
                }
            }
            continue;
        }
        // Bisection round: resolver posts the midpoint root…
        if stalls(r_policy, d.rounds, false) {
            chain.tick(chain.params.move_timeout + 1);
            chain.claim_timeout(fact).expect("resolver stalled");
            by_timeout = true;
            continue;
        }
        let mid = d.lo + (d.hi - d.lo) / 2;
        chain
            .post_mid(fact, resolver.root_at(mid))
            .expect("post_mid accepted");
        // …and the challenger agrees or disagrees.
        if stalls(c_policy, d.rounds, false) {
            chain.tick(chain.params.move_timeout + 1);
            chain.claim_timeout(fact).expect("challenger stalled");
            by_timeout = true;
            continue;
        }
        let agree = challenger.root_at(mid) == resolver.root_at(mid);
        chain.respond(fact, agree).expect("respond accepted");
        if verbose {
            let d2 = fact.dispute.as_ref().unwrap();
            println!(
                "  round {:>2}: mid step {:>7} — challenger {} — interval [{}, {}] ({} steps)",
                d2.rounds,
                mid,
                if agree { "AGREES   " } else { "DISAGREES" },
                d2.lo,
                d2.hi,
                d2.hi - d2.lo
            );
        }
    }
}
