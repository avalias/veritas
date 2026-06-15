//! MockChain: the dispute protocol under logical-tick time (SPEC §8, brief
//! Phase 1.5). The Move package replaces this verbatim in Phase 2; the
//! one-step verdict already comes from the shared `vm::onestep` twin.

use crate::actors::Claim;
use vm::hash::{state_root, Hash};
use vm::merkle::fold_proof;
use vm::onestep::{verify_step, JudgeParams, PageOpening, ProofError, StepProof, Verdict};
use vm::state::{Registers, HALTED, REG_ENC_LEN};
use vm::PAGE_SIZE;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Resolver,
    Challenger,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// Asserted, challenge window open.
    Open,
    /// Bisection in progress.
    Challenged,
    /// Output-binding challenge in progress (SPEC §8.5).
    OutputChallenged,
    /// Claim stands; resolver paid.
    Finalized,
    /// Claim slashed; challenger paid.
    Rejected,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChainError {
    WrongStatus,
    WindowStillOpen,
    WindowClosed,
    NotYourTurn,
    NoPendingMid,
    IntervalNotAtomic,
    IntervalAtomic,
    DeadlineNotPassed,
    InvalidProof(ProofError),
    BadFinalStateProof,
}

/// Bisection state (SPEC §8.3). Invariant: agreed at `lo`, disputed at `hi`.
#[derive(Clone, Debug)]
pub struct Dispute {
    pub lo: u64,
    pub hi: u64,
    pub root_lo: Hash,
    pub root_hi: Hash,
    pub pending_mid: Option<Hash>,
    pub mover: Role,
    pub deadline: u64,
    pub rounds: u32,
}

#[derive(Clone, Debug)]
pub struct Fact {
    pub judge: JudgeParams,
    pub genesis_root: Hash,
    /// Output region base (final-state challenge needs it; SPEC §8.5).
    pub out_base: u64,
    pub claim: Claim,
    pub status: Status,
    pub created: u64,
    pub dispute: Option<Dispute>,
    pub output_deadline: u64,
    /// Net bond deltas after resolution (winner +bond, loser −bond).
    pub resolver_delta: i64,
    pub challenger_delta: i64,
}

#[derive(Clone, Copy, Debug)]
pub struct Params {
    pub bond: u64,
    pub challenge_window: u64,
    pub move_timeout: u64,
}

impl Default for Params {
    fn default() -> Self {
        // Logical ticks; economics are deployment params (SPEC §8.2, Q3).
        Self { bond: 100, challenge_window: 50, move_timeout: 10 }
    }
}

pub struct MockChain {
    pub now: u64,
    pub params: Params,
}

/// Resolver's revelation for the output-binding challenge (SPEC §8.5).
pub struct FinalStateProof {
    pub regs: [u8; REG_ENC_LEN],
    pub mem_root: Hash,
    /// Openings covering the output region pages, in ascending page order.
    pub pages: Vec<(u64, PageOpening)>,
}

impl MockChain {
    pub fn new(params: Params) -> Self {
        Self { now: 0, params }
    }

    pub fn tick(&mut self, dt: u64) {
        self.now += dt;
    }

    pub fn assert_fact(
        &self,
        judge: JudgeParams,
        genesis_root: Hash,
        out_base: u64,
        claim: Claim,
    ) -> Fact {
        Fact {
            judge,
            genesis_root,
            out_base,
            claim,
            status: Status::Open,
            created: self.now,
            dispute: None,
            output_deadline: 0,
            resolver_delta: 0,
            challenger_delta: 0,
        }
    }

    pub fn finalize(&self, fact: &mut Fact) -> Result<(), ChainError> {
        if fact.status != Status::Open {
            return Err(ChainError::WrongStatus);
        }
        if self.now < fact.created + self.params.challenge_window {
            return Err(ChainError::WindowStillOpen);
        }
        fact.status = Status::Finalized;
        Ok(())
    }

    /// Open the bisection game. Both bonds are escrowed from here on.
    pub fn challenge(&self, fact: &mut Fact) -> Result<(), ChainError> {
        if fact.status != Status::Open {
            return Err(ChainError::WrongStatus);
        }
        if self.now >= fact.created + self.params.challenge_window {
            return Err(ChainError::WindowClosed);
        }
        fact.status = Status::Challenged;
        fact.dispute = Some(Dispute {
            lo: 0,
            hi: fact.claim.n,
            root_lo: fact.genesis_root, // agreed by construction (SPEC §7.2)
            root_hi: fact.claim.root_n,
            pending_mid: None,
            mover: Role::Resolver,
            deadline: self.now + self.params.move_timeout,
            rounds: 0,
        });
        Ok(())
    }

    /// Resolver posts the midpoint root of the current interval.
    pub fn post_mid(&self, fact: &mut Fact, root: Hash) -> Result<u64, ChainError> {
        let timeout = self.params.move_timeout;
        let d = dispute_mut(fact)?;
        if d.mover != Role::Resolver {
            return Err(ChainError::NotYourTurn);
        }
        if d.hi - d.lo <= 1 {
            return Err(ChainError::IntervalAtomic);
        }
        d.pending_mid = Some(root);
        d.mover = Role::Challenger;
        d.deadline = self.now + timeout;
        Ok(d.lo + (d.hi - d.lo) / 2)
    }

    /// Challenger agrees/disagrees with the posted midpoint root.
    pub fn respond(&self, fact: &mut Fact, agree: bool) -> Result<(), ChainError> {
        let timeout = self.params.move_timeout;
        let d = dispute_mut(fact)?;
        if d.mover != Role::Challenger {
            return Err(ChainError::NotYourTurn);
        }
        let mid_root = d.pending_mid.take().ok_or(ChainError::NoPendingMid)?;
        let mid = d.lo + (d.hi - d.lo) / 2;
        if agree {
            d.lo = mid;
            d.root_lo = mid_root;
        } else {
            d.hi = mid;
            d.root_hi = mid_root;
        }
        d.mover = Role::Resolver;
        d.deadline = self.now + timeout;
        d.rounds += 1;
        Ok(())
    }

    /// Final one-step verification — submitter-independent; the comparison
    /// decides (SPEC §8.3/§8.4). Invalid proofs abort without state change.
    pub fn submit_proof(&self, fact: &mut Fact, proof: &StepProof) -> Result<Role, ChainError> {
        let (root_lo, root_hi) = {
            let d = dispute_mut(fact)?;
            if d.hi - d.lo != 1 {
                return Err(ChainError::IntervalNotAtomic);
            }
            (d.root_lo, d.root_hi)
        };
        let verdict = verify_step(&root_lo, &root_hi, &fact.judge, proof)
            .map_err(ChainError::InvalidProof)?;
        let winner = match verdict {
            Verdict::ResolverWins => Role::Resolver,
            Verdict::ChallengerWins => Role::Challenger,
        };
        self.settle(fact, winner);
        Ok(winner)
    }

    /// Whoever owes the next move and missed the deadline loses (SPEC §8.2).
    /// At the atomic interval the resolver owes the proof (the challenger
    /// MAY preempt by submitting one).
    pub fn claim_timeout(&self, fact: &mut Fact) -> Result<Role, ChainError> {
        if fact.status == Status::OutputChallenged {
            if self.now <= fact.output_deadline {
                return Err(ChainError::DeadlineNotPassed);
            }
            self.settle(fact, Role::Challenger);
            return Ok(Role::Challenger);
        }
        let d = dispute_mut(fact)?;
        if self.now <= d.deadline {
            return Err(ChainError::DeadlineNotPassed);
        }
        let staller = if d.hi - d.lo == 1 { Role::Resolver } else { d.mover };
        let winner = other(staller);
        self.settle(fact, winner);
        Ok(winner)
    }

    /// Output-binding challenge (SPEC §8.5): make the resolver reveal the
    /// final state and prove the claimed output bytes live there.
    pub fn challenge_output(&self, fact: &mut Fact) -> Result<(), ChainError> {
        if fact.status != Status::Open {
            return Err(ChainError::WrongStatus);
        }
        if self.now >= fact.created + self.params.challenge_window {
            return Err(ChainError::WindowClosed);
        }
        fact.status = Status::OutputChallenged;
        fact.output_deadline = self.now + self.params.move_timeout;
        Ok(())
    }

    /// Resolver's response to `challenge_output`. A reveal that fails the
    /// checks slashes the resolver immediately (the claim is disproven by
    /// the resolver's own state); an honest reveal slashes the challenger.
    pub fn reveal_final_state(
        &self,
        fact: &mut Fact,
        fsp: &FinalStateProof,
    ) -> Result<Role, ChainError> {
        if fact.status != Status::OutputChallenged {
            return Err(ChainError::WrongStatus);
        }
        // Preimage of the claimed final root.
        if state_root(&fsp.mem_root, &fsp.regs) != fact.claim.root_n {
            return Err(ChainError::BadFinalStateProof); // garbage: no decision
        }
        let regs = Registers::decode(&fsp.regs);
        let ok = regs.halted == HALTED
            && regs.step == fact.claim.n
            && self.output_matches(fact, fsp);
        let winner = if ok { Role::Resolver } else { Role::Challenger };
        self.settle(fact, winner);
        Ok(winner)
    }

    fn output_matches(&self, fact: &Fact, fsp: &FinalStateProof) -> bool {
        // Verify each supplied page against mem_root, then compare the
        // claimed output bytes to the region content.
        let want = &fact.claim.output;
        if want.len() < 4 || want.len() > PAGE_SIZE {
            return false; // toy cap: output region is one page
        }
        let page_index = fact.out_base / PAGE_SIZE as u64;
        let Some((idx, opening)) = fsp.pages.first() else {
            return false;
        };
        if *idx != page_index
            || opening.page.len() != PAGE_SIZE
            || opening.siblings.len() != fact.judge.d as usize
            || fold_proof(
                vm::hash::page_leaf_hash(&opening.page),
                *idx,
                &opening.siblings,
            ) != fsp.mem_root
        {
            return false;
        }
        let off = (fact.out_base % PAGE_SIZE as u64) as usize;
        &opening.page[off..off + want.len()] == want.as_slice()
    }

    fn settle(&self, fact: &mut Fact, winner: Role) {
        let b = self.params.bond as i64;
        let (r, c) = match winner {
            Role::Resolver => (b, -b),
            Role::Challenger => (-b, b),
        };
        fact.resolver_delta = r;
        fact.challenger_delta = c;
        fact.status = if winner == Role::Resolver {
            Status::Finalized
        } else {
            Status::Rejected
        };
    }
}

pub fn other(r: Role) -> Role {
    match r {
        Role::Resolver => Role::Challenger,
        Role::Challenger => Role::Resolver,
    }
}

fn dispute_mut(fact: &mut Fact) -> Result<&mut Dispute, ChainError> {
    if fact.status != Status::Challenged {
        return Err(ChainError::WrongStatus);
    }
    fact.dispute.as_mut().ok_or(ChainError::WrongStatus)
}
