//! Resolver and Challenger (brief Phase 1.4): both wrap the same VM but may
//! hold different traces — honesty is exactly "fault: None".

use crate::setup::ToySetup;
use crate::trace::{replay_to, trace_full, Fault, Trace};
use vm::hash::Hash;
use vm::onestep::{build_step_proof, StepProof};

/// What a resolver asserts on-chain (SPEC §8.1 claim).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Claim {
    pub n: u64,
    pub root_n: Hash,
    pub output: Vec<u8>,
}

/// A protocol participant. Role (resolver/challenger) is decided by how the
/// driver uses it; `fault: None` is an honest party.
pub struct Party {
    pub trace: Trace,
    pub fault: Option<Fault>,
}

impl Party {
    pub fn new(setup: &ToySetup, fault: Option<Fault>) -> Self {
        let mut m = setup.machine();
        let trace = trace_full(&mut m, fault, setup.lay.output, 10_000_000);
        Self { trace, fault }
    }

    pub fn claim(&self) -> Claim {
        Claim {
            n: self.trace.n,
            root_n: *self.trace.roots.last().unwrap(),
            output: self.trace.output.clone(),
        }
    }

    pub fn root_at(&self, step: u64) -> Hash {
        self.trace.root_at(step)
    }

    /// Materialize the pre-state at `lo` (replaying this party's own
    /// execution, fault included) and build the §8.4 payload.
    pub fn build_proof(&self, setup: &ToySetup, lo: u64) -> StepProof {
        let m = replay_to(setup.machine(), self.fault, lo);
        debug_assert_eq!(m.state_root(), self.root_at(lo), "replay mismatch");
        build_step_proof(&m, &setup.program_tree)
    }
}
