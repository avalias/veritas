//! Per-step trace utilities (SPEC §7.4, §9.1).
//!
//! `trace_digest` implements the conformance digest of SPEC §9.1:
//! `H(root_0 ‖ root_1 ‖ … ‖ root_N)` — untagged, a test artifact rather
//! than an on-chain object. Phase 1 adds checkpointed traces; this module
//! is the level-0 (per-step) machinery.

use crate::exec::{Machine, StepOutcome};
use crate::hash::Hash;
use sha3::{Digest, Sha3_256};

#[derive(Debug, PartialEq, Eq)]
pub enum RunError {
    /// The machine did not reach a terminal state within the step budget.
    StepLimit,
}

#[derive(Debug)]
pub struct RunResult {
    /// Final value of the step register == trace length N.
    pub steps: u64,
    pub outcome: StepOutcome,
    pub final_root: Hash,
}

/// Run until HALT/TRAP without materializing roots (checkpoint-mode shape).
pub fn run_to_terminal(m: &mut Machine, max_steps: u64) -> Result<RunResult, RunError> {
    let mut executed = 0u64;
    loop {
        if executed >= max_steps {
            return Err(RunError::StepLimit);
        }
        let outcome = m.step().expect("run_to_terminal called on terminal state");
        executed += 1;
        if outcome != StepOutcome::Ran {
            return Ok(RunResult {
                steps: m.regs.step,
                outcome,
                final_root: m.state_root(),
            });
        }
    }
}

/// Run to terminal, streaming every state root (incl. root_0) into the
/// conformance digest. This is the cross-platform golden of Invariant 1.
pub fn trace_digest(m: &mut Machine, max_steps: u64) -> Result<(Hash, RunResult), RunError> {
    let mut h = Sha3_256::new();
    h.update(m.state_root());
    let mut executed = 0u64;
    loop {
        if executed >= max_steps {
            return Err(RunError::StepLimit);
        }
        let outcome = m.step().expect("trace_digest called on terminal state");
        executed += 1;
        h.update(m.state_root());
        if outcome != StepOutcome::Ran {
            return Ok((
                h.finalize().into(),
                RunResult {
                    steps: m.regs.step,
                    outcome,
                    final_root: m.state_root(),
                },
            ));
        }
    }
}

/// Materialize every root (root_0 ..= root_N). Dispute-segment shape; tests
/// use it to cross-check `trace_digest`.
pub fn per_step_roots(m: &mut Machine, max_steps: u64) -> Result<(Vec<Hash>, RunResult), RunError> {
    let mut roots = vec![m.state_root()];
    let mut executed = 0u64;
    loop {
        if executed >= max_steps {
            return Err(RunError::StepLimit);
        }
        let outcome = m.step().expect("per_step_roots called on terminal state");
        executed += 1;
        roots.push(m.state_root());
        if outcome != StepOutcome::Ran {
            return Ok((
                roots,
                RunResult {
                    steps: m.regs.step,
                    outcome,
                    final_root: m.state_root(),
                },
            ));
        }
    }
}
