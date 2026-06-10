//! Traces and fault injection (brief Phase 1.3/1.4).

use vm::exec::{Machine, StepOutcome};
use vm::hash::Hash;

/// A deliberately corrupted transition: the step `step → step+1` executes
/// normally, then the state is mutated, so every honest party's trace
/// diverges from index `step + 1` onward.
#[derive(Clone, Copy, Debug)]
pub struct Fault {
    pub step: u64,
    pub kind: FaultKind,
}

#[derive(Clone, Copy, Debug)]
pub enum FaultKind {
    /// Flip one accumulator bit ("corrupt one micro-op result").
    FlipAccBit(u8),
    /// Flip one bit of one memory byte ("flip a bit in one tensor").
    /// In a never-rewritten region (weights) the corruption provably
    /// reaches the final root.
    FlipMemBit { addr: u64, bit: u8 },
}

fn apply(m: &mut Machine, kind: FaultKind) {
    match kind {
        FaultKind::FlipAccBit(b) => m.regs.acc ^= 1i64 << (b % 64),
        FaultKind::FlipMemBit { addr, bit } => {
            let v = m.mem.read_u8(addr) ^ (1 << (bit % 8));
            m.mem.write(addr, &[v]);
        }
    }
}

/// Full per-step trace: `roots[i]` = state root at step i (0..=N).
pub struct Trace {
    pub roots: Vec<Hash>,
    pub n: u64,
    pub outcome: StepOutcome,
    /// Raw bytes of the output region `[n u32][ids…]` at the final state.
    pub output: Vec<u8>,
}

impl Trace {
    pub fn root_at(&self, step: u64) -> Hash {
        self.roots[step as usize]
    }
}

/// Run to terminal recording every root, optionally injecting `fault`.
pub fn trace_full(m: &mut Machine, fault: Option<Fault>, out_base: u64, max: u64) -> Trace {
    let mut roots = vec![m.state_root()];
    let outcome = loop {
        assert!((roots.len() as u64) <= max, "step budget exceeded");
        let s = m.regs.step;
        let out = m.step().expect("terminal mid-trace");
        if let Some(f) = fault {
            if f.step == s {
                apply(m, f.kind);
            }
        }
        roots.push(m.state_root());
        if out != StepOutcome::Ran {
            break out;
        }
    };
    let n_out = m.mem.read_u32(out_base) as usize;
    let mut output = Vec::with_capacity(4 + 4 * n_out);
    for i in 0..4 + 4 * n_out as u64 {
        output.push(m.mem.read_u8(out_base + i));
    }
    Trace { n: m.regs.step, roots, outcome, output }
}

/// Replay (with the same fault) to exactly `step` — used to materialize the
/// pre-state a one-step proof opens against.
pub fn replay_to(mut m: Machine, fault: Option<Fault>, step: u64) -> Machine {
    while m.regs.step < step {
        let s = m.regs.step;
        m.step().expect("replay hit terminal early");
        if let Some(f) = fault {
            if f.step == s {
                apply(&mut m, f.kind);
            }
        }
    }
    m
}

/// Checkpoint trace: roots every `k` steps plus the final root
/// (brief Phase 1.3 "checkpoint every K micro-ops, configurable").
pub struct CkptTrace {
    pub k: u64,
    pub roots: Vec<(u64, Hash)>,
}

pub fn trace_every_k(m: &mut Machine, k: u64, max: u64) -> CkptTrace {
    assert!(k >= 1);
    let mut roots = vec![(0, m.state_root())];
    loop {
        assert!(m.regs.step <= max, "step budget exceeded");
        let out = m.step().expect("terminal mid-trace");
        if m.regs.step.is_multiple_of(k) || out != StepOutcome::Ran {
            roots.push((m.regs.step, m.state_root()));
        }
        if out != StepOutcome::Ran {
            return CkptTrace { k, roots };
        }
    }
}
