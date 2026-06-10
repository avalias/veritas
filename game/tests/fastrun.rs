//! Conformance C-14 (SPEC §11): the native checkpoint-mode runtime must
//! reproduce the per-step oracle's committed state byte-for-byte at every
//! checkpoint. This is the test that licenses running the honest path at
//! native speed — if it holds, the interpreter only ever runs in disputes.

use game::setup::ToySetup;
use game::trace::trace_full;
use toy_model::forward;

#[test]
fn c14_native_checkpoint_roots_equal_oracle() {
    let s = ToySetup::new("Zero overhead?", 5);
    // Oracle: full per-step trace (the slow, eager path).
    let mut m = s.machine();
    let oracle = trace_full(&mut m, None, s.lay.output, 10_000_000);

    // Native: fast flat-memory forward + dirty-page checkpoint hashing.
    let boundaries: Vec<(u64, u32)> = s
        .compiled
        .token_boundaries
        .iter()
        .copied()
        .zip(s.compiled.boundary_pcs.iter().copied())
        .collect();
    let native = forward::run_committed(
        &s.lay,
        s.image.clone(),
        s.compiled.n_prompt,
        s.compiled.n_gen,
        &boundaries,
    );

    // Every boundary root must match the oracle's root at that exact step.
    assert_eq!(native.boundary_roots.len(), boundaries.len());
    for (i, (step, _)) in boundaries.iter().enumerate() {
        assert_eq!(
            native.boundary_roots[i],
            oracle.root_at(*step),
            "checkpoint {i} (step {step}) diverged"
        );
    }
    // Final halted state and output bytes too.
    assert_eq!(native.final_root, *oracle.roots.last().unwrap(), "final root");
    assert_eq!(native.output, oracle.output, "output bytes");
    assert_eq!(native.tokens.len(), 5);

    // And the pure (no-hashing) run decodes identically.
    let pure = forward::run_pure(&s.lay, s.image.clone(), s.compiled.n_prompt, s.compiled.n_gen);
    assert_eq!(pure, native.tokens);
}
