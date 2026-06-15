//! Compiler ⇄ VM equivalence: the predicted step count must equal the VM's
//! actual step register exactly (brief Phase 1.2: "output the micro-op
//! program + expected step count"), and decoding must be deterministic.

use compiler::compile_toy;
use toy_model::layout::{genesis_image, Layout, MEM_DEPTH};
use toy_model::model::{tokenize, ToyModel, WEIGHT_SEED};
use vm::exec::{Machine, StepOutcome};
use vm::trace::run_to_terminal;

fn build(prompt: &str, n_gen: usize) -> (Machine, compiler::Compiled) {
    let lay = Layout::new();
    let model = ToyModel::generate(WEIGHT_SEED);
    let toks = tokenize(prompt);
    let compiled = compile_toy(&lay, toks.len(), n_gen);
    let image = genesis_image(&lay, &model, &toks);
    let m = Machine::with_image(MEM_DEPTH, compiled.p, compiled.program.clone(), &image);
    (m, compiled)
}

/// Read the output region: [n u32][ids u32 …].
fn read_output(m: &Machine, lay: &Layout) -> Vec<u32> {
    let n = m.mem.read_u32(lay.output) as usize;
    (0..n).map(|i| m.mem.read_u32(lay.output + 4 + 4 * i as u64)).collect()
}

#[test]
fn predicted_step_count_is_exact() {
    let (mut m, compiled) = build("Will it rain?", 5);
    let result = run_to_terminal(&mut m, 2_000_000).expect("terminates");
    assert_eq!(result.outcome, StepOutcome::Halted, "must HALT, not trap");
    assert_eq!(
        result.steps, compiled.total_steps,
        "compiler's step prediction must be exact"
    );
    // Token boundaries are monotonically increasing and end at HALT − 1.
    assert!(compiled.token_boundaries.windows(2).all(|w| w[0] < w[1]));
    assert_eq!(
        *compiled.token_boundaries.last().unwrap() + 1,
        compiled.total_steps,
        "last boundary + HALT == total"
    );
}

#[test]
fn decode_is_deterministic_and_well_formed() {
    let lay = Layout::new();
    let (mut m1, _) = build("abc", 6);
    let (mut m2, _) = build("abc", 6);
    run_to_terminal(&mut m1, 2_000_000).unwrap();
    run_to_terminal(&mut m2, 2_000_000).unwrap();
    assert_eq!(m1.state_root(), m2.state_root(), "bit-identical reruns");
    let out1 = read_output(&m1, &lay);
    let out2 = read_output(&m2, &lay);
    assert_eq!(out1, out2);
    assert_eq!(out1.len(), 6, "exactly n_gen tokens");
    assert!(out1.iter().all(|&t| t < 96), "token ids in vocab");
}

#[test]
fn different_prompts_diverge() {
    // Sanity that the model actually reads its input: different prompts of
    // equal length should (with these random weights) produce different
    // states. Not a consensus property — a fixture-quality check.
    let (mut m1, _) = build("aaaa", 4);
    let (mut m2, _) = build("aaab", 4);
    run_to_terminal(&mut m1, 2_000_000).unwrap();
    run_to_terminal(&mut m2, 2_000_000).unwrap();
    assert_ne!(m1.state_root(), m2.state_root());
}
