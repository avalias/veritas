//! Honest-path overhead benchmark (run with --release):
//!
//!   cargo run -p benches --release
//!
//! Measures, on the toy judge (integer path — the quantized-model case):
//!   A. pure native forward (no commitments)        — "ordinary inference"
//!   B. native checkpoint mode (A + dirty-page hashing at token boundaries)
//!   C. genesis tree build (per-judge setup, amortized across inferences)
//!   D. interpreter with eager per-write hashing    — the old worst case
//!   E. full per-step trace                          — dispute-only path
//! plus SHA3 throughput and dirty-page statistics.
//!
//! The claim under test (SPEC §1.4): for integer models the determinism
//! tax on math is zero (associativity), so B − A ≈ hashing only.
//! All ratios are integer basis points — no floats in this workspace.

use game::trace::trace_full;
use std::time::Instant;
use toy_model::forward;
use vm::hash::page_leaf_hash;
use vm::merkle::MerkleTree;
use vm::trace::run_to_terminal;
use vm::PAGE_SIZE;

fn micros(f: impl FnOnce()) -> u128 {
    let t = Instant::now();
    f();
    t.elapsed().as_micros()
}

/// Best of `n` runs (steadier on a laptop).
fn best_of(n: usize, mut f: impl FnMut() -> u128) -> u128 {
    (0..n).map(|_| f()).min().unwrap()
}

fn main() {
    let prompt = "Will ETH close above 4000 USD on Friday?";
    let n_gen = 20usize;
    let s = game::setup::ToySetup::new(prompt, n_gen);
    let n_pos = s.compiled.n_prompt + n_gen - 1;
    let boundaries: Vec<(u64, u32)> = s
        .compiled
        .token_boundaries
        .iter()
        .copied()
        .zip(s.compiled.boundary_pcs.iter().copied())
        .collect();

    println!("== toy judge, integer path ==");
    println!(
        "prompt {} chars, {} generated tokens, {} positions, {} micro-ops, {} instructions",
        prompt.len(),
        n_gen,
        n_pos,
        s.compiled.total_steps,
        s.compiled.program.len()
    );

    // A. pure native — the inference baseline.
    let t_pure = best_of(3, || {
        let img = s.image.clone();
        micros(|| {
            let toks = forward::run_pure(&s.lay, img, s.compiled.n_prompt, n_gen);
            assert_eq!(toks.len(), n_gen);
        })
    });

    // B. native checkpoint mode (includes genesis tree build — separated
    // out via C below).
    let mut dirty_stats = Vec::new();
    let t_commit_total = best_of(3, || {
        let img = s.image.clone();
        micros(|| {
            let out = forward::run_committed(
                &s.lay,
                img,
                s.compiled.n_prompt,
                n_gen,
                &boundaries,
            );
            dirty_stats = out.dirty_per_ckpt.clone();
        })
    });

    // C. genesis tree build alone (per-judge setup cost).
    let t_genesis = best_of(3, || {
        let img = s.image.clone();
        micros(|| {
            let leaves: Vec<_> = img.chunks_exact(PAGE_SIZE).map(page_leaf_hash).collect();
            let tree = MerkleTree::from_leaf_hashes(
                toy_model::layout::MEM_DEPTH,
                leaves,
                page_leaf_hash(&[0u8; PAGE_SIZE]),
            );
            std::hint::black_box(tree.root());
        })
    });

    // D. interpreter with eager per-write incremental hashing.
    let t_machine_build = best_of(2, || {
        micros(|| {
            std::hint::black_box(s.machine().state_root());
        })
    });
    let t_interp = best_of(2, || {
        let mut m = s.machine();
        micros(|| {
            run_to_terminal(&mut m, 100_000_000).unwrap();
        })
    });

    // E. full per-step trace (dispute-segment materialization shape).
    let t_trace = {
        let mut m = s.machine();
        micros(|| {
            trace_full(&mut m, None, s.lay.output, 100_000_000);
        })
    };

    // SHA3 throughput (single thread, software).
    let mb = 32usize;
    let page = vec![0xABu8; PAGE_SIZE];
    let t_sha = micros(|| {
        for _ in 0..(mb * 1024) {
            std::hint::black_box(page_leaf_hash(&page));
        }
    });
    let sha_mbps = (mb as u128 * 1_000_000) / t_sha.max(1);

    let t_commit = t_commit_total.saturating_sub(t_genesis);
    let dirty_total: usize = dirty_stats.iter().sum();
    let dirty_max = dirty_stats.iter().max().copied().unwrap_or(0);
    // Integer basis points: 10_000 = 100%.
    let bp = |num: u128, den: u128| (num * 10_000) / den.max(1);

    println!("\n| path | time (µs) | vs native |");
    println!("|---|---|---|");
    println!("| A. native, no commitments | {t_pure} | 1.00× |");
    println!(
        "| B. native + checkpoint commitments | {t_commit} | {}.{:02}× |",
        bp(t_commit, t_pure) / 10_000,
        (bp(t_commit, t_pure) % 10_000) / 100
    );
    println!("| C. genesis tree (per-judge, amortized) | {t_genesis} | — |");
    println!(
        "| D. interpreter, eager hashing | {t_interp} | {}.{:02}× |",
        bp(t_interp, t_pure) / 10_000,
        (bp(t_interp, t_pure) % 10_000) / 100
    );
    println!(
        "| E. full per-step trace (disputes only) | {t_trace} | {}.{:02}× |",
        bp(t_trace, t_pure) / 10_000,
        (bp(t_trace, t_pure) % 10_000) / 100
    );
    println!("| machine build (genesis, interpreter side) | {t_machine_build} | — |");

    let overhead_bp = bp(t_commit.saturating_sub(t_pure), t_pure);
    println!(
        "\ncommitment overhead (B − A)/A: {}.{:02}% — the entire honest-path cost",
        overhead_bp / 100,
        overhead_bp % 100
    );
    println!(
        "dirty pages: {} total across {} checkpoints (max {}/ckpt) ≈ {} KiB hashed per inference",
        dirty_total,
        dirty_stats.len(),
        dirty_max,
        dirty_total * PAGE_SIZE / 1024
    );
    println!("sha3-256 software throughput (1 KiB blocks, 1 thread): ~{sha_mbps} MB/s");
    println!(
        "\nnotes: toy compute is tiny, so hashing share here is a WORST case vs \
         real models; page hashing parallelizes freely (independent leaves) and \
         ARMv8.2 SHA3 / batched checkpoints push it further down."
    );
}
