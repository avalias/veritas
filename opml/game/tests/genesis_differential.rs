//! Differential: the on-chain genesis construction (opml::genesis::genesis_for_item)
//! reproduces the REAL genesis_image memory root, and pins the convention that a
//! Fact's `genesis_root` is the STATE root `H(0x02 ‖ mem_root ‖ regs=0)`, not the
//! bare memory root. Weights-free (toy judge, MEM_DEPTH=10 ⇒ 1 MiB), CI-runnable.
//!
//! This is the Rust mirror of opml::genesis (Move) over a real layout. The Move
//! `merkle::fold`/`page_leaf` are already held byte-identical to the Rust
//! `fold_proof`/`page_leaf_hash` by the gen_move_vectors equivalence suite, so a
//! green differential here plus the green Move synthetic mechanism test
//! (opml::genesis_tests) closes the loop end-to-end on a real genesis image.

use toy_model::layout::{genesis_image, Layout, MEM_DEPTH};
use toy_model::model::{tokenize, ToyModel, WEIGHT_SEED};
use vm::exec::Machine;
use vm::hash::{page_leaf_hash, state_root, Hash};
use vm::merkle::{fold_proof, MerkleTree};
use vm::state::REG_ENC_LEN;
use vm::PAGE_SIZE;

fn page_leaves(image: &[u8]) -> Vec<Hash> {
    image.chunks_exact(PAGE_SIZE).map(page_leaf_hash).collect()
}

#[test]
fn genesis_for_item_matches_real_image_and_state_root_convention() {
    let lay = Layout::new();
    let model = ToyModel::generate(WEIGHT_SEED);
    let toks = tokenize("did the rocket reach orbit");
    assert!(!toks.is_empty());

    let full = genesis_image(&lay, &model, &toks);
    let z0 = page_leaf_hash(&[0u8; PAGE_SIZE]);
    let zero_regs = [0u8; REG_ENC_LEN]; // genesis registers are all zero

    // Reference: the full-image memory root, and the genesis STATE root exactly
    // as a Fact carries it (Machine::state_root over the genesis image).
    let mem_root_full = MerkleTree::from_leaf_hashes(MEM_DEPTH, page_leaves(&full), z0).root();
    let fact_genesis_root = Machine::with_image(MEM_DEPTH, 0, vec![], &full).state_root();
    assert_eq!(
        state_root(&mem_root_full, &zero_regs),
        fact_genesis_root,
        "Fact.genesis_root must be state_root(mem_root, regs=0)"
    );

    // Static image = the genesis image with the INPUT region zeroed (toy input
    // region is a single page at lay.input).
    let input_page = (lay.input / PAGE_SIZE as u64) as usize;
    let mut static_img = full.clone();
    for b in &mut static_img[input_page * PAGE_SIZE..(input_page + 1) * PAGE_SIZE] {
        *b = 0;
    }
    let mut tree = MerkleTree::from_leaf_hashes(MEM_DEPTH, page_leaves(&static_img), z0);
    let static_mem_root = tree.root();

    // The builder's per-page proof (siblings under the static tree).
    let idx = input_page as u64;
    let siblings = tree.prove(idx);
    let new_page: [u8; PAGE_SIZE] = full[input_page * PAGE_SIZE..(input_page + 1) * PAGE_SIZE]
        .try_into()
        .unwrap();

    // The on-chain genesis_for_item logic, in Rust:
    //  (a) the target page was Z_0 under the running (static) root,
    assert_eq!(fold_proof(z0, idx, &siblings), static_mem_root, "old leaf must be Z_0");
    //  (b) folding the real page in yields the genesis memory root.
    let genesis_mem_root = fold_proof(page_leaf_hash(&new_page), idx, &siblings);
    tree.update_leaf_hash(idx, page_leaf_hash(&new_page));
    assert_eq!(genesis_mem_root, tree.root());
    assert_eq!(
        genesis_mem_root, mem_root_full,
        "the construction must reproduce the real genesis_image memory root"
    );

    // And the value a market must compare a Fact's genesis_root against:
    assert_eq!(
        state_root(&genesis_mem_root, &zero_regs),
        fact_genesis_root,
        "genesis_for_item(mem) wrapped in state_root(_, regs=0) == the Fact's genesis_root"
    );
}
