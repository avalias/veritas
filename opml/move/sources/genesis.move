/// genesis.move — per-question genesis construction (SPEC §7.2).
///
/// A market binds a Fact to (program_root, genesis_root). For the binding to be
/// SOUND, genesis_root must be CONSTRUCTED on-chain from the question's input
/// bytes — never trusted from the caller — otherwise a lying resolver could
/// assert a Fact over a *different* input and still win bisection (dispute.move
/// takes `genesis_root` as a caller-supplied argument).
///
/// This module folds the question's input pages into the judge's audited static
/// image root, proving each target page was the zero page before insertion, and
/// chaining the root forward: static_genesis_root → … → genesis_F (the disputable
/// interval start, "agreed by construction, on-chain"). Twin of the Rust
/// reference (the `genesis_image` memory tree + `MerkleTree::update_leaf_hash`).
module opml::genesis;

use opml::merkle;

const PAGE_SIZE: u64 = 1024;

/// A target page was not the zero page under the running root — the supplied
/// Merkle proof does not place Z_0 at this index (a lying genesis attempt).
const E_OLD_LEAF_NOT_ZERO: u64 = 1;
/// pages / indices / siblings lengths disagree.
const E_BAD_PROOF_SHAPE: u64 = 2;

/// Z_0 — the leaf hash of a zero page: H(0x00 ‖ 0^1024).
public fun zero_page_leaf(): vector<u8> {
    let mut z = vector<u8>[];
    let mut i = 0;
    while (i < PAGE_SIZE) { z.push_back(0u8); i = i + 1; };
    merkle::page_leaf(&z)
}

/// Construct genesis_F from the audited `static_genesis_root` by inserting the
/// question's input pages in ASCENDING index order (SPEC §7.2). For each page,
/// the supplied sibling proof must show the page was zero (old leaf == Z_0)
/// under the *running* root; then the page is folded in and the root advances.
/// Returns genesis_F. The caller (a market) compares it against the Fact's
/// `genesis_root` so a Fact can only resolve an item if it ran the judge on
/// THIS item's input.
public fun genesis_for_item(
    static_genesis_root: vector<u8>,
    pages: &vector<vector<u8>>,
    indices: &vector<u64>,
    siblings: &vector<vector<vector<u8>>>,
): vector<u8> {
    let n = pages.length();
    assert!(indices.length() == n && siblings.length() == n, E_BAD_PROOF_SHAPE);
    let z0 = zero_page_leaf();
    let mut cur = static_genesis_root;
    let mut i = 0;
    while (i < n) {
        let idx = *indices.borrow(i);
        let sibs = siblings.borrow(i);
        // the target page must CURRENTLY be the zero page (no overwrite of real data)
        assert!(merkle::fold(copy z0, idx, sibs) == cur, E_OLD_LEAF_NOT_ZERO);
        // fold the real page in; the root advances to include it
        cur = merkle::fold(merkle::page_leaf(pages.borrow(i)), idx, sibs);
        i = i + 1;
    };
    cur
}
