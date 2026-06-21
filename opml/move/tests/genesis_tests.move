/// Tests for opml::genesis (SPEC §7.2 per-question genesis construction).
///
/// Synthetic depth-2 memory tree (4 pages): pages 0,1 are "static" (e.g. weights),
/// pages 2,3 are the INPUT region — zero in the static image, filled at genesis.
/// The expected roots are built directly from the same merkle primitives, so this
/// pins the construction MECHANISM with no model weights (CI-runnable). The
/// byte-exact match against the real `genesis_image` is a separate weights-gated
/// integration check.
#[test_only]
module opml::genesis_tests;

use opml::genesis;
use opml::merkle;

const PAGE_SIZE: u64 = 1024;

/// A 1024-byte page filled with `tag` (tag 0 == the zero page).
fun page(tag: u8): vector<u8> {
    let mut p = vector<u8>[];
    let mut i = 0;
    while (i < PAGE_SIZE) { p.push_back(tag); i = i + 1; };
    p
}

#[test]
fun genesis_construction_depth2_two_input_pages() {
    let p0 = page(11); // static page 0
    let p1 = page(22); // static page 1
    let g2 = page(33); // input page 2 genesis content
    let g3 = page(44); // input page 3 genesis content

    let l0 = merkle::page_leaf(&p0);
    let l1 = merkle::page_leaf(&p1);
    let z0 = genesis::zero_page_leaf();
    let left = merkle::node(&l0, &l1); // shared left subtree (static)

    // static image root: leaves [l0, l1, Z0, Z0]
    let static_root = merkle::node(&left, &merkle::node(&z0, &z0));

    // expected genesis_F: leaves [l0, l1, leaf(g2), leaf(g3)]
    let l2 = merkle::page_leaf(&g2);
    let l3 = merkle::page_leaf(&g3);
    let genesis_f = merkle::node(&left, &merkle::node(&l2, &l3));

    // siblings (bottom-up), reflecting the tree state when each page is inserted:
    //  - inserting idx 2: level0 sib = idx3 leaf (still Z0), level1 sib = left
    //  - inserting idx 3: level0 sib = idx2 leaf (now l2), level1 sib = left
    let sibs2 = vector[copy z0, copy left];
    let sibs3 = vector[copy l2, copy left];

    let pages = vector[g2, g3];
    let indices = vector[2u64, 3u64];
    let siblings = vector[sibs2, sibs3];

    let got = genesis::genesis_for_item(static_root, &pages, &indices, &siblings);
    assert!(got == genesis_f, 0);
}

#[test]
#[expected_failure(abort_code = 1, location = opml::genesis)]
fun rejects_lying_genesis_over_nonzero_page() {
    // A resolver tries to "insert" over page 0, which is NOT zero (it holds
    // static weights). The old-leaf == Z_0 check must abort (E_OLD_LEAF_NOT_ZERO).
    let p0 = page(11);
    let p1 = page(22);
    let l0 = merkle::page_leaf(&p0);
    let l1 = merkle::page_leaf(&p1);
    let z0 = genesis::zero_page_leaf();
    let left = merkle::node(&l0, &l1);
    let static_root = merkle::node(&left, &merkle::node(&z0, &z0));

    // attempt to overwrite idx 0 (a non-zero static page)
    let evil = page(99);
    let sibs0 = vector[copy l1, merkle::node(&z0, &z0)]; // level0 sib = l1, level1 sib = right subtree
    let pages = vector[evil];
    let indices = vector[0u64];
    let siblings = vector[sibs0];

    // fold(Z0, 0, sibs0) != static_root (idx0 is l0, not Z0) → abort 1
    let _ = genesis::genesis_for_item(static_root, &pages, &indices, &siblings);
}
