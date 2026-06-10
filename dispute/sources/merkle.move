/// Tagged SHA3-256 hashing and Merkle path folding (SPEC §2.2, §3.4).
/// Twin of vm/src/hash.rs + vm/src/merkle.rs (verification side only).
module dispute::merkle;

use std::hash;

const PAGE_SIZE: u64 = 1024;

const TAG_PAGE_LEAF: u8 = 0x00;
const TAG_NODE: u8 = 0x01;
const TAG_STATE: u8 = 0x02;
const TAG_PROG_LEAF: u8 = 0x03;

fun tagged2(tag: u8, a: &vector<u8>, b: &vector<u8>): vector<u8> {
    let mut buf = vector[tag];
    buf.append(*a);
    buf.append(*b);
    hash::sha3_256(buf)
}

fun tagged1(tag: u8, a: &vector<u8>): vector<u8> {
    let mut buf = vector[tag];
    buf.append(*a);
    hash::sha3_256(buf)
}

/// H(0x00 ‖ page), page exactly 1024 bytes.
public fun page_leaf(page: &vector<u8>): vector<u8> {
    assert!(page.length() == PAGE_SIZE, 0);
    tagged1(TAG_PAGE_LEAF, page)
}

/// H(0x03 ‖ instr_96).
public fun prog_leaf(instr: &vector<u8>): vector<u8> {
    assert!(instr.length() == 96, 0);
    tagged1(TAG_PROG_LEAF, instr)
}

/// H(0x01 ‖ left ‖ right).
public fun node(l: &vector<u8>, r: &vector<u8>): vector<u8> {
    tagged2(TAG_NODE, l, r)
}

/// H(0x02 ‖ mem_root ‖ regs_45) (SPEC §3.3).
public fun state_root(mem_root: &vector<u8>, regs: &vector<u8>): vector<u8> {
    assert!(mem_root.length() == 32 && regs.length() == 45, 0);
    tagged2(TAG_STATE, mem_root, regs)
}

/// Recompute the root implied by (leaf, index, siblings) — LSB-first fold
/// (SPEC §3.4). Same function verifies inclusion and computes post-write
/// roots (SPEC §8.4 V6–V7).
public fun fold(leaf: vector<u8>, index: u64, sibs: &vector<vector<u8>>): vector<u8> {
    let mut cur = leaf;
    let mut l = 0u64;
    let n = sibs.length();
    while (l < n) {
        let sib = &sibs[l];
        cur = if (((index >> (l as u8)) & 1) == 0) {
            node(&cur, sib)
        } else {
            node(sib, &cur)
        };
        l = l + 1;
    };
    cur
}
