//! Commitment hashing (SPEC §2.2): SHA3-256 with 1-byte domain tags.
//!
//! SHA3-256 is the commitment hash because it is native in Sui Move's stdlib
//! (`std::hash::sha3_256`) — the on-chain verifier must recompute these
//! exact hashes. blake3 (artifact identity) never appears here; it is never
//! recomputed on-chain.

use sha3::{Digest, Sha3_256};

pub type Hash = [u8; 32];

/// Domain tags (SPEC §2.2). A tagged preimage can never be reinterpreted as
/// a different node kind (second-preimage guard).
pub const TAG_PAGE_LEAF: u8 = 0x00;
pub const TAG_NODE: u8 = 0x01;
pub const TAG_STATE: u8 = 0x02;
pub const TAG_PROG_LEAF: u8 = 0x03;
pub const TAG_TRACE_LEAF: u8 = 0x04;
pub const TAG_SCHED_LEAF: u8 = 0x05;
pub const TAG_JUDGE: u8 = 0x06;

/// H(tag ‖ parts[0] ‖ parts[1] ‖ …)
pub fn tagged(tag: u8, parts: &[&[u8]]) -> Hash {
    let mut h = Sha3_256::new();
    h.update([tag]);
    for p in parts {
        h.update(p);
    }
    h.finalize().into()
}

/// Memory page leaf: H(0x00 ‖ page). Preimage is exactly 1025 bytes.
pub fn page_leaf_hash(page: &[u8]) -> Hash {
    debug_assert_eq!(page.len(), crate::PAGE_SIZE);
    tagged(TAG_PAGE_LEAF, &[page])
}

/// Interior node for ALL trees: H(0x01 ‖ left ‖ right). Preimage 65 bytes.
pub fn node_hash(left: &Hash, right: &Hash) -> Hash {
    tagged(TAG_NODE, &[left, right])
}

/// Program leaf: H(0x03 ‖ instr_96). Preimage 97 bytes.
pub fn prog_leaf_hash(instr: &[u8; crate::isa::INSTR_ENC_LEN]) -> Hash {
    tagged(TAG_PROG_LEAF, &[instr])
}

/// State root: H(0x02 ‖ mem_root ‖ regs_45). Preimage 78 bytes (SPEC §3.3).
pub fn state_root(mem_root: &Hash, regs: &[u8; crate::state::REG_ENC_LEN]) -> Hash {
    tagged(TAG_STATE, &[mem_root, regs])
}
