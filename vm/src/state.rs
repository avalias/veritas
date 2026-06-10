//! Machine state: register file and Merkle-committed memory (SPEC §3).

use crate::hash::{page_leaf_hash, Hash};
use crate::merkle::MerkleTree;
use crate::PAGE_SIZE;

/// `halted` register values (SPEC §3.2).
pub const RUNNING: u8 = 0;
pub const HALTED: u8 = 1;
pub const TRAPPED: u8 = 2;

/// Canonical register encoding length (SPEC §3.2): 4+1+8+8+8+16.
pub const REG_ENC_LEN: usize = 45;

/// Register file (SPEC §3.2). Signed registers are i64; their canonical
/// encoding is the LE bytes of the two's-complement bit pattern (§2.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Registers {
    pub pc: u32,
    pub halted: u8,
    pub step: u64,
    pub acc: i64,
    pub aux: i64,
    pub idx: [u32; 4],
}

impl Registers {
    /// Canonical 45-byte encoding, field order and offsets per SPEC §3.2.
    pub fn encode(&self) -> [u8; REG_ENC_LEN] {
        let mut b = [0u8; REG_ENC_LEN];
        b[0..4].copy_from_slice(&self.pc.to_le_bytes());
        b[4] = self.halted;
        b[5..13].copy_from_slice(&self.step.to_le_bytes());
        b[13..21].copy_from_slice(&self.acc.to_le_bytes());
        b[21..29].copy_from_slice(&self.aux.to_le_bytes());
        for (j, r) in self.idx.iter().enumerate() {
            b[29 + 4 * j..33 + 4 * j].copy_from_slice(&r.to_le_bytes());
        }
        b
    }

    pub fn decode(b: &[u8; REG_ENC_LEN]) -> Self {
        let u32le = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let i64le = |o: usize| i64::from_le_bytes(b[o..o + 8].try_into().unwrap());
        Self {
            pc: u32le(0),
            halted: b[4],
            step: u64::from_le_bytes(b[5..13].try_into().unwrap()),
            acc: i64le(13),
            aux: i64le(21),
            idx: [u32le(29), u32le(33), u32le(37), u32le(41)],
        }
    }
}

/// Byte-addressable memory with an incrementally-maintained page Merkle tree.
///
/// Every write keeps the tree exact (O(log n) re-hash, SPEC §3.4) — this is
/// the per-step-capable representation. Checkpoint mode (dirty-page hashing)
/// arrives with the trace machinery in Phase 1+.
pub struct CommittedMemory {
    d: u8,
    pages: Vec<[u8; PAGE_SIZE]>,
    tree: MerkleTree,
}

impl CommittedMemory {
    pub fn new_zero(d: u8) -> Self {
        assert!(
            (1..=crate::MAX_MEM_DEPTH).contains(&d),
            "memory depth {d} out of range"
        );
        let zero_leaf = page_leaf_hash(&[0u8; PAGE_SIZE]);
        Self {
            d,
            pages: vec![[0u8; PAGE_SIZE]; 1usize << d],
            tree: MerkleTree::uniform(d, zero_leaf),
        }
    }

    pub fn depth(&self) -> u8 {
        self.d
    }

    /// Total memory size in bytes: 2^d pages × 1024.
    pub fn mem_bytes(&self) -> u64 {
        (1u64 << self.d) * PAGE_SIZE as u64
    }

    pub fn root(&self) -> Hash {
        self.tree.root()
    }

    pub fn page(&self, index: u64) -> &[u8; PAGE_SIZE] {
        &self.pages[index as usize]
    }

    pub fn prove_page(&self, index: u64) -> Vec<Hash> {
        self.tree.prove(index)
    }

    /// Bulk page install (genesis building). Re-hashes the page path.
    pub fn set_page(&mut self, index: u64, page: [u8; PAGE_SIZE]) {
        self.pages[index as usize] = page;
        self.tree.update_leaf_hash(index, page_leaf_hash(&page));
    }

    /// Read `len` bytes at `addr`. Callers must have validated bounds and
    /// alignment (SPEC §4.4 T3/T7) — alignment guarantees no page straddle.
    pub fn read(&self, addr: u64, len: usize) -> &[u8] {
        let (p, off) = (addr / PAGE_SIZE as u64, (addr % PAGE_SIZE as u64) as usize);
        debug_assert!(off + len <= PAGE_SIZE, "page-straddling read");
        &self.pages[p as usize][off..off + len]
    }

    pub fn read_u8(&self, addr: u64) -> u8 {
        self.read(addr, 1)[0]
    }

    pub fn read_u16(&self, addr: u64) -> u16 {
        u16::from_le_bytes(self.read(addr, 2).try_into().unwrap())
    }

    pub fn read_u32(&self, addr: u64) -> u32 {
        u32::from_le_bytes(self.read(addr, 4).try_into().unwrap())
    }

    /// Write `bytes` at `addr` (single page, validated by caller) and
    /// incrementally update the tree.
    pub fn write(&mut self, addr: u64, bytes: &[u8]) {
        let (p, off) = (addr / PAGE_SIZE as u64, (addr % PAGE_SIZE as u64) as usize);
        debug_assert!(off + bytes.len() <= PAGE_SIZE, "page-straddling write");
        self.pages[p as usize][off..off + bytes.len()].copy_from_slice(bytes);
        self.tree
            .update_leaf_hash(p, page_leaf_hash(&self.pages[p as usize]));
    }

    /// Test oracle: rebuild the tree from scratch and return its root.
    /// Conformance test C-4 asserts this always equals `root()`.
    pub fn recompute_root_full(&self) -> Hash {
        let leaves: Vec<Hash> = self.pages.iter().map(|p| page_leaf_hash(p)).collect();
        MerkleTree::from_leaf_hashes(self.d, leaves, page_leaf_hash(&[0u8; PAGE_SIZE])).root()
    }
}
