//! Full binary Merkle tree over leaf hashes (SPEC §3.4).
//!
//! One implementation serves the memory tree (page leaves), the program tree
//! (instruction leaves), and later the schedule/trace trees — they differ
//! only in their leaf-hash rule and the root they are checked against.
//!
//! Proof verification folds LSB-first: at level `l`, bit `l` of the leaf
//! index says whether the running hash is the left (0) or right (1) child.
//! Position is bound by this fold order; the index is not in the preimage.

use crate::hash::{node_hash, page_leaf_hash, Hash};

/// Dense tree: `levels[0]` = 2^depth leaf hashes, …, `levels[depth]` = [root].
///
/// Memory cost is ~2× the leaf-hash array. Fine through depth ~20 (64 MB of
/// nodes); a sparse zero-subtree representation is a Phase 3 optimization.
#[derive(Clone)]
pub struct MerkleTree {
    depth: u8,
    levels: Vec<Vec<Hash>>,
}

impl MerkleTree {
    /// Build from leaf hashes, padding to 2^depth with `pad` (e.g. the
    /// zero-page leaf hash for memory, the zero-instruction leaf hash for
    /// programs — SPEC §3.5).
    pub fn from_leaf_hashes(depth: u8, mut leaves: Vec<Hash>, pad: Hash) -> Self {
        let n = 1usize << depth;
        assert!(leaves.len() <= n, "too many leaves for depth {depth}");
        leaves.resize(n, pad);
        let mut levels = Vec::with_capacity(depth as usize + 1);
        levels.push(leaves);
        for l in 0..depth as usize {
            let below = &levels[l];
            let mut above = Vec::with_capacity(below.len() / 2);
            for pair in below.chunks_exact(2) {
                above.push(node_hash(&pair[0], &pair[1]));
            }
            levels.push(above);
        }
        Self { depth, levels }
    }

    /// All-`pad` tree in O(2^depth) memcpy but O(depth) hashing, using the
    /// zero-subtree chain Z_{l+1} = H(0x01 ‖ Z_l ‖ Z_l) (SPEC §3.4).
    pub fn uniform(depth: u8, pad: Hash) -> Self {
        let mut z = pad;
        let mut levels = Vec::with_capacity(depth as usize + 1);
        for l in 0..=depth as usize {
            levels.push(vec![z; 1usize << (depth as usize - l)]);
            z = node_hash(&z, &z);
        }
        Self { depth, levels }
    }

    pub fn depth(&self) -> u8 {
        self.depth
    }

    pub fn root(&self) -> Hash {
        self.levels[self.depth as usize][0]
    }

    pub fn leaf_hash(&self, index: u64) -> Hash {
        self.levels[0][index as usize]
    }

    /// Sibling hashes bottom-up (depth entries) — the opening for `index`.
    pub fn prove(&self, index: u64) -> Vec<Hash> {
        let mut sibs = Vec::with_capacity(self.depth as usize);
        let mut i = index as usize;
        for l in 0..self.depth as usize {
            sibs.push(self.levels[l][i ^ 1]);
            i >>= 1;
        }
        sibs
    }

    /// O(depth) incremental update after one leaf changes (SPEC §3.4).
    pub fn update_leaf_hash(&mut self, index: u64, leaf: Hash) {
        let mut i = index as usize;
        self.levels[0][i] = leaf;
        for l in 0..self.depth as usize {
            let parent = node_hash(&self.levels[l][i & !1], &self.levels[l][i | 1]);
            i >>= 1;
            self.levels[l + 1][i] = parent;
        }
    }

    /// Batched leaf update: shared ancestors are re-hashed ONCE per level
    /// instead of once per touched leaf — the checkpoint-flush fast path
    /// (k dirty pages cost ~k + k/2 + … node hashes, not k·depth).
    /// `BTreeSet` keeps iteration order deterministic (Invariant 2).
    pub fn update_leaf_hashes_bulk(&mut self, updates: &[(u64, Hash)]) {
        use std::collections::BTreeSet;
        let mut frontier: BTreeSet<usize> = BTreeSet::new();
        for (index, leaf) in updates {
            self.levels[0][*index as usize] = *leaf;
            frontier.insert((*index as usize) >> 1);
        }
        for l in 0..self.depth as usize {
            let mut next = BTreeSet::new();
            for &i in &frontier {
                self.levels[l + 1][i] =
                    node_hash(&self.levels[l][2 * i], &self.levels[l][2 * i + 1]);
                next.insert(i >> 1);
            }
            frontier = next;
        }
    }
}

/// Recompute the root implied by (`leaf`, `index`, `siblings`) — the same
/// fold the on-chain verifier performs, both to check inclusion and to
/// compute a post-write root from the modified leaf (SPEC §8.4 V6–V7).
pub fn fold_proof(leaf: Hash, index: u64, siblings: &[Hash]) -> Hash {
    let mut cur = leaf;
    for (l, sib) in siblings.iter().enumerate() {
        cur = if (index >> l) & 1 == 0 {
            node_hash(&cur, sib)
        } else {
            node_hash(sib, &cur)
        };
    }
    cur
}

pub fn verify_inclusion(root: &Hash, leaf: Hash, index: u64, siblings: &[Hash]) -> bool {
    fold_proof(leaf, index, siblings) == *root
}

/// Zero-page subtree hashes Z_0..Z_max (SPEC §3.4): Z_0 = H(0x00 ‖ 0^1024).
pub fn zero_page_subtrees(max_level: u8) -> Vec<Hash> {
    let mut out = Vec::with_capacity(max_level as usize + 1);
    let mut z = page_leaf_hash(&[0u8; crate::PAGE_SIZE]);
    for _ in 0..=max_level {
        out.push(z);
        z = node_hash(&z, &z);
    }
    out
}
