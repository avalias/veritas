//! Native checkpoint-mode runtime (SPEC §9.1): the toy forward pass as
//! ordinary fast integer Rust, producing BIT-IDENTICAL committed states to
//! the per-step VM at every checkpoint — without interpreting micro-ops or
//! hashing per step.
//!
//! Why this is sound: every arithmetic step routes through the SAME
//! normative helpers as the VM (`vm::exec::{rnd, sat8, sat16, trunc_div}`),
//! integer addition is associative (so dot-product evaluation order is
//! free), and writes land at the same `Layout` addresses the compiler
//! bakes into the program. Conformance C-14 (game/tests/fastrun.rs) holds
//! every checkpoint root to exact equality with the per-step oracle.
//!
//! This file is the existence proof for the "zero math overhead for
//! quantized models" claim: the only honest-path cost left is hashing.

use crate::layout::{Layout, MEM_DEPTH};
use crate::params::*;
use crate::scales::*;
use vm::exec::{rnd, sat16, sat8, trunc_div};
use vm::hash::{page_leaf_hash, state_root, Hash};
use vm::merkle::MerkleTree;
use vm::state::Registers;
use vm::PAGE_SIZE;

/// Flat memory with dirty-page tracking — the checkpoint-mode state.
pub struct FlatMem {
    pub bytes: Vec<u8>,
    dirty: Vec<bool>,
}

impl FlatMem {
    pub fn new(image: Vec<u8>) -> Self {
        let pages = image.len() / PAGE_SIZE;
        Self { bytes: image, dirty: vec![false; pages] }
    }

    pub fn mark(&mut self, addr: u64) {
        self.dirty[(addr as usize) / PAGE_SIZE] = true;
    }

    pub fn w8(&mut self, addr: u64, v: u8) {
        self.bytes[addr as usize] = v;
        self.mark(addr);
    }

    pub fn w16(&mut self, addr: u64, v: u16) {
        self.bytes[addr as usize..addr as usize + 2].copy_from_slice(&v.to_le_bytes());
        self.mark(addr);
    }

    pub fn w32(&mut self, addr: u64, v: u32) {
        self.bytes[addr as usize..addr as usize + 4].copy_from_slice(&v.to_le_bytes());
        self.mark(addr);
    }

    pub fn r8i(&self, addr: u64) -> i64 {
        self.bytes[addr as usize] as i8 as i64
    }

    pub fn r16i(&self, addr: u64) -> i64 {
        i16::from_le_bytes(self.bytes[addr as usize..addr as usize + 2].try_into().unwrap()) as i64
    }

    pub fn r32(&self, addr: u64) -> u32 {
        u32::from_le_bytes(self.bytes[addr as usize..addr as usize + 4].try_into().unwrap())
    }

    pub fn r32i(&self, addr: u64) -> i64 {
        self.r32(addr) as i32 as i64
    }

    pub fn slice(&self, addr: u64, len: usize) -> &[u8] {
        &self.bytes[addr as usize..addr as usize + len]
    }

    /// Drain the dirty set (ascending page order).
    pub fn take_dirty(&mut self) -> Vec<u64> {
        let mut out = Vec::new();
        for (i, d) in self.dirty.iter_mut().enumerate() {
            if *d {
                out.push(i as u64);
                *d = false;
            }
        }
        out
    }
}

/// i8 dot product, any evaluation order (associative): i32 accumulation is
/// exact here (≤ 64·127·127 < 2^20) and vectorizes well.
pub fn dot8(a: &[u8], b: &[u8]) -> i64 {
    let mut acc = 0i32;
    for (x, y) in a.iter().zip(b) {
        acc += (*x as i8 as i32) * (*y as i8 as i32);
    }
    acc as i64
}

/// i16 dot product over LE byte slices (products fit i32, sum needs i64).
pub fn dot16(a: &[u8], b: &[u8]) -> i64 {
    let mut acc = 0i64;
    for (x, y) in a.chunks_exact(2).zip(b.chunks_exact(2)) {
        let xv = i16::from_le_bytes([x[0], x[1]]) as i32;
        let yv = i16::from_le_bytes([y[0], y[1]]) as i32;
        acc += (xv * yv) as i64;
    }
    acc
}

pub struct NativeOutcome {
    /// State root at each token boundary (one per position).
    pub boundary_roots: Vec<Hash>,
    /// Root of the final (halted) state.
    pub final_root: Hash,
    /// Output region bytes `[n][ids…]`.
    pub output: Vec<u8>,
    pub tokens: Vec<u32>,
    /// Dirty pages hashed per checkpoint (bench statistic).
    pub dirty_per_ckpt: Vec<usize>,
}

/// Run the forward pass natively, committing checkpoint roots.
/// `boundaries[p] = (step, pc)` from the compiler (its static schedule).
pub fn run_committed(
    lay: &Layout,
    image: Vec<u8>,
    n_prompt: usize,
    n_gen: usize,
    boundaries: &[(u64, u32)],
) -> NativeOutcome {
    // Genesis tree: hash every page once (the static cost the resolver
    // pays at judge setup, not per inference).
    let mut mem = FlatMem::new(image);
    let leaves: Vec<Hash> =
        mem.bytes.chunks_exact(PAGE_SIZE).map(page_leaf_hash).collect();
    let mut tree = MerkleTree::from_leaf_hashes(
        MEM_DEPTH,
        leaves,
        page_leaf_hash(&[0u8; PAGE_SIZE]),
    );

    let mut boundary_roots = Vec::with_capacity(boundaries.len());
    let mut dirty_per_ckpt = Vec::with_capacity(boundaries.len());
    let mut tokens = Vec::with_capacity(n_gen);
    let (mut last_acc, mut last_aux): (i64, i64) = (0, 0);
    let mut last_tok: u32 = 0;

    let n_pos = n_prompt + n_gen - 1;
    let mut p = 0usize;
    while p < n_pos {
        let (acc, aux, tok) = position(lay, &mut mem, p, n_prompt, &mut tokens);
        last_acc = acc;
        last_aux = aux;
        last_tok = tok;

        // Checkpoint: flush dirty pages through the tree (bulk — shared
        // ancestors hashed once), commit the root with the register file
        // the VM holds at this exact step.
        let dirty = mem.take_dirty();
        dirty_per_ckpt.push(dirty.len());
        let updates: Vec<(u64, Hash)> = dirty
            .iter()
            .map(|pg| (*pg, page_leaf_hash(mem.slice(*pg * PAGE_SIZE as u64, PAGE_SIZE))))
            .collect();
        tree.update_leaf_hashes_bulk(&updates);
        let (step, pc) = boundaries[p];
        let regs = Registers {
            pc,
            halted: 0,
            step,
            acc: last_acc,
            aux: last_aux,
            idx: [last_tok, 0, 0, 0], // K_TOK persists; loop counters reset
        };
        boundary_roots.push(state_root(&tree.root(), &regs.encode()));
        p += 1;
    }

    // Final HALT state: pc unchanged (the HALT slot == last boundary pc),
    // halted = 1, step + 1; memory untouched by HALT.
    let (last_step, last_pc) = *boundaries.last().unwrap();
    let final_regs = Registers {
        pc: last_pc,
        halted: 1,
        step: last_step + 1,
        acc: last_acc,
        aux: last_aux,
        idx: [last_tok, 0, 0, 0],
    };
    let final_root = state_root(&tree.root(), &final_regs.encode());

    let out_len = 4 + 4 * mem.r32(lay.output) as usize;
    NativeOutcome {
        boundary_roots,
        final_root,
        output: mem.slice(lay.output, out_len).to_vec(),
        tokens,
        dirty_per_ckpt,
    }
}

/// Pure native run (no hashing at all) — the "ordinary inference" baseline
/// for the overhead benchmark.
pub fn run_pure(lay: &Layout, image: Vec<u8>, n_prompt: usize, n_gen: usize) -> Vec<u32> {
    let mut mem = FlatMem::new(image);
    let mut tokens = Vec::with_capacity(n_gen);
    let n_pos = n_prompt + n_gen - 1;
    let mut p = 0usize;
    while p < n_pos {
        position(lay, &mut mem, p, n_prompt, &mut tokens);
        p += 1;
    }
    tokens
}

/// One position of the forward pass — semantics mirror the compiled
/// program op for op (same helpers, same write addresses, same order of
/// requantization). Returns the (acc, aux, tok_id) register values live at
/// the boundary.
fn position(
    lay: &Layout,
    mem: &mut FlatMem,
    p: usize,
    n_prompt: usize,
    tokens: &mut Vec<u32>,
) -> (i64, i64, u32) {
    let du = D as u64;
    // LDIDX: current token id.
    let tok_id = if p < n_prompt {
        mem.r32(lay.input + 4 + 4 * p as u64)
    } else {
        mem.r32(lay.tok)
    };

    // Embedding + position embedding: x[c] = sat8(emb + pos).
    let mut acc_live = 0i64;
    for c in 0..du {
        let e = mem.r8i(lay.emb + tok_id as u64 * du + c);
        let po = mem.r8i(lay.pos + p as u64 * du + c);
        acc_live = e.wrapping_add(po);
        mem.w8(lay.x + c, sat8(acc_live) as u8);
    }
    let mut aux_live = 0i64; // tracked through every ARGMAX scan

    for l in 0..LAYERS {
        rmsnorm(lay, mem, lay.g1[l], &mut acc_live);
        // QKV per head; K/V append at cache position p.
        for h in 0..HEADS {
            let row0 = (h * D_HEAD) as u64 * du;
            for r in 0..D_HEAD as u64 {
                let xn = lay.xn;
                let q = rnd(dot_row(mem, lay.wq[l] + row0 + r * du, xn), S_QK_I16);
                mem.w16(lay.q + (h * D_HEAD * 2) as u64 + 2 * r, sat16(q) as u16);
                let k = rnd(dot_row(mem, lay.wk[l] + row0 + r * du, xn), S_QK_I16);
                mem.w16(lay.kc[l][h] + (p * D_HEAD * 2) as u64 + 2 * r, sat16(k) as u16);
                let v = rnd(dot_row(mem, lay.wv[l] + row0 + r * du, xn), S_PROJ_I8);
                acc_live = v; // CLAMP8 leaves acc at the pre-clamp value
                mem.w8(lay.vt[l][h] + r * MAX_SEQ as u64 + p as u64, sat8(v) as u8);
            }
        }
        for h in 0..HEADS {
            let q_base = lay.q + (h * D_HEAD * 2) as u64;
            // Attention logits over positions 0..=p.
            for j in 0..=p as u64 {
                let logit = rnd(
                    dot16(
                        mem.slice(q_base, D_HEAD * 2),
                        mem.slice(lay.kc[l][h] + j * (D_HEAD * 2) as u64, D_HEAD * 2),
                    ),
                    S_LOGIT_Q411,
                );
                mem.w32(lay.att32 + 4 * j, logit as u32);
            }
            // Max scan (ARGMAX_STEP semantics: strictly greater, aux ← j).
            let mut mx = i32::MIN as i64;
            for j in 0..=p as u64 {
                let v = mem.r32i(lay.att32 + 4 * j);
                if v > mx {
                    mx = v;
                    aux_live = j as i64;
                }
            }
            let neg = mx.wrapping_mul(-1);
            mem.w32(lay.neg_max, neg as u32);
            // exp(logit − max) via the committed LUT.
            for j in 0..=p as u64 {
                let xq = mem.r32i(lay.att32 + 4 * j).wrapping_add(mem.r32i(lay.neg_max));
                let e = lut(mem, lay.lut_exp, xq);
                mem.w32(lay.e32 + 4 * j, e as u32);
            }
            let mut sum = 0i64;
            for j in 0..=p as u64 {
                sum = sum.wrapping_add(mem.r32i(lay.e32 + 4 * j));
            }
            mem.w32(lay.sum, sum as u32);
            // probs = (e·2^14)/sum >> 7, saturated to Q0.7.
            for j in 0..=p as u64 {
                let pq = rnd(
                    trunc_div(mem.r32i(lay.e32 + 4 * j).wrapping_mul(16384), mem.r32i(lay.sum)),
                    S_PROB_Q07,
                );
                mem.w8(lay.probs + j, sat8(pq) as u8);
            }
            // ctx[r] = Σ probs·Vᵀ over the full 64-lane line (stale lanes
            // beyond p are zero — same memory model as the VM).
            for r in 0..D_HEAD as u64 {
                let ctx = rnd(
                    dot8(
                        mem.slice(lay.probs, MAX_SEQ),
                        mem.slice(lay.vt[l][h] + r * MAX_SEQ as u64, MAX_SEQ),
                    ),
                    S_CTX_I8,
                );
                acc_live = ctx;
                mem.w8(lay.attnx + (h * D_HEAD) as u64 + r, sat8(ctx) as u8);
            }
        }
        // O-projection + residual.
        for c in 0..du {
            let o = rnd(dot_row(mem, lay.wo[l] + c * du, lay.attnx), S_PROJ_I8);
            acc_live = o.wrapping_add(mem.r8i(lay.x + c));
            mem.w8(lay.x + c, sat8(acc_live) as u8);
        }
        // FFN.
        rmsnorm(lay, mem, lay.g2[l], &mut acc_live);
        for r in 0..FFN as u64 {
            let up = rnd(dot_row(mem, lay.w1[l] + r * du, lay.xn), S_FFN_Q411);
            let sl = rnd(lut(mem, lay.lut_silu, up), S_SILU_I8);
            acc_live = sl;
            mem.w8(lay.h_ffn + r, sat8(sl) as u8);
        }
        for c in 0..du {
            let mut s = 0i64;
            for j in 0..(FFN / 64) as u64 {
                s = s.wrapping_add(dot8(
                    mem.slice(lay.w2[l] + c * FFN as u64 + 64 * j, 64),
                    mem.slice(lay.h_ffn + 64 * j, 64),
                ));
            }
            let dn = rnd(s, S_FFN_DOWN_I8);
            acc_live = dn.wrapping_add(mem.r8i(lay.x + c));
            mem.w8(lay.x + c, sat8(acc_live) as u8);
        }
    }

    // Decode head at decision positions.
    if p >= n_prompt - 1 {
        rmsnorm(lay, mem, lay.gf, &mut acc_live);
        for v in 0..VOCAB as u64 {
            let lg = dot_row(mem, lay.head + v * du, lay.xn);
            mem.w32(lay.logits + 4 * v, lg as u32);
        }
        let mut mx = i32::MIN as i64;
        for v in 0..VOCAB as u64 {
            let lv = mem.r32i(lay.logits + 4 * v);
            if lv > mx {
                mx = lv;
                aux_live = v as i64;
            }
        }
        let g = (p - (n_prompt - 1)) as u64;
        let token = aux_live as u32;
        mem.w32(lay.tok, token);
        mem.w32(lay.output + 4 + 4 * g, token);
        acc_live = (g + 1) as i64; // LDC(g+1) before the count store
        mem.w32(lay.output, (g + 1) as u32);
        tokens.push(token);
    }
    (acc_live, aux_live, tok_id)
}

/// One 64-lane weight-row dot against an activation line.
fn dot_row(mem: &FlatMem, w_row: u64, xn: u64) -> i64 {
    dot8(mem.slice(w_row, D), mem.slice(xn, D))
}

/// LUT16 semantics: index = sat16(x) + 32768, entry sign-extended.
fn lut(mem: &FlatMem, table: u64, x: i64) -> i64 {
    mem.r16i(table + 2 * (sat16(x) as i64 + 32768) as u64)
}

/// RMSNorm recipe (SPEC §6.4): x → xn with the given gains.
fn rmsnorm(lay: &Layout, mem: &mut FlatMem, gamma: u64, acc_live: &mut i64) {
    let ss = dot8(mem.slice(lay.x, D), mem.slice(lay.x, D));
    let mean = trunc_div(ss, D as i64);
    let r = lut(mem, lay.lut_rsqrt, mean);
    mem.w32(lay.r32, r as u32);
    for c in 0..D as u64 {
        let v = rnd(
            mem.r8i(lay.x + c).wrapping_mul(mem.r32i(lay.r32)).wrapping_mul(mem.r32i(gamma + 4 * c)),
            S_NORM,
        );
        *acc_live = v;
        mem.w8(lay.xn + c, sat8(v) as u8);
    }
}
