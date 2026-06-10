//! Native integer Qwen3 forward — the checkpoint-mode runtime at real
//! scale (SPEC §9.1). Same discipline as the toy: every arithmetic step
//! uses the normative helpers (`vm::exec::{rnd, sat8, sat16, trunc_div}`),
//! every write lands at a committed `QwenLayout` address, dirty pages are
//! hashed only at per-token checkpoints. Row-parallelism is the
//! §9.1-licensed kind: distinct output cells, results written by a single
//! thread.
//!
//! Checkpoint register files are PROVISIONAL (pc/step = position index)
//! until the Qwen VM compiler pins its schedule — the hashing cost, which
//! is what the §1.4 benchmark measures, is identical either way.

use crate::image::Tables;
use crate::layout::{QwenLayout, MAX_SEQ, MEM_DEPTH};
use crate::quant::{IntModel, NormSite, SHIFT};
use std::time::Instant;
use toy_model::forward::{dot16, FlatMem};
use vm::exec::{rnd, sat16, trunc_div};
use vm::hash::{page_leaf_hash, state_root, Hash};
use vm::merkle::MerkleTree;
use vm::state::Registers;
use vm::PAGE_SIZE;

/// i8-weight × i16-activation dot over LE bytes. 64-lane i32 partials
/// (≤ 64·127·32767 < 2^29) accumulate into i64 — exact, vectorizes well.
fn dot_w8_x16(w: &[u8], x: &[u8]) -> i64 {
    let mut acc = 0i64;
    for (wc, xc) in w.chunks(64).zip(x.chunks(128)) {
        let mut part = 0i32;
        for (a, b) in wc.iter().zip(xc.chunks_exact(2)) {
            part += (*a as i8 as i32) * (i16::from_le_bytes([b[0], b[1]]) as i32);
        }
        acc += part as i64;
    }
    acc
}

pub struct Native<'a> {
    pub lay: &'a QwenLayout,
    pub im: &'a IntModel,
    pub tables: &'a Tables,
    pub threads: usize,
    /// Persistent worker pool (kernels crate): NEON GEMV + no per-call
    /// thread spawns. Bit-exactness vs the scalar reference is enforced by
    /// kernels' own equality tests.
    pub pool: kernels::Pool,
}

/// Per-run measurements for the §1.4 report.
#[derive(Default, Debug, Clone)]
pub struct RunStats {
    pub compute_us: u128,
    pub hash_us: u128,
    pub genesis_us: u128,
    pub dirty_pages: usize,
    pub tokens: Vec<u32>,
    pub boundary_roots: Vec<Hash>,
    /// Memory-tree roots alone (no register file) — the Qwen C-14 oracle
    /// comparison is on these until the compiler-pinned registers land in
    /// the runtime.
    pub boundary_mem_roots: Vec<Hash>,
}

impl<'a> Native<'a> {
    fn lut(&self, table: &[i16], x: i64) -> i64 {
        table[(sat16(x) as i64 + 32768) as usize] as i64
    }

    /// Probe helper: exp LUT lookup with the runtime's exact semantics.
    pub fn tables_exp_at(&self, x: i64) -> i64 {
        self.lut(&self.tables.exp, x)
    }

    /// RMSNorm from the i32 residual carrier to i16 matmul inputs. The
    /// mean-square is computed over PER-ELEMENT prescaled values (rnd by
    /// the calibrated site.pre_shift) so it lands in the rsqrt LUT domain;
    /// r ≈ 2^(14+k)/√ms_q and the folded γ accounts for it.
    fn rmsnorm_to_i16(&self, mem: &mut FlatMem, src: u64, dst: u64, site: &NormSite, n: u64) {
        debug_assert!(site.elem_pre);
        // Stage sat16(rnd(x, pre)) in the committed xp array, then square the
        // STAGED values — the exact MAC16 chain the VM program runs. (sat16
        // never binds on calibrated runs: xp <= sqrt(ss) <= 2^13-ish.)
        for c in 0..n {
            let xp = rnd(mem.r32i(src + 4 * c), site.pre_shift);
            mem.w16(self.lay.xp + 2 * c, sat16(xp) as u16);
        }
        let mut ss = 0i64;
        for c in 0..n {
            let xp = mem.r16i(self.lay.xp + 2 * c);
            ss = ss.wrapping_add(xp.wrapping_mul(xp));
        }
        let mean = trunc_div(ss, n as i64);
        let r = self.lut(&self.tables.rsqrt, mean);
        mem.w32(self.lay.r32, r as u32);
        for c in 0..n {
            // x_q(i32)·r(≤2^15)·γm(≤2^28) ≤ 2^31·2^15·2^28 = 2^74 — too
            // big; stage it: t = rnd(x·r, 14) keeps t ≤ 2^32, then ·γm.
            let t = rnd(mem.r32i(src + 4 * c).wrapping_mul(r), 14);
            let v = rnd(t.wrapping_mul(site.gamma_m[c as usize] as i64), site.shift - 14);
            mem.w16(dst + 2 * c, sat16(v) as u16);
        }
    }

    /// QK-norm over one head's i16 vector, in place (pre_shift 16).
    fn qknorm_i16(&self, mem: &mut FlatMem, base: u64, site: &NormSite, dh: u64) {
        let mut ss = 0i64;
        for d in 0..dh {
            let v = mem.r16i(base + 2 * d);
            ss = ss.wrapping_add(v.wrapping_mul(v));
        }
        let mean = rnd(trunc_div(ss, dh as i64), site.pre_shift);
        let r = self.lut(&self.tables.rsqrt, mean);
        mem.w32(self.lay.r32, r as u32);
        for d in 0..dh {
            let v = rnd(
                mem.r16i(base + 2 * d)
                    .wrapping_mul(r)
                    .wrapping_mul(site.gamma_m[d as usize] as i64),
                site.shift,
            );
            mem.w16(base + 2 * d, sat16(v) as u16);
        }
    }

    /// Rotary (rotate-half pairing) on an i16 head vector at `base`.
    fn rope(&self, mem: &mut FlatMem, base: u64, pos: usize, dh: u64) {
        let half = (dh / 2) as usize;
        for p2 in 0..half {
            let c = self.tables.cos[pos * half + p2] as i64;
            let s = self.tables.sin[pos * half + p2] as i64;
            let ns = self.tables.nsin[pos * half + p2] as i64;
            let a = mem.r16i(base + 2 * p2 as u64);
            let b = mem.r16i(base + 2 * (p2 + half) as u64);
            // MAC16+MAC16+SHIFT_RNDN(14)+CLAMP16, exactly (SPEC §6.4).
            let na = rnd(a.wrapping_mul(c).wrapping_add(b.wrapping_mul(ns)), 14);
            let nb = rnd(a.wrapping_mul(s).wrapping_add(b.wrapping_mul(c)), 14);
            mem.w16(self.lay.t_na16, sat16(na) as u16); // VM staging cell
            mem.w16(base + 2 * (p2 + half) as u64, sat16(nb) as u16);
            mem.w16(base + 2 * p2 as u64, sat16(na) as u16);
        }
    }

    /// Row-parallel projection: rows of `w` (rows×cols i8) · x (i16),
    /// requant by per-channel m — NEON + persistent pool.
    #[allow(dead_code)] // non-blocked reference path
    fn proj(&self, mem: &FlatMem, w: u64, m: &[i32], rows: u64, cols: u64, x16: u64) -> Vec<i64> {
        let x = mem.slice(x16, 2 * cols as usize);
        let wbytes = mem.slice(w, (rows * cols) as usize);
        let mut out = vec![0i64; rows as usize];
        kernels::gemv_bytes(&self.pool, wbytes, x, rows as usize, cols as usize, m, SHIFT, &mut out);
        out
    }

    /// Blocked projection: per-block activation scales, per-(row,block)
    /// multipliers, exact i64 accumulation (kernels::gemv_blocked).
    fn proj_b(&self, mem: &FlatMem, w: u64, m: &(Vec<i32>, u8), rows: u64, cols: u64, x16: u64) -> Vec<i64> {
        let x = mem.slice(x16, 2 * cols as usize);
        let wbytes = mem.slice(w, (rows * cols) as usize);
        let mut out = vec![0i64; rows as usize];
        kernels::gemv_blocked_bytes(&self.pool, wbytes, x, rows as usize, cols as usize, &m.0, m.1, &mut out);
        out
    }

    /// EVAL hook: full integer logits (committed semantics: rnd(dot,11))
    /// recomputed from the current xn. Threaded; read-only.
    pub fn head_logits(&self, mem: &FlatMem, out: &mut [i64]) {
        let h = self.im.cfg.hidden_size;
        let vocab = self.im.cfg.vocab_size;
        let x = mem.slice(self.lay.xn, 2 * h);
        let w = mem.slice(self.lay.head_w, vocab * h);
        // Per-(row,block) multipliers restore one common logit scale.
        kernels::gemv_blocked_bytes(&self.pool, w, x, vocab, h, &self.im.m_head.0, self.im.m_head.1, out);
    }

    /// One position. `decide` ⇒ run the LM head and return the argmax token.
    pub fn position(&self, mem: &mut FlatMem, pos: usize, tok: u32, decide: bool) -> Option<u32> {
        self.position_impl(mem, pos, tok, decide, usize::MAX)
    }

    /// Probe hook: run only the first `upto` layers (no head).
    pub fn position_prefix(&self, mem: &mut FlatMem, pos: usize, tok: u32, upto: usize) {
        self.position_impl(mem, pos, tok, false, upto);
    }

    fn position_impl(
        &self,
        mem: &mut FlatMem,
        pos: usize,
        tok: u32,
        decide: bool,
        upto: usize,
    ) -> Option<u32> {
        let cfg = &self.im.cfg;
        let (h, dh, f) = (cfg.hidden_size as u64, cfg.head_dim as u64, cfg.intermediate_size as u64);
        let (nh, nkv) = (cfg.num_attention_heads as u64, cfg.num_key_value_heads as u64);
        let lay = self.lay;


        // Embedding into the i32 residual carrier (global scale, no
        // rescales anywhere — i32 spans Qwen3's full dynamic range).
        for c in 0..h {
            let e = mem.r8i(lay.emb + tok as u64 * h + c);
            // ST32 semantics: low-32 truncation (value bounded ~2^18 anyway).
            mem.w32(lay.x + 4 * c, rnd(e.wrapping_mul(self.im.m_emb[c as usize] as i64), SHIFT) as u32);
        }

        for (l, lw) in self.im.layers.iter().enumerate().take(upto) {
            let a = &lay.layers[l];
            self.rmsnorm_to_i16(mem, lay.x, lay.xn, &lw.norm1, h);
            // QKV projections (row-parallel), stores single-threaded.
            let qv = self.proj_b(mem, a.wq, &lw.mq, nh * dh, h, lay.xn);
            for (r, v) in qv.iter().enumerate() {
                mem.w16(lay.q + 2 * r as u64, sat16(*v) as u16);
            }
            let kv = self.proj_b(mem, a.wk, &lw.mk, nkv * dh, h, lay.xn);
            let krow = a.kc + pos as u64 * nkv * dh * 2;
            for (r, v) in kv.iter().enumerate() {
                mem.w16(krow + 2 * r as u64, sat16(*v) as u16);
            }
            let vv = self.proj_b(mem, a.wv, &lw.mv, nkv * dh, h, lay.xn);
            let vrow = a.vc + pos as u64 * nkv * dh * 2;
            for (r, v) in vv.iter().enumerate() {
                mem.w16(vrow + 2 * r as u64, sat16(*v) as u16);
            }
            // QK-norm + rotary.
            for hd in 0..nh {
                let base = lay.q + hd * dh * 2;
                self.qknorm_i16(mem, base, &lw.qnorm, dh);
                self.rope(mem, base, pos, dh);
            }
            for kvh in 0..nkv {
                let base = krow + kvh * dh * 2;
                self.qknorm_i16(mem, base, &lw.knorm, dh);
                self.rope(mem, base, pos, dh);
            }
            // Attention per q-head (GQA) on the persistent pool; heads
            // write disjoint dh-sized chunks of ctx.
            let mut ctx = vec![0i64; (nh * dh) as usize];
            {
                let memr: &FlatMem = mem;
                kernels::run_disjoint_i64(&self.pool, &mut ctx, dh as usize, &|hd, out| {
                    attn_head(self, memr, l, hd as u64, pos, out)
                });
            }
            // Stores + probs/att32 scratch writes happen single-threaded
            // inside attn_head? No — attn_head is read-only; replay its
            // scratch writes here for committed-state fidelity.
            replay_attn_scratch(self, mem, l, pos);
            for (i, v) in ctx.iter().enumerate() {
                mem.w16(lay.attnx + 2 * i as u64, sat16(*v) as u16);
            }
            // O-projection + residual (i32 carrier, no saturation risk).
            let ov = self.proj_b(mem, a.wo, &lw.mo, h, nh * dh, lay.attnx);
            for (c, v) in ov.iter().enumerate() {
                let with_res = v.wrapping_add(mem.r32i(lay.x + 4 * c as u64));
                mem.w32(lay.x + 4 * c as u64, with_res as u32); // ST32 truncation
            }
            // FFN: silu(gate)·up → down, with residual.
            self.rmsnorm_to_i16(mem, lay.x, lay.xn, &lw.norm2, h);
            let gv = self.proj_b(mem, a.w_gate, &lw.m_gate, f, h, lay.xn);
            let uv = self.proj_b(mem, a.w_up, &lw.m_up, f, h, lay.xn);
            for r in 0..f as usize {
                // silu(g) = g·σ(g) with σ from the EXP LUT — gate stays at
                // its per-layer i16 scale; only σ's argument saturates at
                // Q4.11's ±16, where σ is genuinely 0/1 anyway. Every scratch
                // write below is a committed VM cell (boundary equality).
                let g = sat16(gv[r]) as i64;
                let u = sat16(uv[r]) as i64;
                mem.w16(lay.g16, g as u16);
                mem.w16(lay.u16, u as u16);
                mem.w32(lay.up32, u as u32);
                let x411 = rnd(g.wrapping_mul(lw.m_sig as i64), SHIFT);
                mem.w32(lay.t_x, x411 as u32);
                // Sign byte: CLAMP8(x411·2^30) ∈ {0, 127, −128} — the VM's
                // JEQ-on-0x80 sign dispatch.
                mem.w8(lay.t_sign, vm::exec::sat8(x411.wrapping_mul(1 << 30)) as u8);
                // σ via e^{-|g|} only (the exp LUT saturates for positive
                // arguments): σ(g≥0) = 2^28/(2^14+em), σ(g<0) = em·2^14/(2^14+em).
                let em = self.lut(&self.tables.exp, if x411 >= 0 { -x411 } else { x411 });
                let sig = if x411 >= 0 {
                    mem.w32(lay.t_den, (16384 + em) as u32);
                    trunc_div(1i64 << 28, 16384 + em)
                } else {
                    mem.w32(lay.t_ep, em as u32);
                    mem.w32(lay.t_den, (16384 + em) as u32);
                    trunc_div(em << 14, 16384 + em)
                };
                mem.w32(lay.t_sig, sig as u32);
                let hpre = rnd(g.wrapping_mul(sig), 14); // g·σ at s_g
                mem.w32(lay.silu32, hpre as u32);
                let prod = hpre.wrapping_mul(u);
                let hq = rnd(prod.wrapping_mul(lw.m_h[r] as i64), SHIFT);
                mem.w16(lay.h_ffn + 2 * r as u64, sat16(hq) as u16);
            }
            let dv = self.proj_b(mem, a.w_down, &lw.m_down, h, f, lay.h_ffn);
            for (c, v) in dv.iter().enumerate() {
                let with_res = v.wrapping_add(mem.r32i(lay.x + 4 * c as u64));
                mem.w32(lay.x + 4 * c as u64, with_res as u32); // ST32 truncation
            }
        }

        if !decide {
            return None;
        }
        // LM head (tied embeddings), streaming/chunked: logits cycle through
        // ONE page (ARGMAX_OFF pattern) — nothing vocab-sized is committed.
        self.rmsnorm_to_i16(mem, lay.x, lay.xn, &self.im.norm_f, h);
        mem.w32(lay.saved_max, i32::MIN as u32);
        let vocab = self.im.cfg.vocab_size;
        let mut logits = vec![0i64; vocab];
        self.head_logits(mem, &mut logits);
        // Deterministic argmax: strictly-greater, ascending ⇒ lowest id ties.
        // The VM compares the STORED (i32-truncated) logits; ascending scan,
        // strictly greater. Calibrated logits are ~2^20, far from binding.
        let mut win = ((i32::MIN as i64), 0u32);
        for (v, &s) in logits.iter().enumerate() {
            let s32 = (s as i32) as i64;
            if s32 > win.0 {
                win = (s32, v as u32);
            }
        }
        // Committed-state writes of the streaming head: the cycled buffer
        // page retains the LAST chunk's logits; saved_max the final max.
        replay_head_scratch(self, mem, win.0);
        mem.w32(lay.tok, win.1);
        Some(win.1)
    }
}

/// One attention head, READ-ONLY (parallel-safe); ctx written by caller.
#[allow(clippy::needless_range_loop)]
fn attn_head(n: &Native, mem: &FlatMem, l: usize, hd: u64, pos: usize, out: &mut [i64]) {
    let cfg = &n.im.cfg;
    let (dh, nkv, nh) = (
        cfg.head_dim as u64,
        cfg.num_key_value_heads as u64,
        cfg.num_attention_heads as u64,
    );
    let a = &n.lay.layers[l];

    let kvh = hd / (nh / nkv);
    let qs = mem.slice(n.lay.q + hd * dh * 2, (dh * 2) as usize);
    let mut att = vec![0i64; pos + 1];
    let mut mx = i32::MIN as i64;
    for (j, slot) in att.iter_mut().enumerate() {
        let kb = a.kc + j as u64 * nkv * dh * 2 + kvh * dh * 2;
        let acc = dot16(qs, mem.slice(kb, (dh * 2) as usize));
        // ST32 truncation: the VM's att32 cells are i32.
        *slot = (rnd(acc.wrapping_mul(n.im.layers[l].m_logit as i64), SHIFT) as i32) as i64;
        if *slot > mx {
            mx = *slot;
        }
    }
    let mut exps = vec![0i64; pos + 1];
    let mut sum = 0i64;
    for j in 0..=pos {
        exps[j] = n.lut(&n.tables.exp, att[j].wrapping_sub(mx));
        sum = sum.wrapping_add(exps[j]);
    }
    let mut probs = [0i64; MAX_SEQ];
    for j in 0..=pos {
        // Q0.14 i16 probabilities (was Q0.7 — 7 extra bits of softmax
        // resolution feed straight into hidden-state fidelity).
        probs[j] = sat16(trunc_div(exps[j].wrapping_mul(16384), sum)) as i64;
    }
    // ctx = Σ_j p_j(Q14) · V[j](i16) over row-major V (gather along j).
    for slot in out.iter_mut() {
        *slot = 0;
    }
    for (j, &pj) in probs.iter().enumerate().take(pos + 1) {
        let vrow = mem.slice(a.vc + (j as u64 * nkv * dh + kvh * dh) * 2, 2 * dh as usize);
        for (d, slot) in out.iter_mut().enumerate() {
            let vv = i16::from_le_bytes([vrow[2 * d], vrow[2 * d + 1]]) as i64;
            *slot = slot.wrapping_add(pj.wrapping_mul(vv));
        }
    }
    for slot in out.iter_mut() {
        *slot = rnd(*slot, 14);
    }
}

/// Replay the LAST head's attention scratch into committed memory (att32,
/// e32, probs, sum, neg_max) — the VM's scratch is whatever the final head
/// left behind; parallel heads must not race those writes.
#[allow(clippy::needless_range_loop)]
fn replay_attn_scratch(n: &Native, mem: &mut FlatMem, l: usize, pos: usize) {
    let cfg = &n.im.cfg;
    let (dh, nkv, nh) = (
        cfg.head_dim as u64,
        cfg.num_key_value_heads as u64,
        cfg.num_attention_heads as u64,
    );
    let a = &n.lay.layers[l];
    let hd = nh - 1;
    let kvh = hd / (nh / nkv);
    let qs = mem.slice(n.lay.q + hd * dh * 2, (dh * 2) as usize).to_vec();
    let mut mx = i32::MIN as i64;
    let mut att = vec![0i64; pos + 1];
    for (j, slot) in att.iter_mut().enumerate() {
        let kb = a.kc + j as u64 * nkv * dh * 2 + kvh * dh * 2;
        let acc = dot16(&qs, mem.slice(kb, (dh * 2) as usize));
        *slot = (rnd(acc.wrapping_mul(n.im.layers[l].m_logit as i64), SHIFT) as i32) as i64;
        if *slot > mx {
            mx = *slot;
        }
    }
    let mut sum = 0i64;
    for j in 0..=pos {
        let e = n.lut(&n.tables.exp, att[j].wrapping_sub(mx));
        mem.w32(n.lay.e32 + 4 * j as u64, e as u32);
        mem.w32(n.lay.att32 + 4 * j as u64, att[j] as u32);
        sum = sum.wrapping_add(e);
    }
    mem.w32(n.lay.neg_max, mx.wrapping_mul(-1) as u32);
    mem.w32(n.lay.sum, sum as u32);
    for j in 0..=pos {
        let e = mem.r32i(n.lay.e32 + 4 * j as u64);
        let p = sat16(trunc_div(e.wrapping_mul(16384), sum));
        mem.w16(n.lay.probs + 2 * j as u64, p as u16);
    }
}

/// The streaming head's committed scratch: final saved_max value (the
/// winning logit) — the logit_buf page content is the last chunk's values,
/// reproduced cheaply here as the final chunk only.
fn replay_head_scratch(n: &Native, mem: &mut FlatMem, max_logit: i64) {
    let h = n.im.cfg.hidden_size as u64;
    let vocab = n.im.cfg.vocab_size;
    // The cycled page's survivors are the LAST writer of each slot — i.e.
    // the final 256 rows (the tail chunk covers offsets 0..tail, the chunk
    // before it the rest).
    let x = mem.slice(n.lay.xn, 2 * h as usize).to_vec();
    let blocks = h as usize / 64;
    for v in vocab - 256..vocab {
        let wrow = mem.slice(n.lay.head_w + v as u64 * h, h as usize).to_vec();
        let mut acc = 0i64;
        for b in 0..blocks {
            let p = dot_w8_x16(&wrow[b * 64..(b + 1) * 64], &x[b * 128..(b + 1) * 128]);
            acc = acc.wrapping_add(p.wrapping_mul(n.im.m_head.0[v * blocks + b] as i64));
        }
        let s = rnd(acc, n.im.m_head.1);
        mem.w32(n.lay.logit_buf + 4 * (v % 256) as u64, s as u32);
    }
    mem.w32(n.lay.saved_max, max_logit as u32);
    mem.w32(n.lay.v_cell, vocab as u32);
}

// ---------------------------------------------------------------------------
// Full decode runs
// ---------------------------------------------------------------------------

/// Decode with per-token checkpoint commitments; returns stats + roots.
pub fn run_committed(
    n: &Native,
    image: Vec<u8>,
    prompt: &[u32],
    n_gen: usize,
) -> RunStats {
    let mut stats = RunStats::default();
    let t0 = Instant::now();
    let mut mem = FlatMem::new(image);
    let leaves: Vec<Hash> = mem.bytes.chunks_exact(PAGE_SIZE).map(page_leaf_hash).collect();
    let mut tree =
        MerkleTree::from_leaf_hashes(MEM_DEPTH, leaves, page_leaf_hash(&[0u8; PAGE_SIZE]));
    stats.genesis_us = t0.elapsed().as_micros();

    let mut tok = prompt[0];
    let n_pos = prompt.len() + n_gen - 1;
    for pos in 0..n_pos {
        let decide = pos >= prompt.len() - 1;
        let tc = Instant::now();
        let next = n.position(&mut mem, pos, tok, decide);
        stats.compute_us += tc.elapsed().as_micros();

        let th = Instant::now();
        let dirty = mem.take_dirty();
        stats.dirty_pages += dirty.len();
        let updates: Vec<(u64, Hash)> = dirty
            .iter()
            .map(|pg| (*pg, page_leaf_hash(mem.slice(*pg * PAGE_SIZE as u64, PAGE_SIZE))))
            .collect();
        tree.update_leaf_hashes_bulk(&updates);
        stats.boundary_mem_roots.push(tree.root());
        // Provisional register file (see module docs).
        let regs = Registers {
            pc: 0,
            halted: 0,
            step: pos as u64 + 1,
            acc: 0,
            aux: 0,
            idx: [tok, 0, 0, 0],
        };
        stats.boundary_roots.push(state_root(&tree.root(), &regs.encode()));
        stats.hash_us += th.elapsed().as_micros();

        if let Some(t) = next {
            stats.tokens.push(t);
            if t == n.im.cfg.eos_token_id {
                break;
            }
            tok = t;
        } else {
            tok = prompt[pos + 1];
        }
    }
    stats
}

/// Pipelined commitment: a hasher thread owns the Merkle tree and digests
/// each position's dirty-page snapshot while the main thread computes the
/// next position. Wall-clock ≈ max(compute, hash) per token instead of
/// compute + hash — with hash ≪ compute the overhead is just the snapshot
/// memcpy. Roots are identical to the sequential path (same pages, same
/// tree, same registers).
pub fn run_committed_pipelined(
    n: &Native,
    image: Vec<u8>,
    prompt: &[u32],
    n_gen: usize,
) -> (RunStats, u128) {
    let mut stats = RunStats::default();
    let t0 = Instant::now();
    let mut mem = FlatMem::new(image);
    let leaves: Vec<Hash> = mem.bytes.chunks_exact(PAGE_SIZE).map(page_leaf_hash).collect();
    let mut tree =
        MerkleTree::from_leaf_hashes(MEM_DEPTH, leaves, page_leaf_hash(&[0u8; PAGE_SIZE]));
    stats.genesis_us = t0.elapsed().as_micros();

    type Job = (Vec<(u64, Vec<u8>)>, Registers);
    let (tx, rx) = std::sync::mpsc::channel::<Job>();
    let wall = Instant::now();
    let mut roots = Vec::new();
    std::thread::scope(|sc| {
        let hasher = sc.spawn(move || {
            let mut out = Vec::new();
            while let Ok((pages, regs)) = rx.recv() {
                let updates: Vec<(u64, Hash)> = pages
                    .iter()
                    .map(|(idx, bytes)| (*idx, page_leaf_hash(bytes)))
                    .collect();
                tree.update_leaf_hashes_bulk(&updates);
                out.push(state_root(&tree.root(), &regs.encode()));
            }
            out
        });

        let mut tok = prompt[0];
        let n_pos = prompt.len() + n_gen - 1;
        for pos in 0..n_pos {
            let decide = pos >= prompt.len() - 1;
            let next = n.position(&mut mem, pos, tok, decide);
            // Snapshot dirty pages (small memcpy) and hand off.
            let dirty = mem.take_dirty();
            stats.dirty_pages += dirty.len();
            let snapshot: Vec<(u64, Vec<u8>)> = dirty
                .iter()
                .map(|pg| (*pg, mem.slice(*pg * PAGE_SIZE as u64, PAGE_SIZE).to_vec()))
                .collect();
            let regs = Registers {
                pc: 0,
                halted: 0,
                step: pos as u64 + 1,
                acc: 0,
                aux: 0,
                idx: [tok, 0, 0, 0],
            };
            tx.send((snapshot, regs)).expect("hasher alive");
            if let Some(t) = next {
                stats.tokens.push(t);
                if t == n.im.cfg.eos_token_id {
                    break;
                }
                tok = t;
            } else {
                tok = prompt[pos + 1];
            }
        }
        drop(tx);
        roots = hasher.join().expect("hasher");
    });
    let wall_us = wall.elapsed().as_micros();
    stats.boundary_roots = roots;
    (stats, wall_us)
}

/// Pure decode, no commitments — the "ordinary inference" baseline.
pub fn run_pure(n: &Native, image: Vec<u8>, prompt: &[u32], n_gen: usize) -> (Vec<u32>, u128) {
    let mut mem = FlatMem::new(image);
    let t0 = Instant::now();
    let mut tok = prompt[0];
    let mut tokens = Vec::new();
    let n_pos = prompt.len() + n_gen - 1;
    for pos in 0..n_pos {
        let decide = pos >= prompt.len() - 1;
        if let Some(t) = n.position(&mut mem, pos, tok, decide) {
            tokens.push(t);
            if t == n.im.cfg.eos_token_id {
                break;
            }
            tok = t;
        } else {
            tok = prompt[pos + 1];
        }
    }
    (tokens, t0.elapsed().as_micros())
}
