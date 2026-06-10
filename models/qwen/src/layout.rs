//! Committed memory layout for the Qwen judge (SPEC §7.2 region map),
//! config-driven. Single source of truth for the native runtime and the
//! VM compiler. All tensor bases 64-aligned; weights page-aligned.

use crate::config::QwenConfig;
use vm::PAGE_SIZE;

/// 2^20 pages = 1 GiB: ~580 MiB used by Qwen3-0.6B INT8 + KV + LUTs.
pub const MEM_DEPTH: u8 = 20;

/// Demo context budget (prompt + generation). Attention scratch and rope
/// tables are sized to it; a multiple of 64 so prob lines stay DOT-able.
pub const MAX_SEQ: usize = 96;

#[derive(Clone, Debug)]
pub struct LayerAddrs {
    pub wq: u64,
    pub wk: u64,
    pub wv: u64,
    pub wo: u64,
    pub w_gate: u64,
    pub w_up: u64,
    pub w_down: u64,
    pub mq: u64,
    pub mk: u64,
    pub mv: u64,
    pub mo: u64,
    pub m_gate: u64,
    pub m_up: u64,
    pub m_down: u64,
    pub g1: u64,
    pub g2: u64,
    pub gq: u64,
    pub gk: u64,
    /// Per-layer scalar multiplier cells: logit, ffn-h product.
    pub m_logit_c: u64,
    pub m_h_c: u64,
    /// K cache [MAX_SEQ][kv_heads·head_dim] i16 — rows are DOT16 lines.
    pub kc: u64,
    /// V cache, row-major [MAX_SEQ][kv_heads·head_dim] i8: appends touch
    /// ONE page (the transpose layout dirtied ~96 pages per layer per
    /// token — measured 2.7 MB/position, the dominant commitment cost).
    /// The VM pays per-element MACs for prob·V instead of DOT8 lines; the
    /// honest path owns the priority (FW-7 notes an AXPY line-op fixing
    /// both).
    pub vc: u64,
}

#[derive(Clone, Debug)]
pub struct QwenLayout {
    // constants
    pub c_one_i8: u64,
    pub c_h: u64,      // i32 hidden_size (RMSNorm divisor)
    pub c_dh: u64,     // i32 head_dim (QK-norm divisor)
    pub c_2p14: u64,   // i32 16384
    pub c_neg1: u64,   // i32 −1
    pub c_m_logit: u64,
    pub c_m_h: u64,
    pub c_m_emb: u64,  // embedding-row → residual-scale multiplier
    pub c_i32min: u64, // saved_max reset value
    // scratch
    pub x: u64,        // [h] i32 residual carrier (one global scale)
    pub xn: u64,       // [h] i16 post-norm (matmul inputs need outlier headroom)
    pub q: u64,        // [nh·dh] i16
    pub attnx: u64,    // [nh·dh] i8 ctx concat
    pub att32: u64,    // [MAX_SEQ] i32
    pub e32: u64,      // [MAX_SEQ] i32
    pub probs: u64,    // [MAX_SEQ] i8 Q0.7
    pub r32: u64,
    pub sum: u64,
    pub neg_max: u64,
    pub tok: u64,
    pub silu32: u64,   // i32 cell: silu(gate) Q4.11
    pub up32: u64,     // i32 cell: up Q4.11
    pub h_ffn: u64,    // [ffn] i8
    pub logit_buf: u64, // ONE page cycled by the chunked head (ARGMAX_OFF)
    pub saved_max: u64,
    // io
    pub input: u64,
    pub output: u64,
    // weights
    pub emb: u64, // [vocab][h] i8, per-tensor scale; tied LM head
    pub gf: u64,
    pub layers: Vec<LayerAddrs>,
    // tables
    pub rope_cos: u64,
    pub rope_sin: u64,
    pub rope_nsin: u64,
    pub lut_exp: u64,
    pub lut_rsqrt: u64,
    pub lut_silu: u64,
    pub end: u64,
}

fn align(x: u64, a: u64) -> u64 {
    x.div_ceil(a) * a
}

impl QwenLayout {
    pub fn new(cfg: &QwenConfig) -> Self {
        let pg = PAGE_SIZE as u64;
        let (h, dh, f) = (
            cfg.hidden_size as u64,
            cfg.head_dim as u64,
            cfg.intermediate_size as u64,
        );
        let (nh, nkv) = (cfg.num_attention_heads as u64, cfg.num_key_value_heads as u64);
        let seq = MAX_SEQ as u64;
        let mut at = 0u64;
        let mut take = |n: u64, a: u64| {
            at = align(at, a);
            let b = at;
            at += n;
            b
        };
        let c_one_i8 = take(1, 1);
        let c_h = take(4, 4);
        let c_dh = take(4, 4);
        let c_2p14 = take(4, 4);
        let c_neg1 = take(4, 4);
        let c_m_logit = take(4, 4);
        let c_m_h = take(4, 4);
        let c_m_emb = take(4, 4);
        let c_i32min = take(4, 4);
        let x = take(h * 4, 64);
        let xn = take(h * 2, 64);
        let q = take(nh * dh * 2, 64);
        let attnx = take(nh * dh * 2, 64);
        let att32 = take(seq * 4, 64);
        let e32 = take(seq * 4, 64);
        let probs = take(seq, 64);
        let r32 = take(4, 4);
        let sum = take(4, 4);
        let neg_max = take(4, 4);
        let tok = take(4, 4);
        let silu32 = take(4, 4);
        let up32 = take(4, 4);
        let h_ffn = take(f * 2, 64);
        let logit_buf = take(pg, pg);
        let saved_max = take(4, 4);
        let input = take(pg, pg);
        let output = take(pg, pg);
        let emb = take(cfg.vocab_size as u64 * h, pg);
        let gf = take(h * 4, 64);
        let layers = (0..cfg.num_hidden_layers)
            .map(|_| LayerAddrs {
                wq: take(nh * dh * h, pg),
                wk: take(nkv * dh * h, pg),
                wv: take(nkv * dh * h, pg),
                wo: take(h * nh * dh, pg),
                w_gate: take(f * h, pg),
                w_up: take(f * h, pg),
                w_down: take(h * f, pg),
                mq: take(nh * dh * 4, 64),
                mk: take(nkv * dh * 4, 64),
                mv: take(nkv * dh * 4, 64),
                mo: take(h * 4, 64),
                m_gate: take(f * 4, 64),
                m_up: take(f * 4, 64),
                m_down: take(h * 4, 64),
                g1: take(h * 4, 64),
                g2: take(h * 4, 64),
                gq: take(dh * 4, 64),
                gk: take(dh * 4, 64),
                m_logit_c: take(4, 4),
                m_h_c: take(4, 4),
                kc: take(seq * nkv * dh * 2, pg),
                vc: take(seq * nkv * dh, pg),
            })
            .collect();
        let rope_cos = take(seq * dh, 64); // dh/2 pairs × i16
        let rope_sin = take(seq * dh, 64);
        let rope_nsin = take(seq * dh, 64);
        let lut_exp = take(131072, pg);
        let lut_rsqrt = take(131072, pg);
        let lut_silu = take(131072, pg);
        let end = at;
        assert!(end <= (1u64 << MEM_DEPTH) * pg, "layout exceeds 1 GiB: {end}");
        Self {
            c_one_i8, c_h, c_dh, c_2p14, c_neg1, c_m_logit, c_m_h, c_m_emb, c_i32min,
            x, xn, q, attnx, att32, e32, probs, r32, sum, neg_max, tok,
            silu32, up32, h_ffn, logit_buf, saved_max, input, output,
            emb, gf, layers, rope_cos, rope_sin, rope_nsin,
            lut_exp, lut_rsqrt, lut_silu, end,
        }
    }
}
