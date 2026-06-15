//! FW-6 committed memory layout for the FLOAT Qwen judge. Float values are
//! stored as their IEEE-754 bit patterns in u32 cells; bf16 weights as raw
//! u16 bits. Single source of truth for the committed native runtime
//! (fc.rs) and the float compiler (compiler/src/fqwen.rs).
//!
//! Alignment rules from the float ISA: FDOT A-lines (bf16 weights) are
//! 128-byte/128-aligned ⇒ weight matrices page-aligned with row stride
//! h·2 (2048 | 128 ✓); FDOT B-lines (f32 activations) are 256-byte/
//! 256-aligned ⇒ xn / attnx / h_ffn aligned to 256.

use crate::config::QwenConfig;
use vm::PAGE_SIZE;

/// 2^21 pages = 2 GiB: bf16 weights (~1.5 GiB) + f32 KV + tables.
pub const FMEM_DEPTH: u8 = 21;
pub const FMAX_SEQ: usize = 512;

#[derive(Clone, Debug)]
pub struct FLayerAddrs {
    pub wq: u64,
    pub wk: u64,
    pub wv: u64,
    pub wo: u64,
    pub w_gate: u64,
    pub w_up: u64,
    pub w_down: u64,
    pub ln1: u64,
    pub ln2: u64,
    pub q_norm: u64,
    pub k_norm: u64,
    /// K/V caches, f32 row-major [FMAX_SEQ][nkv·dh].
    pub kc: u64,
    pub vc: u64,
}

#[derive(Clone, Debug)]
pub struct FLayout {
    // f32 constant cells (bit patterns written at genesis)
    pub c_one: u64,      // 1.0
    pub c_neg1f: u64,    // −1.0
    pub c_half: u64,     // 0.5
    pub c_log2e: u64,    // log2(e)
    pub c_nln2hi: u64,   // −LN2_HI
    pub c_nln2lo: u64,   // −LN2_LO
    pub c_exp6: u64,     // 1/720
    pub c_exp5: u64,     // 1/120
    pub c_exp4: u64,     // 1/24
    pub c_exp3: u64,     // 1/6
    pub c_n87: u64,      // −87.0
    pub c_p88: u64,      // 88.0
    pub c_eps: u64,      // 1e-6
    pub c_isqdh: u64,    // 1/sqrt(dh)
    pub c_hf: u64,       // h as f32
    pub c_dhf: u64,      // dh as f32
    // integer constant cells
    pub c_i127: u64,     // 127
    pub c_i2p23: u64,    // 1<<23
    pub c_i65536: u64,   // 1<<16 (bf16 widening multiplier)
    pub c_izero: u64,    // 0 (canonical resets, FGT-false anchor)
    // scratch cells (f32/u32 bits)
    pub facc: u64,   // row/dot accumulator
    pub t1: u64,     // cexp: kf / staging
    pub t2: u64,     // cexp: r
    pub t3: u64,     // cexp: poly / two_k
    pub t4: u64,     // silu staging
    pub flag: u64,   // FGT output (u32 0/1)
    pub fm: u64,     // softmax running max
    pub fsum: u64,   // softmax sum
    pub fr: u64,     // norm scale r
    pub v_cell: u64, // absolute head-row counter
    pub tok: u64,    // decode feedback (u32 token id)
    pub saved_max: u64, // best logit (f32 bits)
    pub input: u64,  // [count][ids…] page
    pub logit_buf: u64, // ONE cycled page of f32 logits
    // state arrays (f32 bits)
    pub x: u64,      // [h]
    pub xn: u64,     // [h]   (256-aligned: FDOT B)
    pub q: u64,      // [nh·dh]
    pub attnx: u64,  // [nh·dh] (256-aligned)
    pub gate: u64,   // [f]
    pub up: u64,     // [f]
    pub h_ffn: u64,  // [f]   (256-aligned)
    pub scores: u64, // [FMAX_SEQ]
    // weights / tables
    pub emb: u64,    // [vocab][h] bf16 (tied head)
    pub ln_f: u64,   // [h] f32
    pub rope_cos: u64, // [FMAX_SEQ][dh/2] f32
    pub rope_sin: u64,
    pub layers: Vec<FLayerAddrs>,
    pub end: u64,
}

fn align(x: u64, a: u64) -> u64 {
    x.div_ceil(a) * a
}

impl FLayout {
    pub fn new(cfg: &QwenConfig) -> Self {
        let pg = PAGE_SIZE as u64;
        let (h, dh, f) =
            (cfg.hidden_size as u64, cfg.head_dim as u64, cfg.intermediate_size as u64);
        let (nh, nkv) = (cfg.num_attention_heads as u64, cfg.num_key_value_heads as u64);
        let seq = FMAX_SEQ as u64;
        let mut at = 0u64;
        let mut take = |n: u64, a: u64| {
            at = align(at, a);
            let b = at;
            at += n;
            b
        };
        let c_one = take(4, 4);
        let c_neg1f = take(4, 4);
        let c_half = take(4, 4);
        let c_log2e = take(4, 4);
        let c_nln2hi = take(4, 4);
        let c_nln2lo = take(4, 4);
        let c_exp6 = take(4, 4);
        let c_exp5 = take(4, 4);
        let c_exp4 = take(4, 4);
        let c_exp3 = take(4, 4);
        let c_n87 = take(4, 4);
        let c_p88 = take(4, 4);
        let c_eps = take(4, 4);
        let c_isqdh = take(4, 4);
        let c_hf = take(4, 4);
        let c_dhf = take(4, 4);
        let c_i127 = take(4, 4);
        let c_i2p23 = take(4, 4);
        let c_i65536 = take(4, 4);
        let c_izero = take(4, 4);
        let facc = take(4, 4);
        let t1 = take(4, 4);
        let t2 = take(4, 4);
        let t3 = take(4, 4);
        let t4 = take(4, 4);
        let flag = take(4, 4);
        let fm = take(4, 4);
        let fsum = take(4, 4);
        let fr = take(4, 4);
        let v_cell = take(4, 4);
        let tok = take(4, 4);
        let saved_max = take(4, 4);
        let input = take(pg, pg);
        let logit_buf = take(pg, pg);
        let x = take(h * 4, 256);
        let xn = take(h * 4, 256);
        let q = take(nh * dh * 4, 256);
        let attnx = take(nh * dh * 4, 256);
        let gate = take(f * 4, 256);
        let up = take(f * 4, 256);
        let h_ffn = take(f * 4, 256);
        let scores = take(seq * 4, 64);
        let emb = take(cfg.vocab_size as u64 * h * 2, pg);
        let ln_f = take(h * 4, 64);
        let rope_cos = take(seq * (dh / 2) * 4, 64);
        let rope_sin = take(seq * (dh / 2) * 4, 64);
        let layers = (0..cfg.num_hidden_layers)
            .map(|_| FLayerAddrs {
                wq: take(nh * dh * h * 2, pg),
                wk: take(nkv * dh * h * 2, pg),
                wv: take(nkv * dh * h * 2, pg),
                wo: take(h * nh * dh * 2, pg),
                w_gate: take(f * h * 2, pg),
                w_up: take(f * h * 2, pg),
                w_down: take(h * f * 2, pg),
                ln1: take(h * 4, 64),
                ln2: take(h * 4, 64),
                q_norm: take(dh * 4, 64),
                k_norm: take(dh * 4, 64),
                kc: take(seq * nkv * dh * 4, pg),
                vc: take(seq * nkv * dh * 4, pg),
            })
            .collect();
        let end = at;
        assert!(end <= (1u64 << FMEM_DEPTH) * pg, "float layout exceeds 2 GiB: {end}");
        Self {
            c_one, c_neg1f, c_half, c_log2e, c_nln2hi, c_nln2lo,
            c_exp6, c_exp5, c_exp4, c_exp3, c_n87, c_p88, c_eps, c_isqdh,
            c_hf, c_dhf, c_i127, c_i2p23, c_i65536, c_izero,
            facc, t1, t2, t3, t4, flag, fm, fsum, fr, v_cell, tok, saved_max,
            input, logit_buf,
            x, xn, q, attnx, gate, up, h_ffn, scores,
            emb, ln_f, rope_cos, rope_sin, layers, end,
        }
    }
}

/// Build the committed genesis image for the float judge: bf16 weights,
/// f32 norm/rope tables, committed constants, prompt tokens.
#[allow(clippy::float_arithmetic)] // f32 constants become committed bit patterns
pub fn fgenesis(lay: &FLayout, m: &crate::fmodel::FModel, prompt: &[u32]) -> Vec<u8> {
    let mut img = vec![0u8; (1usize << FMEM_DEPTH) * PAGE_SIZE];
    let put_u32 = |img: &mut Vec<u8>, at: u64, v: u32| {
        img[at as usize..at as usize + 4].copy_from_slice(&v.to_le_bytes());
    };
    let put_f32 = |img: &mut Vec<u8>, at: u64, v: f32| {
        img[at as usize..at as usize + 4].copy_from_slice(&v.to_bits().to_le_bytes());
    };
    let put_bf16 = |img: &mut Vec<u8>, at: u64, v: &[u16]| {
        for (i, &b) in v.iter().enumerate() {
            img[at as usize + 2 * i..at as usize + 2 * i + 2].copy_from_slice(&b.to_le_bytes());
        }
    };
    let put_f32s = |img: &mut Vec<u8>, at: u64, v: &[f32]| {
        for (i, &x) in v.iter().enumerate() {
            img[at as usize + 4 * i..at as usize + 4 * i + 4]
                .copy_from_slice(&x.to_bits().to_le_bytes());
        }
    };
    let cfg = &m.cfg;
    // Constants — the EXACT f32 bit patterns fmath::cexp and fmodel use.
    put_f32(&mut img, lay.c_one, 1.0);
    put_f32(&mut img, lay.c_neg1f, -1.0);
    put_f32(&mut img, lay.c_half, 0.5);
    put_f32(&mut img, lay.c_log2e, core::f32::consts::LOG2_E);
    put_f32(&mut img, lay.c_nln2hi, -0.693_359_4_f32);
    put_f32(&mut img, lay.c_nln2lo, 2.121_944_4e-4_f32);
    put_f32(&mut img, lay.c_exp6, 1.0 / 720.0);
    put_f32(&mut img, lay.c_exp5, 1.0 / 120.0);
    put_f32(&mut img, lay.c_exp4, 1.0 / 24.0);
    put_f32(&mut img, lay.c_exp3, 1.0 / 6.0);
    put_f32(&mut img, lay.c_n87, -87.0);
    put_f32(&mut img, lay.c_p88, 88.0);
    put_f32(&mut img, lay.c_eps, 1.0 / (cfg.rms_norm_eps_recip as f32));
    put_f32(&mut img, lay.c_isqdh, 1.0 / (cfg.head_dim as f32).sqrt());
    put_f32(&mut img, lay.c_hf, cfg.hidden_size as f32);
    put_f32(&mut img, lay.c_dhf, cfg.head_dim as f32);
    put_u32(&mut img, lay.c_i127, 127);
    put_u32(&mut img, lay.c_i2p23, 1 << 23);
    put_u32(&mut img, lay.c_i65536, 1 << 16);
    // c_izero stays 0 (genesis default).
    // Input page: [count][ids…] (same convention as the integer judge).
    put_u32(&mut img, lay.input, prompt.len() as u32);
    for (i, &t) in prompt.iter().enumerate() {
        put_u32(&mut img, lay.input + 4 + 4 * i as u64, t);
    }
    // Weights + tables.
    put_bf16(&mut img, lay.emb, &m.emb);
    put_f32s(&mut img, lay.ln_f, &m.ln_f);
    let half = cfg.head_dim / 2;
    put_f32s(&mut img, lay.rope_cos, &m.rope_cos[..FMAX_SEQ * half]);
    put_f32s(&mut img, lay.rope_sin, &m.rope_sin[..FMAX_SEQ * half]);
    for (l, a) in lay.layers.iter().enumerate() {
        let lw = &m.layers[l];
        put_bf16(&mut img, a.wq, &lw.wq);
        put_bf16(&mut img, a.wk, &lw.wk);
        put_bf16(&mut img, a.wv, &lw.wv);
        put_bf16(&mut img, a.wo, &lw.wo);
        put_bf16(&mut img, a.w_gate, &lw.w_gate);
        put_bf16(&mut img, a.w_up, &lw.w_up);
        put_bf16(&mut img, a.w_down, &lw.w_down);
        put_f32s(&mut img, a.ln1, &lw.ln1);
        put_f32s(&mut img, a.ln2, &lw.ln2);
        put_f32s(&mut img, a.q_norm, &lw.q_norm);
        put_f32s(&mut img, a.k_norm, &lw.k_norm);
    }
    img
}
