//! FW-6 committed-float Qwen3: "the model as published", deterministically.
//!
//! Weights stay bf16-resident (their published form — widening to f32 is
//! exact); all arithmetic is f32 in the PINNED order defined by
//! kernels::fkernels (the canonical 16-lane/64-block reduction tree) and
//! qwen::fmath (committed exp/sigmoid/rsqrt — no libm anywhere at runtime).
//! Rope cos/sin are frozen-artifact tables (computed once at model prep,
//! committed by hash like the weights — runtime only reads them).
//!
//! Because every value's computation order is pinned, ANY execution
//! backend that implements the same tree — scalar, NEON, a GPU shader —
//! produces bit-identical f32s. That is FW-6's whole point: the model's own
//! float math, exact quality by identity, deterministic enough to dispute.
#![allow(clippy::float_arithmetic)] // FW-6: floats are the committed semantics

use crate::config::QwenConfig;
use crate::fmath::{crsqrt, csilu, csoftmax};
use crate::tensors::SafeTensors;
use kernels::fkernels::{bf16_to_f32, fgemv};
use kernels::Pool;

pub struct FLayer {
    pub wq: Vec<u16>,
    pub wk: Vec<u16>,
    pub wv: Vec<u16>,
    pub wo: Vec<u16>,
    pub w_gate: Vec<u16>,
    pub w_up: Vec<u16>,
    pub w_down: Vec<u16>,
    pub ln1: Vec<f32>,
    pub ln2: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
}

pub struct FModel {
    pub cfg: QwenConfig,
    /// Tied embedding/LM-head matrix, bf16-resident ([vocab][hidden]).
    pub emb: Vec<u16>,
    pub layers: Vec<FLayer>,
    pub ln_f: Vec<f32>,
    /// Frozen rope tables [pos][dh/2], f32 (artifact-committed; see module docs).
    pub rope_cos: Vec<f32>,
    pub rope_sin: Vec<f32>,
    pub max_seq: usize,
}

impl FModel {
    pub fn load(cfg: &QwenConfig, st: &SafeTensors, max_seq: usize) -> Self {
        let f32v = |name: &str| -> Vec<f32> {
            st.bf16_bits(name).iter().map(|&b| bf16_to_f32(b)).collect()
        };
        assert!(cfg.tie_word_embeddings);
        let layers = (0..cfg.num_hidden_layers)
            .map(|l| {
                let p = format!("model.layers.{l}");
                FLayer {
                    wq: st.bf16_bits(&format!("{p}.self_attn.q_proj.weight")),
                    wk: st.bf16_bits(&format!("{p}.self_attn.k_proj.weight")),
                    wv: st.bf16_bits(&format!("{p}.self_attn.v_proj.weight")),
                    wo: st.bf16_bits(&format!("{p}.self_attn.o_proj.weight")),
                    w_gate: st.bf16_bits(&format!("{p}.mlp.gate_proj.weight")),
                    w_up: st.bf16_bits(&format!("{p}.mlp.up_proj.weight")),
                    w_down: st.bf16_bits(&format!("{p}.mlp.down_proj.weight")),
                    ln1: f32v(&format!("{p}.input_layernorm.weight")),
                    ln2: f32v(&format!("{p}.post_attention_layernorm.weight")),
                    q_norm: f32v(&format!("{p}.self_attn.q_norm.weight")),
                    k_norm: f32v(&format!("{p}.self_attn.k_norm.weight")),
                }
            })
            .collect();
        // Frozen-artifact rope tables (offline trig; runtime never calls
        // libm). f64 intermediates, cast to the committed f32 values.
        let half = cfg.head_dim / 2;
        let mut rope_cos = vec![0f32; max_seq * half];
        let mut rope_sin = vec![0f32; max_seq * half];
        for pos in 0..max_seq {
            for i in 0..half {
                let freq = (cfg.rope_theta as f64).powf(-2.0 * i as f64 / cfg.head_dim as f64);
                let ang = pos as f64 * freq;
                rope_cos[pos * half + i] = ang.cos() as f32;
                rope_sin[pos * half + i] = ang.sin() as f32;
            }
        }
        Self {
            cfg: cfg.clone(),
            emb: st.bf16_bits("model.embed_tokens.weight"),
            layers,
            ln_f: f32v("model.norm.weight"),
            rope_cos,
            rope_sin,
            max_seq,
        }
    }
}

/// KV cache, f32: [layer][pos][nkv·dh].
pub struct FState {
    pub kc: Vec<f32>,
    pub vc: Vec<f32>,
    pub pos: usize,
}

impl FState {
    pub fn new(cfg: &QwenConfig, max_seq: usize) -> Self {
        let per = cfg.num_key_value_heads * cfg.head_dim;
        Self {
            kc: vec![0.0; cfg.num_hidden_layers * max_seq * per],
            vc: vec![0.0; cfg.num_hidden_layers * max_seq * per],
            pos: 0,
        }
    }
}

/// Committed f32×f32 sum-of-products for the SMALL dots (norm sums over h,
/// attention scores over dh): a SEQUENTIAL fused-fma chain — one FOP fma
/// micro-op per element in the VM, the simplest possible committed shape.
/// (The big GEMVs keep the 16-lane tree via FDOT; these dots are ≤1024
/// elements and not performance-relevant.)
fn fdot32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0f32;
    for (x, y) in a.iter().zip(b) {
        acc = x.mul_add(*y, acc);
    }
    acc
}

/// Committed RMS-norm: out[c] = (x[c]·r)·γ[c], r = 1/sqrt(mean(x²) + eps).
fn rmsnorm(x: &[f32], gamma: &[f32], eps: f32, out: &mut [f32]) {
    let ss = fdot32(x, x);
    let mean = ss / x.len() as f32;
    let r = crsqrt(mean + eps);
    for c in 0..x.len() {
        out[c] = (x[c] * r) * gamma[c];
    }
}

/// Reused per-position scratch (allocation-free decode loop).
pub struct FScratch {
    x: Vec<f32>,
    xn: Vec<f32>,
    q: Vec<f32>,
    attnx: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    down: Vec<f32>,
    hbuf: Vec<f32>,
    scores: Vec<f32>,
}

impl FScratch {
    pub fn new(cfg: &QwenConfig, max_seq: usize) -> Self {
        let (h, dh) = (cfg.hidden_size, cfg.head_dim);
        let nh = cfg.num_attention_heads;
        Self {
            x: vec![0.0; h],
            xn: vec![0.0; h],
            q: vec![0.0; nh * dh],
            attnx: vec![0.0; nh * dh],
            gate: vec![0.0; cfg.intermediate_size],
            up: vec![0.0; cfg.intermediate_size],
            down: vec![0.0; h],
            hbuf: vec![0.0; dh],
            scores: vec![0.0; max_seq],
        }
    }
}

/// One decode position under committed semantics; returns argmax token if
/// `decide`. Logits land in `logits_out` (vocab-sized) when decide.
#[allow(clippy::too_many_arguments)]
pub fn fposition(
    m: &FModel,
    st: &mut FState,
    sc: &mut FScratch,
    pool: &Pool,
    tok: u32,
    decide: bool,
    logits_out: &mut [f32],
) -> Option<u32> {
    let cfg = &m.cfg;
    let (h, dh) = (cfg.hidden_size, cfg.head_dim);
    let (nh, nkv) = (cfg.num_attention_heads, cfg.num_key_value_heads);
    let half = dh / 2;
    let group = nh / nkv;
    let kv_per = nkv * dh;
    let pos = st.pos;
    let eps = 1.0 / (cfg.rms_norm_eps_recip as f32); // 1e-6, exact
    let inv_sqrt_dh = 1.0 / (dh as f32).sqrt(); // IEEE sqrt+div: pinned

    // Embedding: exact bf16→f32 widening (lookup, no arithmetic).
    let FScratch { x, xn, q, attnx, gate, up, down, hbuf, scores } = sc;
    for (c, slot) in x.iter_mut().enumerate() {
        *slot = bf16_to_f32(m.emb[tok as usize * h + c]);
    }

    for (l, lw) in m.layers.iter().enumerate() {
        rmsnorm(x, &lw.ln1, eps, xn);
        // q/k/v under the committed GEMV tree (row-parallel, bit-stable).
        fgemv(pool, &lw.wq, xn, nh * dh, h, q);
        let kbase = (l * m.max_seq + pos) * kv_per;
        fgemv(pool, &lw.wk, xn, kv_per, h, &mut st.kc[kbase..kbase + kv_per]);
        fgemv(pool, &lw.wv, xn, kv_per, h, &mut st.vc[kbase..kbase + kv_per]);

        // Per-head QK-norm + rope, q heads then k heads (pinned order).
        let rotate = |v: &mut [f32]| {
            for p2 in 0..half {
                let c = m.rope_cos[pos * half + p2];
                let s = m.rope_sin[pos * half + p2];
                let (a, b) = (v[p2], v[p2 + half]);
                let t = b * s;
                let na = a.mul_add(c, -t);
                let t2 = a * s;
                let nb = b.mul_add(c, t2);
                v[p2] = na;
                v[p2 + half] = nb;
            }
        };
        for hd in 0..nh {
            let qs = &mut q[hd * dh..(hd + 1) * dh];
            rmsnorm(qs, &lw.q_norm, eps, hbuf);
            qs.copy_from_slice(hbuf);
            rotate(qs);
        }
        for kvh in 0..nkv {
            let ks = &mut st.kc[kbase + kvh * dh..kbase + (kvh + 1) * dh];
            rmsnorm(ks, &lw.k_norm, eps, hbuf);
            ks.copy_from_slice(hbuf);
            rotate(ks);
        }

        // Attention (causal, GQA), pinned per head.
        let scores = &mut scores[..pos + 1];
        for hd in 0..nh {
            let kvh = hd / group;
            let qs = &q[hd * dh..(hd + 1) * dh];
            for (j, sc) in scores.iter_mut().enumerate() {
                let kb = (l * m.max_seq + j) * kv_per + kvh * dh;
                *sc = fdot32(qs, &st.kc[kb..kb + dh]) * inv_sqrt_dh;
            }
            csoftmax(scores);
            let ctx = &mut attnx[hd * dh..(hd + 1) * dh];
            ctx.fill(0.0);
            for (j, &p) in scores.iter().enumerate() {
                let vb = (l * m.max_seq + j) * kv_per + kvh * dh;
                let vrow = &st.vc[vb..vb + dh];
                for d in 0..dh {
                    ctx[d] = p.mul_add(vrow[d], ctx[d]);
                }
            }
        }
        // O-projection + residual (pinned adds).
        fgemv(pool, &lw.wo, attnx, h, nh * dh, xn);
        for c in 0..h {
            x[c] += xn[c];
        }

        // FFN: silu(gate)·up → down, + residual.
        rmsnorm(x, &lw.ln2, eps, xn);
        let f = cfg.intermediate_size;
        fgemv(pool, &lw.w_gate, xn, f, h, gate);
        fgemv(pool, &lw.w_up, xn, f, h, up);
        for r in 0..f {
            gate[r] = csilu(gate[r]) * up[r];
        }
        fgemv(pool, &lw.w_down, gate, h, f, down);
        for c in 0..h {
            x[c] += down[c];
        }
    }
    st.pos += 1;
    if !decide {
        return None;
    }
    // LM head (tied): committed GEMV over the bf16 embedding matrix.
    rmsnorm(x, &m.ln_f, eps, xn);
    fgemv(pool, &m.emb, xn, cfg.vocab_size, h, logits_out);
    // Pinned argmax: ascending scan, strictly greater ⇒ lowest-id ties.
    let mut win = (f32::NEG_INFINITY, 0u32);
    for (v, &s) in logits_out.iter().enumerate() {
        if s > win.0 {
            win = (s, v as u32);
        }
    }
    Some(win.1)
}
