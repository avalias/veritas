//! Offline quantizer + float reference forward (calibration oracle).
//!
//! THE ONLY FLOAT CODE IN THE REPOSITORY (SPEC §6.3 exception): everything
//! it produces is integer artifacts whose bytes become committed state.
//! The runtime never executes a float instruction.
#![allow(clippy::float_arithmetic)] // loud, deliberate, quarantined

use crate::config::QwenConfig;
use crate::tensors::SafeTensors;

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Requant pair: y = sat(rnd(acc · m, SHIFT)) (SPEC §6.1). One global
/// shift keeps the compiler simple; multipliers carry the per-channel
/// information.
pub const SHIFT: u8 = 20;

fn mpair(real_multiplier: f32) -> i32 {
    let m = (real_multiplier * (1u64 << SHIFT) as f32).round();
    assert!(m >= 1.0 && m < i32::MAX as f32, "multiplier out of range: {real_multiplier}");
    m as i32
}

/// Per-output-channel symmetric int8 quantization of a [rows][cols] matrix.
fn quant_per_channel(w: &[f32], rows: usize, cols: usize) -> (Vec<i8>, Vec<f32>) {
    let mut q = vec![0i8; rows * cols];
    let mut scales = vec![0f32; rows];
    for r in 0..rows {
        let row = &w[r * cols..(r + 1) * cols];
        let amax = row.iter().fold(0f32, |a, &v| a.max(v.abs())).max(1e-8);
        let s = amax / 127.0;
        scales[r] = s;
        for c in 0..cols {
            q[r * cols + c] = (row[c] / s).round().clamp(-127.0, 127.0) as i8;
        }
    }
    (q, scales)
}

fn quant_per_tensor(w: &[f32]) -> (Vec<i8>, f32) {
    let amax = w.iter().fold(0f32, |a, &v| a.max(v.abs())).max(1e-8);
    let s = amax / 127.0;
    (w.iter().map(|&v| (v / s).round().clamp(-127.0, 127.0) as i8).collect(), s)
}

// ---------------------------------------------------------------------------
// Float-domain model (reference + calibration)
// ---------------------------------------------------------------------------

pub struct FloatLayer {
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,
    pub w_gate: Vec<f32>,
    pub w_up: Vec<f32>,
    pub w_down: Vec<f32>,
    pub ln1: Vec<f32>,
    pub ln2: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
}

pub struct FloatModel {
    pub cfg: QwenConfig,
    pub emb: Vec<f32>, // [vocab][hidden], tied with the LM head
    pub layers: Vec<FloatLayer>,
    pub ln_f: Vec<f32>,
}

impl FloatModel {
    pub fn load(cfg: &QwenConfig, st: &SafeTensors) -> Self {
        let f = |name: &str| -> Vec<f32> {
            st.bf16_bits(name).iter().map(|&b| bf16_to_f32(b)).collect()
        };
        assert!(cfg.tie_word_embeddings, "tied embeddings assumed (Qwen3-0.6B)");
        let layers = (0..cfg.num_hidden_layers)
            .map(|l| {
                let p = format!("model.layers.{l}");
                FloatLayer {
                    wq: f(&format!("{p}.self_attn.q_proj.weight")),
                    wk: f(&format!("{p}.self_attn.k_proj.weight")),
                    wv: f(&format!("{p}.self_attn.v_proj.weight")),
                    wo: f(&format!("{p}.self_attn.o_proj.weight")),
                    w_gate: f(&format!("{p}.mlp.gate_proj.weight")),
                    w_up: f(&format!("{p}.mlp.up_proj.weight")),
                    w_down: f(&format!("{p}.mlp.down_proj.weight")),
                    ln1: f(&format!("{p}.input_layernorm.weight")),
                    ln2: f(&format!("{p}.post_attention_layernorm.weight")),
                    q_norm: f(&format!("{p}.self_attn.q_norm.weight")),
                    k_norm: f(&format!("{p}.self_attn.k_norm.weight")),
                }
            })
            .collect();
        Self {
            cfg: cfg.clone(),
            emb: f("model.embed_tokens.weight"),
            layers,
            ln_f: f("model.norm.weight"),
        }
    }
}

fn rmsnorm_f(x: &[f32], gamma: &[f32], out: &mut [f32]) {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let r = 1.0 / (ms + 1e-6).sqrt();
    for i in 0..x.len() {
        out[i] = x[i] * r * gamma[i];
    }
}

fn matvec(w: &[f32], x: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
    for r in 0..rows {
        let mut acc = 0f32;
        let row = &w[r * cols..(r + 1) * cols];
        for c in 0..cols {
            acc += row[c] * x[c];
        }
        out[r] = acc;
    }
}

/// Activation magnitude maxima observed during calibration — these become
/// the static per-tensor scales of the integer model.
#[derive(Debug, Default, Clone)]
pub struct Calib {
    pub res: f32,    // residual stream
    pub xn: f32,     // post-RMSNorm
    pub qk: f32,     // q/k after qk-norm + rotary
    pub v: f32,      // v projections
    pub ffn_h: f32,  // silu(gate)·up
}

pub struct FloatState {
    pub kc: Vec<f32>, // [layer][pos][kv_head][head_dim]
    pub vc: Vec<f32>,
    pub pos: usize,
}

impl FloatState {
    pub fn new(cfg: &QwenConfig, max_seq: usize) -> Self {
        let per = cfg.num_key_value_heads * cfg.head_dim;
        Self {
            kc: vec![0.0; cfg.num_hidden_layers * max_seq * per],
            vc: vec![0.0; cfg.num_hidden_layers * max_seq * per],
            pos: 0,
        }
    }
}

/// One decode step of the float reference; returns argmax token id.
/// Rotary uses the SAME integer Q1.14 tables as the integer runtime so the
/// two implementations share positional encodings exactly.
#[allow(clippy::too_many_arguments)]
pub fn float_forward(
    m: &FloatModel,
    st: &mut FloatState,
    max_seq: usize,
    rope_cos: &[i16],
    rope_sin: &[i16],
    token: u32,
    calib: &mut Calib,
) -> u32 {
    let cfg = &m.cfg;
    let (h, dh) = (cfg.hidden_size, cfg.head_dim);
    let (nh, nkv) = (cfg.num_attention_heads, cfg.num_key_value_heads);
    let pos = st.pos;
    let kv_per = nkv * dh;
    let mut x: Vec<f32> = m.emb[token as usize * h..(token as usize + 1) * h].to_vec();
    let mut xn = vec![0f32; h];
    let mut q = vec![0f32; nh * dh];
    let mut attn = vec![0f32; nh * dh];
    let up = |c: &mut f32, vals: &[f32]| {
        for v in vals {
            *c = c.max(v.abs());
        }
    };

    for (l, lw) in m.layers.iter().enumerate() {
        up(&mut calib.res, &x);
        rmsnorm_f(&x, &lw.ln1, &mut xn);
        up(&mut calib.xn, &xn);
        matvec(&lw.wq, &xn, nh * dh, h, &mut q);
        let kbase = (l * max_seq + pos) * kv_per;
        matvec(&lw.wk, &xn, kv_per, h, &mut st.kc[kbase..kbase + kv_per]);
        matvec(&lw.wv, &xn, kv_per, h, &mut st.vc[kbase..kbase + kv_per]);
        // QK-norm (per head over head_dim) then rotary, q and k alike.
        let rot = |vec: &mut [f32]| {
            for p in 0..dh / 2 {
                let (c, s) = (
                    rope_cos[pos * (dh / 2) + p] as f32 / 16384.0,
                    rope_sin[pos * (dh / 2) + p] as f32 / 16384.0,
                );
                let (a, b) = (vec[p], vec[p + dh / 2]); // rotate_half pairing
                vec[p] = a * c - b * s;
                vec[p + dh / 2] = a * s + b * c;
            }
        };
        for hd in 0..nh {
            let qs = &mut q[hd * dh..(hd + 1) * dh];
            let mut tmp = vec![0f32; dh];
            rmsnorm_f(qs, &lw.q_norm, &mut tmp);
            qs.copy_from_slice(&tmp);
            rot(qs);
        }
        for kv in 0..nkv {
            let ks = &mut st.kc[kbase + kv * dh..kbase + (kv + 1) * dh];
            let mut tmp = vec![0f32; dh];
            rmsnorm_f(ks, &lw.k_norm, &mut tmp);
            ks.copy_from_slice(&tmp);
            rot(ks);
        }
        up(&mut calib.qk, &q);
        up(&mut calib.qk, &st.kc[kbase..kbase + kv_per]);
        up(&mut calib.v, &st.vc[kbase..kbase + kv_per]);
        // Attention (GQA: q head hd uses kv head hd / (nh/nkv)).
        let scale = 1.0 / (dh as f32).sqrt();
        for hd in 0..nh {
            let kv = hd / (nh / nkv);
            let qs = &q[hd * dh..(hd + 1) * dh];
            let mut logits = vec![0f32; pos + 1];
            for j in 0..=pos {
                let kb = (l * max_seq + j) * kv_per + kv * dh;
                logits[j] = qs
                    .iter()
                    .zip(&st.kc[kb..kb + dh])
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
                    * scale;
            }
            let mx = logits.iter().fold(f32::MIN, |a, &b| a.max(b));
            let exps: Vec<f32> = logits.iter().map(|&v| (v - mx).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let out = &mut attn[hd * dh..(hd + 1) * dh];
            out.fill(0.0);
            for j in 0..=pos {
                let vb = (l * max_seq + j) * kv_per + kv * dh;
                let w = exps[j] / sum;
                for d in 0..dh {
                    out[d] += w * st.vc[vb + d];
                }
            }
        }
        let mut o = vec![0f32; h];
        matvec(&lw.wo, &attn, h, nh * dh, &mut o);
        for i in 0..h {
            x[i] += o[i];
        }
        up(&mut calib.res, &x);
        // MLP: silu(gate)·up → down.
        rmsnorm_f(&x, &lw.ln2, &mut xn);
        up(&mut calib.xn, &xn);
        let f = cfg.intermediate_size;
        let mut gate = vec![0f32; f];
        let mut upv = vec![0f32; f];
        matvec(&lw.w_gate, &xn, f, h, &mut gate);
        matvec(&lw.w_up, &xn, f, h, &mut upv);
        let mut hb = vec![0f32; f];
        for i in 0..f {
            let g = gate[i];
            hb[i] = (g / (1.0 + (-g).exp())) * upv[i];
        }
        up(&mut calib.ffn_h, &hb);
        let mut down = vec![0f32; h];
        matvec(&lw.w_down, &hb, h, f, &mut down);
        for i in 0..h {
            x[i] += down[i];
        }
    }
    rmsnorm_f(&x.clone(), &m.ln_f, &mut x);
    // Tied LM head; greedy argmax, ties to lowest id.
    let mut best = (f32::MIN, 0u32);
    for v in 0..cfg.vocab_size {
        let row = &m.emb[v * h..(v + 1) * h];
        let s: f32 = row.iter().zip(&x).map(|(a, b)| a * b).sum();
        if s > best.0 {
            best = (s, v as u32);
        }
    }
    st.pos += 1;
    best.1
}

// ---------------------------------------------------------------------------
// Integer artifacts
// ---------------------------------------------------------------------------

pub struct IntLayer {
    pub wq: Vec<i8>,
    pub wk: Vec<i8>,
    pub wv: Vec<i8>,
    pub wo: Vec<i8>,
    pub w_gate: Vec<i8>,
    pub w_up: Vec<i8>,
    pub w_down: Vec<i8>,
    /// Per-channel requant multipliers (global SHIFT), one per output row.
    pub mq: Vec<i32>,
    pub mk: Vec<i32>,
    pub mv: Vec<i32>,
    pub mo: Vec<i32>,
    pub m_gate: Vec<i32>,
    pub m_up: Vec<i32>,
    pub m_down: Vec<i32>,
    /// RMSNorm gains Q12, attention-norm and ffn-norm.
    pub g1: Vec<i32>,
    pub g2: Vec<i32>,
    /// QK-norm gains Q12 (per head_dim).
    pub gq: Vec<i32>,
    pub gk: Vec<i32>,
}

pub struct IntModel {
    pub cfg: QwenConfig,
    pub emb: Vec<i8>, // per-TENSOR scale (tied head needs a common scale)
    pub layers: Vec<IntLayer>,
    pub gf: Vec<i32>,
    /// q·k logit → Q4.11 multiplier (folds s_qk² and 1/√dh).
    pub m_logit: i32,
    /// silu(g)·up (Q4.11×Q4.11) → i8 ffn-h multiplier.
    pub m_h: i32,
    /// ctx (Q0.7·i8 v) → i8 at v-scale is exact >>7; Wo input scale = s_v.
    pub calib: Calib,
}

/// Quantize with calibration-derived static activation scales.
pub fn quantize(m: &FloatModel, calib: &Calib) -> IntModel {
    let cfg = &m.cfg;
    let (h, dh, f) = (cfg.hidden_size, cfg.head_dim, cfg.intermediate_size);
    let nh = cfg.num_attention_heads;
    let nkv = cfg.num_key_value_heads;
    // Static activation scales (value-per-quantum).
    let s_res = calib.res.max(1e-6) / 127.0;
    let s_xn = calib.xn.max(1e-6) / 127.0;
    let s_qk = calib.qk.max(1e-6) / 16384.0; // i16 path
    let s_v = calib.v.max(1e-6) / 127.0;
    let s_h = calib.ffn_h.max(1e-6) / 127.0;
    let gq12 = |g: &[f32]| -> Vec<i32> {
        g.iter().map(|&v| (v * 4096.0).round() as i32).collect()
    };

    let layers = m
        .layers
        .iter()
        .map(|lw| {
            let (wq, sq) = quant_per_channel(&lw.wq, nh * dh, h);
            let (wk, sk) = quant_per_channel(&lw.wk, nkv * dh, h);
            let (wv, sv) = quant_per_channel(&lw.wv, nkv * dh, h);
            let (wo, so) = quant_per_channel(&lw.wo, h, nh * dh);
            let (wg, sg) = quant_per_channel(&lw.w_gate, f, h);
            let (wu, su) = quant_per_channel(&lw.w_up, f, h);
            let (wd, sd) = quant_per_channel(&lw.w_down, h, f);
            // Multipliers: real_out = acc·s_w·s_in; target quantum varies
            // per destination (SPEC §6.1 pattern).
            let ms = |scales: &[f32], s_in: f32, s_out: f32| -> Vec<i32> {
                scales.iter().map(|&sw| mpair(sw * s_in / s_out)).collect()
            };
            IntLayer {
                mq: ms(&sq, s_xn, s_qk),
                mk: ms(&sk, s_xn, s_qk),
                mv: ms(&sv, s_xn, s_v),
                mo: ms(&so, s_v, s_res),
                m_gate: ms(&sg, s_xn, 1.0 / 2048.0), // → Q4.11
                m_up: ms(&su, s_xn, 1.0 / 2048.0),   // → Q4.11
                m_down: ms(&sd, s_h, s_res),
                wq,
                wk,
                wv,
                wo,
                w_gate: wg,
                w_up: wu,
                w_down: wd,
                g1: gq12(&lw.ln1),
                g2: gq12(&lw.ln2),
                gq: gq12(&lw.q_norm),
                gk: gq12(&lw.k_norm),
            }
        })
        .collect();
    let (emb, _s_emb) = quant_per_tensor(&m.emb);
    // logit multiplier: (Σ q·k)·s_qk²/√dh → Q4.11.
    let m_logit = mpair(s_qk * s_qk / (dh as f32).sqrt() * 2048.0);
    // h = silu_q411·up_q411 → product is Q22 of real value; → i8 at s_h.
    let m_h = mpair(1.0 / (2048.0 * 2048.0) / s_h);
    IntModel {
        cfg: cfg.clone(),
        emb,
        layers,
        gf: gq12(&m.ln_f),
        m_logit,
        m_h,
        calib: calib.clone(),
    }
}
