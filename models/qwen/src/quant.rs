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
    // Underflow ⇒ effectively-zero channel (near-zero weight row): clamp to
    // 1 — harmless, the products are zero anyway. Overflow is a real bug.
    assert!(m < i32::MAX as f32, "multiplier overflow: {real_multiplier}");
    (m as i32).max(1)
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

pub fn rmsnorm_f(x: &[f32], gamma: &[f32], out: &mut [f32]) {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let r = 1.0 / (ms + 1e-6).sqrt();
    for i in 0..x.len() {
        out[i] = x[i] * r * gamma[i];
    }
}

pub fn matvec(w: &[f32], x: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
    for r in 0..rows {
        let mut acc = 0f32;
        let row = &w[r * cols..(r + 1) * cols];
        for c in 0..cols {
            acc += row[c] * x[c];
        }
        out[r] = acc;
    }
}

/// Activation magnitude maxima observed during calibration, PER LAYER —
/// Qwen3's dynamic range (0.05 embeddings → 6000+ attention-sink residual
/// outliers) kills any single static scale; per-layer-site scales + an i16
/// residual carrier with boundary rescales keep everything static (the
/// protocol requirement) while preserving signal.
#[derive(Debug, Clone)]
pub struct Calib {
    /// Residual magnitude entering each layer; [layers] entries plus the
    /// final-norm entry at index `layers`.
    pub res: Vec<f32>,
    pub xn1: Vec<f32>, // post-attn-norm, per layer
    pub xn2: Vec<f32>, // post-ffn-norm, per layer
    pub ms1: Vec<f32>, // mean-square of residual at norm1 input, per layer
    pub ms2: Vec<f32>, // mean-square at norm2 input, per layer
    pub msf: f32,      // mean-square at final-norm input
    pub xnf: f32,      // post final norm
    pub qk: Vec<f32>,  // q/k after qk-norm+rotary, per layer
    pub qk_pre: Vec<f32>, // q/k straight out of the projections (pre-norm!)
    pub v: Vec<f32>,   // v projections, per layer
    pub gate: Vec<f32>, // gate projection output (pre-silu), per layer
    pub up: Vec<f32>,   // up projection output, per layer
    pub ffn_h: Vec<f32>, // silu(gate)·up, per layer
}

impl Calib {
    pub fn new(layers: usize) -> Self {
        Self {
            res: vec![0.0; layers + 1],
            xn1: vec![0.0; layers],
            xn2: vec![0.0; layers],
            ms1: vec![0.0; layers],
            ms2: vec![0.0; layers],
            msf: 0.0,
            xnf: 0.0,
            qk: vec![0.0; layers],
            qk_pre: vec![0.0; layers],
            v: vec![0.0; layers],
            gate: vec![0.0; layers],
            up: vec![0.0; layers],
            ffn_h: vec![0.0; layers],
        }
    }
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

    let msq = |v: &[f32]| v.iter().map(|a| a * a).sum::<f32>() / v.len() as f32;
    for (l, lw) in m.layers.iter().enumerate() {
        up(&mut calib.res[l], &x);
        calib.ms1[l] = calib.ms1[l].max(msq(&x));
        rmsnorm_f(&x, &lw.ln1, &mut xn);
        up(&mut calib.xn1[l], &xn);
        matvec(&lw.wq, &xn, nh * dh, h, &mut q);
        let kbase = (l * max_seq + pos) * kv_per;
        matvec(&lw.wk, &xn, kv_per, h, &mut st.kc[kbase..kbase + kv_per]);
        matvec(&lw.wv, &xn, kv_per, h, &mut st.vc[kbase..kbase + kv_per]);
        up(&mut calib.qk_pre[l], &q);
        up(&mut calib.qk_pre[l], &st.kc[kbase..kbase + kv_per]);
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
        up(&mut calib.qk[l], &q);
        up(&mut calib.qk[l], &st.kc[kbase..kbase + kv_per]);
        up(&mut calib.v[l], &st.vc[kbase..kbase + kv_per]);
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
        up(&mut calib.res[l], &x);
        calib.ms2[l] = calib.ms2[l].max(msq(&x));
        // MLP: silu(gate)·up → down.
        rmsnorm_f(&x, &lw.ln2, &mut xn);
        up(&mut calib.xn2[l], &xn);
        let f = cfg.intermediate_size;
        let mut gate = vec![0f32; f];
        let mut upv = vec![0f32; f];
        matvec(&lw.w_gate, &xn, f, h, &mut gate);
        matvec(&lw.w_up, &xn, f, h, &mut upv);
        up(&mut calib.gate[l], &gate);
        up(&mut calib.up[l], &upv);
        let mut hb = vec![0f32; f];
        for i in 0..f {
            let g = gate[i];
            hb[i] = (g / (1.0 + (-g).exp())) * upv[i];
        }
        up(&mut calib.ffn_h[l], &hb);
        let mut down = vec![0f32; h];
        matvec(&lw.w_down, &hb, h, f, &mut down);
        for i in 0..h {
            x[i] += down[i];
        }
    }
    up(&mut calib.res[m.layers.len()], &x);
    calib.msf = calib.msf.max(msq(&x));
    rmsnorm_f(&x.clone(), &m.ln_f, &mut x);
    up(&mut calib.xnf, &x);
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

/// A folded RMSNorm site: out = sat(rnd(x · r · gamma_m[c], shift)).
/// `pre_shift` is the static range reduction applied to the mean-square
/// before the rsqrt LUT (0 for the i8 path, 14 for the i16 q/k path,
/// SPEC §6.4 + the rsqrt domain note).
#[derive(Clone, Debug)]
pub struct NormSite {
    pub gamma_m: Vec<i32>,
    pub shift: u8,
    /// Mean-square range reduction before the rsqrt LUT. For i16 inputs:
    /// applied to the SUM (pre_shift = 16). For the i32 residual carrier:
    /// applied PER ELEMENT before squaring (calibrated per layer so the
    /// reduced mean-square lands in the LUT domain).
    pub pre_shift: u8,
    /// True ⇒ per-element prescale (i32 carrier sites).
    pub elem_pre: bool,
}

/// Fold a vector of real per-channel factors into (M[i32], shift) with the
/// largest |M| near 2^28 — healthy precision, no i64 overflow downstream.
fn fold_factors(factors: &[f32]) -> (Vec<i32>, u8) {
    let amax = factors.iter().fold(0f32, |a, &v| a.max(v.abs())).max(1e-30);
    let mut shift = 0u8;
    while shift < 62 && (amax * ((1u64 << (shift + 1)) as f32)) < ((1u64 << 28) as f32) {
        shift += 1;
    }
    let m: Vec<i32> = factors
        .iter()
        .map(|&v| {
            let q = (v * (1u64 << shift) as f32).round();
            assert!(q.abs() < i32::MAX as f32, "fold overflow");
            q as i32
        })
        .collect();
    (m, shift)
}

fn norm_site(gamma: &[f32], real_per_r_unit: f32, pre_shift: u8, elem_pre: bool) -> NormSite {
    let factors: Vec<f32> = gamma.iter().map(|&g| g * real_per_r_unit).collect();
    let (gamma_m, shift) = fold_factors(&factors);
    NormSite { gamma_m, shift, pre_shift, elem_pre }
}

/// Norm over the i32 carrier: per-element prescale k chosen from the
/// calibrated mean-square so ss/h lands inside the rsqrt LUT domain
/// (≤ 2^14, with 4 bits of safety). r = lut[ms>>?] ≈ 2^(14+k)/√ms_q, so
/// the folded factor is γ/(2^(14+k)·s_xn).
fn carrier_norm_site(gamma: &[f32], ms_real: f32, s_res: f32, s_xn: f32) -> NormSite {
    let ms_q = (ms_real.max(1e-12) / (s_res * s_res)) as f64; // in quanta²
    let mut k = 0u8;
    while (ms_q / 4f64.powi(k as i32)) > 1024.0 * 16.0 && k < 31 {
        k += 1; // each k halves elements ⇒ quarters the mean-square
    }
    let factor = 1.0 / ((1u64 << (14 + k as u32)) as f32 * s_xn);
    let factors: Vec<f32> = gamma.iter().map(|&g| g * factor).collect();
    let (gamma_m, shift) = fold_factors(&factors);
    NormSite { gamma_m, shift, pre_shift: k, elem_pre: true }
}

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
    pub norm1: NormSite,
    pub norm2: NormSite,
    pub qnorm: NormSite,
    pub knorm: NormSite,
    /// q·k logit → Q4.11 (folds s_qk[l]² and 1/√dh).
    pub m_logit: i32,
    /// gate (s_g i16) → Q4.11 for the sigmoid's exp input.
    pub m_sig: i32,
    /// (g·σ(g) >> 14) · up → i16 at s_h[l] (folds s_g·s_up).
    pub m_h: i32,
}

pub struct IntModel {
    pub cfg: QwenConfig,
    pub emb: Vec<i8>, // per-TENSOR scale (tied head needs a common scale)
    /// Embedding-row → residual-scale requant (s_emb / s_res[0]).
    pub m_emb: i32,
    pub layers: Vec<IntLayer>,
    pub norm_f: NormSite,
    pub calib: Calib,
}

/// Quantize with PER-LAYER calibration-derived static activation scales
/// and an i16 residual carrier (real value = quant · scale):
///   residual x: i16, s_res[l], rescaled at layer exits (m_res)
///   post-norm xn: i8, s_xn{1,2}[l]      q/k: i16, s_qk[l]
///   v/ctx: i8, s_v[l] (ctx = probs·v >> 7 stays at s_v)
///   ffn hidden: i8, s_h[l]              logits: raw acc (argmax-safe)
/// All norm sites read i16-squared sums range-reduced by 2^16 before the
/// rsqrt LUT ⇒ r ≈ 2^22/√ms ⇒ folded factor = γ/(2^22·s_out).
pub fn quantize(m: &FloatModel, calib: &Calib) -> IntModel {
    let cfg = &m.cfg;
    let (h, dh, f) = (cfg.hidden_size, cfg.head_dim, cfg.intermediate_size);
    let nh = cfg.num_attention_heads;
    let nkv = cfg.num_key_value_heads;
    let nl = cfg.num_hidden_layers;
    // ONE global i32 residual scale: max magnitude maps to 2^29 (2 bits of
    // headroom). Qwen3's full 0.05→6400 dynamic range fits with ~16 bits
    // of resolution for embedding-sized values — no rescales, no clipping.
    let res_max = calib.res.iter().fold(1e-6f32, |a, &b| a.max(b));
    let s_res_g = res_max / (1u64 << 29) as f32;
    let s_res: Vec<f32> = calib.res.iter().map(|_| s_res_g).collect();
    let s_xn1: Vec<f32> = calib.xn1.iter().map(|&v| v.max(1e-6) / 32767.0).collect();
    let s_xn2: Vec<f32> = calib.xn2.iter().map(|&v| v.max(1e-6) / 32767.0).collect();
    let s_qk: Vec<f32> = calib.qk.iter().map(|&v| v.max(1e-6) / 16384.0).collect();
    let s_qk_pre: Vec<f32> = calib.qk_pre.iter().map(|&v| v.max(1e-6) / 32767.0).collect();
    let s_gate: Vec<f32> = calib.gate.iter().map(|&v| v.max(1e-6) / 32767.0).collect();
    let s_up: Vec<f32> = calib.up.iter().map(|&v| v.max(1e-6) / 32767.0).collect();
    let s_v: Vec<f32> = calib.v.iter().map(|&v| v.max(1e-6) / 127.0).collect();
    let s_h: Vec<f32> = calib.ffn_h.iter().map(|&v| v.max(1e-6) / 32767.0).collect();
    let s_xnf = calib.xnf.max(1e-6) / 32767.0;

    let layers = (0..nl)
        .map(|l| {
            let lw = &m.layers[l];
            let (wq, sq) = quant_per_channel(&lw.wq, nh * dh, h);
            let (wk, sk) = quant_per_channel(&lw.wk, nkv * dh, h);
            let (wv, sv) = quant_per_channel(&lw.wv, nkv * dh, h);
            let (wo, so) = quant_per_channel(&lw.wo, h, nh * dh);
            let (wg, sg) = quant_per_channel(&lw.w_gate, f, h);
            let (wu, su) = quant_per_channel(&lw.w_up, f, h);
            let (wd, sd) = quant_per_channel(&lw.w_down, h, f);
            let ms = |scales: &[f32], s_in: f32, s_out: f32| -> Vec<i32> {
                scales.iter().map(|&sw| mpair(sw * s_in / s_out)).collect()
            };
            IntLayer {
                // Pre-norm targets! Post-norm scale would saturate the
                // raw projections 10-100x before qk-norm runs.
                mq: ms(&sq, s_xn1[l], s_qk_pre[l]),
                mk: ms(&sk, s_xn1[l], s_qk_pre[l]),
                mv: ms(&sv, s_xn1[l], s_v[l]),
                mo: ms(&so, s_v[l], s_res[l]),
                m_gate: ms(&sg, s_xn2[l], s_gate[l]), // i16, per-layer scale
                m_up: ms(&su, s_xn2[l], s_up[l]),     // i16, per-layer scale
                m_down: ms(&sd, s_h[l], s_res[l + 1]),
                wq,
                wk,
                wv,
                wo,
                w_gate: wg,
                w_up: wu,
                w_down: wd,
                norm1: carrier_norm_site(&lw.ln1, calib.ms1[l], s_res_g, s_xn1[l]),
                norm2: carrier_norm_site(&lw.ln2, calib.ms2[l], s_res_g, s_xn2[l]),
                qnorm: norm_site(&lw.q_norm, 1.0 / ((1u64 << 22) as f32 * s_qk[l]), 16, false),
                knorm: norm_site(&lw.k_norm, 1.0 / ((1u64 << 22) as f32 * s_qk[l]), 16, false),
                m_logit: mpair(s_qk[l] * s_qk[l] / (dh as f32).sqrt() * 2048.0),
                m_sig: mpair(s_gate[l] * 2048.0), // g(s_g) → Q4.11 for σ
                m_h: mpair(s_gate[l] * s_up[l] / s_h[l]),
            }
        })
        .collect();
    let (emb, s_emb) = quant_per_tensor(&m.emb);
    IntModel {
        cfg: cfg.clone(),
        emb,
        m_emb: mpair(s_emb / s_res[0]),
        layers,
        norm_f: carrier_norm_site(&m.ln_f, calib.msf, s_res_g, s_xnf),
        calib: calib.clone(),
    }
}
