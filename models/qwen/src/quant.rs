//! Offline quantizer + float reference forward (calibration oracle).
//!
//! THE ONLY FLOAT CODE IN THE REPOSITORY (SPEC §6.3 exception): everything
//! it produces is integer artifacts whose bytes become committed state.
//! The runtime never executes a float instruction.
#![allow(clippy::float_arithmetic)] // loud, deliberate, quarantined
#![allow(clippy::needless_range_loop)] // float reference favors index clarity

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

/// Per-(row, 64-col-block) symmetric int8 quantization — llama Q8_0's
/// block structure. Scales fold into the per-(row,block) M tables at zero
/// runtime cost.
fn quant_per_rowblock(w: &[f32], rows: usize, cols: usize) -> (Vec<i8>, Vec<f32>) {
    let blocks = cols / 64;
    let mut q = vec![0i8; rows * cols];
    let mut scales = vec![0f32; rows * blocks];
    for r in 0..rows {
        for b in 0..blocks {
            let off = r * cols + b * 64;
            let blk = &w[off..off + 64];
            let amax = blk.iter().fold(0f32, |a, &v| a.max(v.abs())).max(1e-8);
            let s = amax / 127.0;
            scales[r * blocks + b] = s;
            for c in 0..64 {
                q[off + c] = (blk[c] / s).round().clamp(-127.0, 127.0) as i8;
            }
        }
    }
    (q, scales)
}

/// Per-(row,block) M from BLOCKED weight scales (rows×blocks) and blocked
/// activation scales — same normalized-shift discipline as m_table.
fn m_table_bw(sw_rb: &[f32], s_blocks: &[f32], s_out: f32) -> (Vec<i32>, u8) {
    let blocks = s_blocks.len();
    let mut fmax = 1e-30f32;
    for (i, &swv) in sw_rb.iter().enumerate() {
        fmax = fmax.max(swv * s_blocks[i % blocks] / s_out);
    }
    let mut shift = 0u8;
    while shift < 62 && (fmax * ((1u64 << (shift + 1)) as f32)) < ((1u64 << 23) as f32) {
        shift += 1;
    }
    let out: Vec<i32> = sw_rb
        .iter()
        .enumerate()
        .map(|(i, &swv)| {
            let m = (swv * s_blocks[i % blocks] / s_out * ((1u64 << shift) as f32)).round();
            assert!(m < ((1u64 << 25) as f32), "blocked multiplier overflow");
            (m as i32).max(1)
        })
        .collect();
    (out, shift)
}

/// Per-output-channel symmetric int8 quantization of a [rows][cols] matrix.
#[allow(dead_code)] // kept: the non-blocked reference path
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

pub fn upch(acc: &mut Vec<f32>, vals: &[f32]) {
    if acc.is_empty() {
        acc.resize(vals.len(), 0.0);
    }
    for (a, v) in acc.iter_mut().zip(vals) {
        *a = a.max(v.abs());
    }
}

/// Range-free log2 magnitude histogram for PERCENTILE calibration. Max-based
/// activation scales provably degrade with more data (a single rare outlier
/// blows out the scale, coarsening resolution for the bulk — measured: 3×
/// more max-calibration data took PPL 421→1145). A high percentile clips the
/// outlier tail and reclaims that range. Bins are 16/octave over 2^-30..2^20.
pub const HIST_BPO: i32 = 16;
pub const HIST_OFF: i32 = 30 * HIST_BPO; // bin 0 ↔ 2^-30
pub const HIST_BINS: usize = 50 * HIST_BPO as usize; // up to 2^20

#[derive(Debug, Clone)]
pub struct Hist {
    counts: Vec<u32>,
    total: u64,
}

impl Default for Hist {
    fn default() -> Self {
        Self { counts: vec![0; HIST_BINS], total: 0 }
    }
}

impl Hist {
    pub fn observe(&mut self, vals: &[f32]) {
        for &v in vals {
            let a = v.abs();
            let bin = if a < 1e-9 {
                0
            } else {
                ((a.log2() * HIST_BPO as f32).round() as i32 + HIST_OFF)
                    .clamp(0, HIST_BINS as i32 - 1) as usize
            };
            self.counts[bin] += 1;
            self.total += 1;
        }
    }

    /// Magnitude threshold at fraction `p` (e.g. 0.999) of observed values.
    pub fn percentile(&self, p: f32) -> f32 {
        if self.total == 0 {
            return 1e-6;
        }
        let target = (p as f64 * self.total as f64) as u64;
        let mut cum = 0u64;
        for (b, &c) in self.counts.iter().enumerate() {
            cum += c as u64;
            if cum >= target {
                return 2f32.powf((b as i32 - HIST_OFF) as f32 / HIST_BPO as f32);
            }
        }
        2f32.powf((HIST_BINS as i32 - 1 - HIST_OFF) as f32 / HIST_BPO as f32)
    }
}

/// Per-site activation-scale source: max (legacy) or a calibrated percentile.
/// `QCAL_PCTL=0.999` (env) switches the activation scales to percentile.
pub fn pctl_env() -> Option<f32> {
    std::env::var("QCAL_PCTL").ok().and_then(|s| s.parse::<f32>().ok()).filter(|&p| p > 0.0 && p < 1.0)
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
    /// Per-channel |max| of matmul inputs (SmoothQuant equalization):
    pub xn1_ch: Vec<Vec<f32>>, // [layer][hidden] — q/k/v input
    pub xn2_ch: Vec<Vec<f32>>, // [layer][hidden] — gate/up input
    pub h_ch: Vec<Vec<f32>>,   // [layer][ffn]    — down input
    pub xnf_ch: Vec<f32>,      // [hidden]        — LM-head input
    pub xnf: f32,      // post final norm
    pub qk: Vec<f32>,  // q/k after qk-norm+rotary, per layer
    pub qk_pre: Vec<f32>, // q/k straight out of the projections (pre-norm!)
    pub v: Vec<f32>,   // v projections, per layer
    pub gate: Vec<f32>, // gate projection output (pre-silu), per layer
    pub up: Vec<f32>,   // up projection output, per layer
    pub ffn_h: Vec<f32>, // silu(gate)·up, per layer
    /// Percentile-calibration histograms, parallel to the scalar maxima
    /// above (only the sites that set activation scales).
    pub res_h: Vec<Hist>,
    pub qk_h: Vec<Hist>,
    pub qk_pre_h: Vec<Hist>,
    pub v_h: Vec<Hist>,
    pub gate_h: Vec<Hist>,
    pub up_h: Vec<Hist>,
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
            xn1_ch: vec![vec![]; layers],
            xn2_ch: vec![vec![]; layers],
            h_ch: vec![vec![]; layers],
            xnf_ch: vec![],
            xnf: 0.0,
            qk: vec![0.0; layers],
            qk_pre: vec![0.0; layers],
            v: vec![0.0; layers],
            gate: vec![0.0; layers],
            up: vec![0.0; layers],
            ffn_h: vec![0.0; layers],
            res_h: vec![Hist::default(); layers + 1],
            qk_h: vec![Hist::default(); layers],
            qk_pre_h: vec![Hist::default(); layers],
            v_h: vec![Hist::default(); layers],
            gate_h: vec![Hist::default(); layers],
            up_h: vec![Hist::default(); layers],
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
    let logits = float_forward_impl(m, st, max_seq, rope_cos, rope_sin, token, calib);
    let mut best = (f32::MIN, 0u32);
    for (v, &s) in logits.iter().enumerate() {
        if s > best.0 {
            best = (s, v as u32);
        }
    }
    best.1
}

/// Like `float_forward` but returns the FULL logits vector (eval harness).
#[allow(clippy::too_many_arguments)]
pub fn float_forward_logits(
    m: &FloatModel,
    st: &mut FloatState,
    max_seq: usize,
    rope_cos: &[i16],
    rope_sin: &[i16],
    token: u32,
    calib: &mut Calib,
) -> Vec<f32> {
    // Run the standard forward to advance KV, then recompute the head from
    // scratch is wasteful; instead temporarily replicate: easiest correct
    // path — call float_forward then recompute logits via the last hidden
    // state is unavailable. So: duplicate the body minus argmax.
    // (Pragmatic eval-only duplication, kept adjacent to the original.)
    let _ = float_forward; // keep the pair visually linked
    float_forward_impl(m, st, max_seq, rope_cos, rope_sin, token, calib)
}

#[allow(clippy::too_many_arguments)]
fn float_forward_impl(
    m: &FloatModel,
    st: &mut FloatState,
    max_seq: usize,
    rope_cos: &[i16],
    rope_sin: &[i16],
    token: u32,
    calib: &mut Calib,
) -> Vec<f32> {
    // identical to float_forward through the final norm…
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
        calib.res_h[l].observe(&x);
        calib.ms1[l] = calib.ms1[l].max(msq(&x));
        rmsnorm_f(&x, &lw.ln1, &mut xn);
        up(&mut calib.xn1[l], &xn);
        upch(&mut calib.xn1_ch[l], &xn);
        matvec(&lw.wq, &xn, nh * dh, h, &mut q);
        let kbase = (l * max_seq + pos) * kv_per;
        matvec(&lw.wk, &xn, kv_per, h, &mut st.kc[kbase..kbase + kv_per]);
        matvec(&lw.wv, &xn, kv_per, h, &mut st.vc[kbase..kbase + kv_per]);
        up(&mut calib.qk_pre[l], &q);
        up(&mut calib.qk_pre[l], &st.kc[kbase..kbase + kv_per]);
        calib.qk_pre_h[l].observe(&q);
        calib.qk_pre_h[l].observe(&st.kc[kbase..kbase + kv_per]);
        let rot = |vecv: &mut [f32]| {
            for p2 in 0..dh / 2 {
                let (c, s) = (
                    rope_cos[pos * (dh / 2) + p2] as f32 / 16384.0,
                    rope_sin[pos * (dh / 2) + p2] as f32 / 16384.0,
                );
                let (a, b) = (vecv[p2], vecv[p2 + dh / 2]);
                vecv[p2] = a * c - b * s;
                vecv[p2 + dh / 2] = a * s + b * c;
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
        calib.qk_h[l].observe(&q);
        calib.qk_h[l].observe(&st.kc[kbase..kbase + kv_per]);
        calib.v_h[l].observe(&st.vc[kbase..kbase + kv_per]);
        let scale = 1.0 / (dh as f32).sqrt();
        for hd in 0..nh {
            let kv = hd / (nh / nkv);
            let qs = &q[hd * dh..(hd + 1) * dh];
            let mut logits = vec![0f32; pos + 1];
            for j in 0..=pos {
                let kb = (l * max_seq + j) * kv_per + kv * dh;
                logits[j] = qs.iter().zip(&st.kc[kb..kb + dh]).map(|(a, b)| a * b).sum::<f32>() * scale;
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
        calib.res_h[l].observe(&x);
        calib.ms2[l] = calib.ms2[l].max(msq(&x));
        rmsnorm_f(&x, &lw.ln2, &mut xn);
        up(&mut calib.xn2[l], &xn);
        upch(&mut calib.xn2_ch[l], &xn);
        let f = cfg.intermediate_size;
        let mut gate = vec![0f32; f];
        let mut upv = vec![0f32; f];
        matvec(&lw.w_gate, &xn, f, h, &mut gate);
        matvec(&lw.w_up, &xn, f, h, &mut upv);
        up(&mut calib.gate[l], &gate);
        up(&mut calib.up[l], &upv);
        calib.gate_h[l].observe(&gate);
        calib.up_h[l].observe(&upv);
        let mut hb = vec![0f32; f];
        for i in 0..f {
            let g = gate[i];
            hb[i] = (g / (1.0 + (-g).exp())) * upv[i];
        }
        up(&mut calib.ffn_h[l], &hb);
        upch(&mut calib.h_ch[l], &hb);
        let mut down = vec![0f32; h];
        matvec(&lw.w_down, &hb, h, f, &mut down);
        for i in 0..h {
            x[i] += down[i];
        }
    }
    up(&mut calib.res[m.layers.len()], &x);
    calib.res_h[m.layers.len()].observe(&x);
    calib.msf = calib.msf.max(msq(&x));
    rmsnorm_f(&x.clone(), &m.ln_f, &mut x);
    up(&mut calib.xnf, &x);
    upch(&mut calib.xnf_ch, &x);
    let mut logits = vec![0f32; cfg.vocab_size];
    for v in 0..cfg.vocab_size {
        let row = &m.emb[v * h..(v + 1) * h];
        logits[v] = row.iter().zip(&x).map(|(a, b)| a * b).sum();
    }
    st.pos += 1;
    logits
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
fn carrier_norm_site(gamma: &[f32], ms_real: f32, s_res: f32, s_xn_ch: &[f32]) -> NormSite {
    let ms_q = (ms_real.max(1e-12) / (s_res * s_res)) as f64; // in quanta²
    let mut k = 0u8;
    while (ms_q / 4f64.powi(k as i32)) > 1024.0 * 16.0 && k < 31 {
        k += 1; // each k halves elements ⇒ quarters the mean-square
    }
    let base = 1.0 / (1u64 << (14 + k as u32)) as f32;
    let factors: Vec<f32> =
        gamma.iter().zip(s_xn_ch).map(|(&g, &sj)| g * base / sj).collect();
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
    /// Per-(row,block) requant multipliers + their per-matrix shifts.
    pub mq: (Vec<i32>, u8),
    pub mk: (Vec<i32>, u8),
    pub mv: (Vec<i32>, u8),
    pub mo: (Vec<i32>, u8),
    pub m_gate: (Vec<i32>, u8),
    pub m_up: (Vec<i32>, u8),
    pub m_down: (Vec<i32>, u8),
    pub norm1: NormSite,
    pub norm2: NormSite,
    pub qnorm: NormSite,
    pub knorm: NormSite,
    /// q·k logit → Q4.11 (folds s_qk[l]² and 1/√dh).
    pub m_logit: i32,
    /// gate (s_g i16) → Q4.11 for the sigmoid's exp input.
    pub m_sig: i32,
    /// (g·σ(g) >> 14) · up → i16, PER CHANNEL (SmoothQuant fold on h).
    pub m_h: Vec<i32>,
}

pub struct IntModel {
    pub cfg: QwenConfig,
    /// EVAL-ONLY metadata: the common logit quantum (real units/quantum).
    pub s_logit_eval: f32,
    pub s_emb_eval: f32,
    pub s_xnf_eval: f32,
    /// LM head, PER-ROW quantized (per-tensor murdered logit precision:
    /// typical weights got ~8 levels against the matrix absmax). m_head
    /// restores a COMMON logit scale per row — ordering stays valid.
    pub head_w: Vec<i8>,
    pub m_head: (Vec<i32>, u8),
    pub emb: Vec<i8>, // per-TENSOR scale (tied head needs a common scale)
    /// Embedding-row → residual-scale requant, PER CHANNEL (undoes the
    /// head-side SmoothQuant column scaling on the tied matrix).
    pub m_emb: Vec<i32>,
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
/// SmoothQuant vector for one consumer family: s_j = (act_j^α / wcol_j^(1−α)),
/// clamped — outlier activation channels migrate into the (per-channel-
/// quantized) weights. Xiao et al., arXiv 2211.10438.
fn smooth_vec(act_ch: &[f32], wcols: &[&[f32]], rows_cols: &[(usize, usize)], alpha: f32) -> Vec<f32> {
    let n = act_ch.len();
    let mut wmax = vec![1e-8f32; n];
    for (w, &(rows, cols)) in wcols.iter().zip(rows_cols) {
        assert_eq!(cols, n);
        for r in 0..rows {
            for j in 0..n {
                wmax[j] = wmax[j].max(w[r * cols + j].abs());
            }
        }
    }
    (0..n)
        .map(|j| {
            let a = act_ch[j].max(1e-6);
            (a.powf(alpha) / wmax[j].powf(1.0 - alpha)).clamp(0.05, 20.0)
        })
        .collect()
}

fn scale_cols(w: &[f32], rows: usize, cols: usize, s: &[f32]) -> Vec<f32> {
    let mut out = w.to_vec();
    for r in 0..rows {
        for j in 0..cols {
            out[r * cols + j] *= s[j];
        }
    }
    out
}

fn div_vec(a: &[f32], s: &[f32]) -> Vec<f32> {
    a.iter().zip(s).map(|(x, d)| x / d).collect()
}

const ALPHA: f32 = 0.5;

/// Calibration headroom: maxima are measured on a finite calibration set;
/// eval-time activations exceeding them would SATURATE (the probe showed
/// top-logit compression — the clipping fingerprint). One extra ~bit of
/// range kills the clipping for a small resolution cost.
const HEADROOM: f32 = 1.25;

/// Per-64-channel-block activation scales from smoothed per-channel maxima.
fn block_scales(act_sm_ch: &[f32]) -> Vec<f32> {
    act_sm_ch
        .chunks(64)
        .map(|blk| {
            let mx = blk.iter().fold(1e-6f32, |a, &b| a.max(b));
            mx * HEADROOM / 32767.0
        })
        .collect()
}

/// Per-channel view of block scales (channel j → its block's scale).
fn per_channel(blocks: &[f32], n: usize) -> Vec<f32> {
    (0..n).map(|j| blocks[j / 64]).collect()
}

/// Per-(row, block) multipliers with a per-matrix normalized shift: the
/// largest M lands near 2^23 (so 48-block i64 accumulation stays exact),
/// and the matching shift is returned for the final round-half-even.
#[allow(dead_code)] // kept: per-row-uniform variant of m_table_bw
fn m_table(sw: &[f32], s_blocks: &[f32], s_out: f32) -> (Vec<i32>, u8) {
    let mut fmax = 1e-30f32;
    for &r in sw {
        for &b in s_blocks {
            fmax = fmax.max(r * b / s_out);
        }
    }
    let mut shift = 0u8;
    while shift < 62 && (fmax * ((1u64 << (shift + 1)) as f32)) < ((1u64 << 23) as f32) {
        shift += 1;
    }
    let mut out = Vec::with_capacity(sw.len() * s_blocks.len());
    for &r in sw {
        for &b in s_blocks {
            let m = (r * b / s_out * ((1u64 << shift) as f32)).round();
            assert!(m < ((1u64 << 25) as f32), "blocked multiplier overflow");
            out.push((m as i32).max(1));
        }
    }
    (out, shift)
}

pub fn quantize(m: &FloatModel, calib: &Calib) -> IntModel {
    let cfg = &m.cfg;
    let (h, dh, f) = (cfg.hidden_size, cfg.head_dim, cfg.intermediate_size);
    let nh = cfg.num_attention_heads;
    let nkv = cfg.num_key_value_heads;
    let nl = cfg.num_hidden_layers;
    // ONE global i32 residual scale: max magnitude maps to 2^29 (2 bits of
    // headroom). Qwen3's full 0.05→6400 dynamic range fits with ~16 bits
    // of resolution for embedding-sized values — no rescales, no clipping.
    // Activation-range source: max (default) or QCAL_PCTL percentile.
    // MEASURED (night-3): percentile clipping HURTS these sites —
    // {0.9999: 1220, 0.999: 1031} PPL vs max 421. The large q/k/gate/up/v
    // activations are SIGNAL (attention logits and SwiGLU gates ride the
    // tail), not range-wasting noise; clipping them via a tighter scale
    // saturates real information. Kept env-gated for reproducibility; max
    // stays the default. The real gap is logit RESOLUTION (bits), not
    // clipping — see the quality ladder.
    let pctl = pctl_env();
    let amax_or_pctl = |max_v: f32, h: &Hist| -> f32 {
        match pctl {
            Some(p) => h.percentile(p).max(1e-6),
            None => max_v.max(1e-6),
        }
    };
    let res_max = calib.res.iter().fold(1e-6f32, |a, &b| a.max(b));
    let s_res_g = res_max / (1u64 << 29) as f32; // i32: 4x headroom built in
    let s_res: Vec<f32> = calib.res.iter().map(|_| s_res_g).collect();
    // per-tensor xn scales superseded by smoothed per-layer values below

    let s_qk: Vec<f32> =
        calib.qk.iter().zip(&calib.qk_h).map(|(&v, h)| amax_or_pctl(v, h) * HEADROOM / 16384.0).collect();
    let s_qk_pre: Vec<f32> =
        calib.qk_pre.iter().zip(&calib.qk_pre_h).map(|(&v, h)| amax_or_pctl(v, h) * HEADROOM / 32767.0).collect();
    let s_gate: Vec<f32> =
        calib.gate.iter().zip(&calib.gate_h).map(|(&v, h)| amax_or_pctl(v, h) * HEADROOM / 32767.0).collect();
    let s_up: Vec<f32> =
        calib.up.iter().zip(&calib.up_h).map(|(&v, h)| amax_or_pctl(v, h) * HEADROOM / 32767.0).collect();
    let s_v: Vec<f32> =
        calib.v.iter().zip(&calib.v_h).map(|(&v, h)| amax_or_pctl(v, h) * HEADROOM / 32767.0).collect();



    let layers = (0..nl)
        .map(|l| {
            let lw = &m.layers[l];
            // SmoothQuant: equalize matmul inputs into the weights.
            let s1 = smooth_vec(
                &calib.xn1_ch[l],
                &[&lw.wq, &lw.wk, &lw.wv],
                &[(nh * dh, h), (nkv * dh, h), (nkv * dh, h)],
                ALPHA,
            );
            let s2 = smooth_vec(
                &calib.xn2_ch[l],
                &[&lw.w_gate, &lw.w_up],
                &[(f, h), (f, h)],
                ALPHA,
            );
            let sh = smooth_vec(&calib.h_ch[l], &[&lw.w_down], &[(h, f)], ALPHA);
            let (wq, sq) = quant_per_rowblock(&scale_cols(&lw.wq, nh * dh, h, &s1), nh * dh, h);
            let (wk, sk) = quant_per_rowblock(&scale_cols(&lw.wk, nkv * dh, h, &s1), nkv * dh, h);
            let (wv, sv) = quant_per_rowblock(&scale_cols(&lw.wv, nkv * dh, h, &s1), nkv * dh, h);
            let (wo, so) = quant_per_rowblock(&lw.wo, h, nh * dh);
            let (wg, sg) = quant_per_rowblock(&scale_cols(&lw.w_gate, f, h, &s2), f, h);
            let (wu, su) = quant_per_rowblock(&scale_cols(&lw.w_up, f, h, &s2), f, h);
            let (wd, sd) = quant_per_rowblock(&scale_cols(&lw.w_down, h, f, &sh), h, f);
            // Per-BLOCK activation scales from smoothed per-channel maxima
            // (composes analytically — max-based calibration).
            let xn1_b = block_scales(&div_vec(&calib.xn1_ch[l], &s1));
            let xn2_b = block_scales(&div_vec(&calib.xn2_ch[l], &s2));
            let h_b = block_scales(&div_vec(&calib.h_ch[l], &sh));
            let xn1_ch_s = per_channel(&xn1_b, h);
            let xn2_ch_s = per_channel(&xn2_b, h);
            let h_ch_s = per_channel(&h_b, f);
            IntLayer {
                // Pre-norm targets! Post-norm scale would saturate the
                // raw projections 10-100x before qk-norm runs. All inputs
                // carry per-block scales → per-(row,block) M tables.
                mq: m_table_bw(&sq, &xn1_b, s_qk_pre[l]),
                mk: m_table_bw(&sk, &xn1_b, s_qk_pre[l]),
                mv: m_table_bw(&sv, &xn1_b, s_v[l]),
                mo: {
                    // attnx input is per-tensor (s_v): constant block scales.
                    let attnx_b = vec![s_v[l]; (nh * dh) / 64];
                    m_table_bw(&so, &attnx_b, s_res[l])
                },
                m_gate: m_table_bw(&sg, &xn2_b, s_gate[l]),
                m_up: m_table_bw(&su, &xn2_b, s_up[l]),
                m_down: m_table_bw(&sd, &h_b, s_res[l + 1]),
                wq,
                wk,
                wv,
                wo,
                w_gate: wg,
                w_up: wu,
                w_down: wd,
                norm1: {
                    let g: Vec<f32> = lw.ln1.iter().zip(&s1).map(|(g, s)| g / s).collect();
                    carrier_norm_site(&g, calib.ms1[l], s_res_g, &xn1_ch_s)
                },
                norm2: {
                    let g: Vec<f32> = lw.ln2.iter().zip(&s2).map(|(g, s)| g / s).collect();
                    carrier_norm_site(&g, calib.ms2[l], s_res_g, &xn2_ch_s)
                },
                qnorm: norm_site(&lw.q_norm, 1.0 / ((1u64 << 22) as f32 * s_qk[l]), 16, false),
                knorm: norm_site(&lw.k_norm, 1.0 / ((1u64 << 22) as f32 * s_qk[l]), 16, false),
                m_logit: mpair(s_qk[l] * s_qk[l] / (dh as f32).sqrt() * 2048.0),
                m_sig: mpair(s_gate[l] * 2048.0), // g(s_g) → Q4.11 for σ
                m_h: sh
                    .iter()
                    .zip(&h_ch_s)
                    .map(|(shj, hsj)| mpair(s_gate[l] * s_up[l] / (hsj * shj)))
                    .collect(),
            }
        })
        .collect();
    // Head/embedding (tied): smooth the final-norm output into the emb
    // columns; the embedding READ path undoes it per channel via m_emb.
    let sf = smooth_vec(&calib.xnf_ch, &[&m.emb], &[(cfg.vocab_size, h)], ALPHA);
    let smoothed_emb = scale_cols(&m.emb, cfg.vocab_size, h, &sf);
    let (emb, s_embs) = quant_per_tensor(&smoothed_emb);
    // Head role: PER-ROW quantization + per-row multipliers restoring one
    // common logit scale s_L (chosen so max multiplier ≈ 16·2^SHIFT).
    let (head_w, s_rows) = quant_per_rowblock(&smoothed_emb, cfg.vocab_size, h);
    let xnf_b = block_scales(&div_vec(&calib.xnf_ch, &sf));
    let xnf_ch_s = per_channel(&xnf_b, h);
    let s_xnf_t = xnf_b.iter().fold(1e-12f32, |a, &b| a.max(b)); // eval info
    let s_row_max = s_rows.iter().fold(1e-12f32, |a, &b| a.max(b));
    let s_b_max = xnf_b.iter().fold(1e-12f32, |a, &b| a.max(b));
    let s_logit = s_row_max * s_b_max / 16.0;
    let m_head = m_table_bw(&s_rows, &xnf_b, s_logit);
    let gf: Vec<f32> = m.ln_f.iter().zip(&sf).map(|(g, s)| g / s).collect();
    IntModel {
        cfg: cfg.clone(),
        s_logit_eval: s_logit,
        s_emb_eval: s_embs,
        s_xnf_eval: s_xnf_t,
        head_w,
        m_head,
        emb,
        m_emb: sf.iter().map(|sj| mpair(s_embs / (sj * s_res_g))).collect(),
        layers,
        norm_f: carrier_norm_site(&gf, calib.msf, s_res_g, &xnf_ch_s),
        calib: calib.clone(),
    }
}
