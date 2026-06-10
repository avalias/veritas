//! Toy model weights: random-but-fixed INT8 (brief-sanctioned; determinism
//! is the point, training quality is not).

use crate::params::*;
use vm::fixtures::XorShift64;

/// Weight seed — part of the model's identity; changing it is a new model.
pub const WEIGHT_SEED: u64 = 0x70FD_01CE_2026; // arbitrary fixed constant

pub struct LayerWeights {
    /// Wq/Wk/Wv: [D][D] i8, row-major, rows are DOT8 lines. The first
    /// D_HEAD·h rows of each belong to head h (head-major row blocks).
    pub wq: Vec<i8>,
    pub wk: Vec<i8>,
    pub wv: Vec<i8>,
    /// Wo: [D][D] i8 — attn-concat → residual.
    pub wo: Vec<i8>,
    /// W1: [FFN][D] up-projection.
    pub w1: Vec<i8>,
    /// W2: [D][FFN] down-projection (rows are 4 DOT8 lines).
    pub w2: Vec<i8>,
    /// RMSNorm gains, ≈Q6 (values 32..96 ⇒ 0.5..1.5): attn-norm, ffn-norm.
    pub g1: Vec<i32>,
    pub g2: Vec<i32>,
}

pub struct ToyModel {
    /// Token embedding [VOCAB][D] i8.
    pub emb: Vec<i8>,
    /// Learned absolute position embedding [MAX_SEQ][D] i8 (toy stand-in
    /// for rotary, which arrives with Qwen in Phase 3).
    pub pos: Vec<i8>,
    pub layers: Vec<LayerWeights>,
    /// Final RMSNorm gain.
    pub gf: Vec<i32>,
    /// LM head [VOCAB][D] i8.
    pub head: Vec<i8>,
}

fn i8_vec(rng: &mut XorShift64, n: usize) -> Vec<i8> {
    (0..n).map(|_| rng.next_u64() as u8 as i8).collect()
}

fn gamma_vec(rng: &mut XorShift64, n: usize) -> Vec<i32> {
    // 32..=95 ≈ 0.5..1.5 in Q6 — keeps RMSNorm output in healthy i8 range.
    (0..n).map(|_| 32 + (rng.next_u64() % 64) as i32).collect()
}

impl ToyModel {
    pub fn generate(seed: u64) -> Self {
        let mut rng = XorShift64::new(seed);
        let layers = (0..LAYERS)
            .map(|_| LayerWeights {
                wq: i8_vec(&mut rng, D * D),
                wk: i8_vec(&mut rng, D * D),
                wv: i8_vec(&mut rng, D * D),
                wo: i8_vec(&mut rng, D * D),
                w1: i8_vec(&mut rng, FFN * D),
                w2: i8_vec(&mut rng, D * FFN),
                g1: gamma_vec(&mut rng, D),
                g2: gamma_vec(&mut rng, D),
            })
            .collect();
        Self {
            emb: i8_vec(&mut rng, VOCAB * D),
            pos: i8_vec(&mut rng, MAX_SEQ * D),
            layers,
            gf: gamma_vec(&mut rng, D),
            head: i8_vec(&mut rng, VOCAB * D),
        }
    }
}

/// Char-level tokenizer: printable ASCII 32..=127 → 0..=95.
/// Total and deterministic: out-of-range bytes clamp into the range
/// (documented quirk of the toy; the real tokenizer arrives in Phase 3).
pub fn tokenize(text: &str) -> Vec<u32> {
    text.bytes()
        .map(|b| (b.clamp(32, 127) - 32) as u32)
        .collect()
}

pub fn detokenize(tokens: &[u32]) -> String {
    tokens
        .iter()
        .map(|&t| ((t.min(95) as u8) + 32) as char)
        .collect()
}
