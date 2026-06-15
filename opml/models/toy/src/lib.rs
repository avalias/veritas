//! Phase 1 toy model (SPEC brief: 2-layer char-level transformer, d=64,
//! 2 heads, vocab 96, INT8, per-tensor scales).
//!
//! Weights are random-but-fixed (seeded xorshift) — the brief explicitly
//! allows this: training quality is irrelevant, determinism is the point.
//! All requantization scales are the documented constants in `scales`.
//!
//! Notably: the LUT tables are generated with **pure integer arithmetic**
//! (i128 fixed point, integer Newton sqrt — no libm, no floats anywhere).
//! This is stricter than SPEC §6.3 requires (floats were allowed offline)
//! and removes the cross-platform-libm risk from the golden hashes
//! entirely.

pub mod forward;
pub mod layout;
pub mod luts;
pub mod model;

/// Architecture constants (brief-pinned).
pub mod params {
    pub const VOCAB: usize = 96; // printable ASCII 32..127, token = byte − 32
    pub const D: usize = 64; // d_model
    pub const HEADS: usize = 2;
    pub const D_HEAD: usize = 32; // 32 i16 lanes = one 64-byte DOT16 line
    pub const LAYERS: usize = 2;
    pub const FFN: usize = 256; // 4×d_model, four DOT8 lines per row
    pub const MAX_SEQ: usize = 64; // == one DOT8 line of attention probs
}

/// Per-tensor requantization scales — every (multiplier, shift) the
/// compiler bakes in, with the range proof for each (SPEC §6.1, §6.4).
/// Multipliers are all 1 for the toy (pure shifts); Qwen will use real
/// gemmlowp pairs.
pub mod scales {
    /// i8·i8 dot over d=64: |acc| ≤ 64·127·127 < 2^20 → i8 via >>13.
    pub const S_PROJ_I8: u8 = 13;
    /// Same accumulator → i16 (Q/K path): 2^20 → ±2^14 via >>6.
    pub const S_QK_I16: u8 = 6;
    /// q·k over 32 i16 lanes: |acc| < 32·2^14·2^14 = 2^33 → Q4.11 (±2^14)
    /// via >>19 (≈ ±4.0 logits into the exp LUT domain).
    pub const S_LOGIT_Q411: u8 = 19;
    /// exp output is Q1.14; prob = e·2^14/sum → Q0.14 → Q0.7 via >>7.
    pub const S_PROB_Q07: u8 = 7;
    /// ctx = Σ prob(Q0.7)·v(i8): Σprob ≈ 2^7 → value scale ≈ v via >>7.
    pub const S_CTX_I8: u8 = 7;
    /// FFN up-proj acc (2^20) → Q4.11 via >>6 (±2^14 into SiLU LUT).
    pub const S_FFN_Q411: u8 = 6;
    /// SiLU out Q4.11 (±2^14) → i8 via >>8.
    pub const S_SILU_I8: u8 = 8;
    /// FFN down-proj over 256 lanes: |acc| ≤ 256·127·127 < 2^22 → >>15.
    pub const S_FFN_DOWN_I8: u8 = 15;
    /// RMSNorm: out = x·rsqrt(Q2.14)·γ(≈Q6) >> 14 ⇒ ≈ (x/rms)·γ_q6.
    pub const S_NORM: u8 = 14;
}
