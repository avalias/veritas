//! Memory layout + genesis image (SPEC §7.2's region table, toy edition).
//!
//! Single source of truth for every address the compiler bakes into
//! instructions and the game uses to read outputs. All tensor bases are
//! 64-aligned (DOT lines), all region bases page-aligned where it matters.

use crate::luts;
use crate::model::ToyModel;
use crate::params::*;
use vm::PAGE_SIZE;

/// Memory depth: 2^10 pages = 1 MiB (≈ 533 KiB used; next power of two).
pub const MEM_DEPTH: u8 = 10;

#[derive(Clone, Debug)]
pub struct Layout {
    // -- constants (page 0) --
    pub c_one_i8: u64,  // i8 1 — the MAC8 "add x·1" residual trick
    pub c_d: u64,       // i32 64 — RMSNorm mean divisor
    pub c_2p14: u64,    // i32 16384 — softmax numerator scale
    pub c_neg1: u64,    // i32 −1 — negate-via-multiply (no NEG op)
    // -- scratch --
    pub x: u64,        // [D] i8 residual stream
    pub xn: u64,       // [D] i8 normed activations
    pub q: u64,        // [HEADS][D_HEAD] i16 query
    pub attnx: u64,    // [D] i8 concat head outputs
    pub h_ffn: u64,    // [FFN] i8
    pub logits: u64,   // [VOCAB] i32
    pub att32: u64,    // [MAX_SEQ] i32 attention logits (one head at a time)
    pub e32: u64,      // [MAX_SEQ] i32 exp values
    pub probs: u64,    // [MAX_SEQ] i8 Q0.7 — exactly one DOT8 line
    pub r32: u64,      // i32 rsqrt result
    pub sum: u64,      // i32 exp sum
    pub neg_max: u64,  // i32 −max(logits)
    pub tok: u64,      // u32 current token cell
    // -- io --
    pub input: u64,  // [n u32][ids u32 …], one page
    pub output: u64, // [n u32][ids u32 …], one page
    // -- weights --
    pub emb: u64,
    pub pos: u64,
    pub wq: [u64; LAYERS],
    pub wk: [u64; LAYERS],
    pub wv: [u64; LAYERS],
    pub wo: [u64; LAYERS],
    pub w1: [u64; LAYERS],
    pub w2: [u64; LAYERS],
    pub g1: [u64; LAYERS],
    pub g2: [u64; LAYERS],
    pub gf: u64,
    pub head: u64,
    // -- KV cache --
    /// K[l][h]: [MAX_SEQ][D_HEAD] i16 — each row is one DOT16 line.
    pub kc: [[u64; HEADS]; LAYERS],
    /// Vᵀ[l][h]: [D_HEAD][MAX_SEQ] i8 — each row is one DOT8 line, so the
    /// prob·value contraction reads contiguous lines.
    pub vt: [[u64; HEADS]; LAYERS],
    // -- LUTs (page-aligned, 128 KiB each) --
    pub lut_exp: u64,
    pub lut_rsqrt: u64,
    pub lut_silu: u64,
    pub end: u64,
}

fn align(x: u64, a: u64) -> u64 {
    x.div_ceil(a) * a
}

impl Layout {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        let pg = PAGE_SIZE as u64;
        let mut at = 0u64;
        let mut take = |n: u64, a: u64| {
            at = align(at, a);
            let base = at;
            at += n;
            base
        };
        let l = Layout {
            c_one_i8: take(1, 1),
            c_d: take(4, 4),
            c_2p14: take(4, 4),
            c_neg1: take(4, 4),
            x: take(D as u64, 64),
            xn: take(D as u64, 64),
            q: take((HEADS * D_HEAD * 2) as u64, 64),
            attnx: take(D as u64, 64),
            h_ffn: take(FFN as u64, 64),
            logits: take((VOCAB * 4) as u64, 64),
            att32: take((MAX_SEQ * 4) as u64, 64),
            e32: take((MAX_SEQ * 4) as u64, 64),
            probs: take(MAX_SEQ as u64, 64),
            r32: take(4, 4),
            sum: take(4, 4),
            neg_max: take(4, 4),
            tok: take(4, 4),
            input: take(pg, pg),
            output: take(pg, pg),
            emb: take((VOCAB * D) as u64, 64),
            pos: take((MAX_SEQ * D) as u64, 64),
            wq: [0; LAYERS],
            wk: [0; LAYERS],
            wv: [0; LAYERS],
            wo: [0; LAYERS],
            w1: [0; LAYERS],
            w2: [0; LAYERS],
            g1: [0; LAYERS],
            g2: [0; LAYERS],
            gf: 0,
            head: 0,
            kc: [[0; HEADS]; LAYERS],
            vt: [[0; HEADS]; LAYERS],
            lut_exp: 0,
            lut_rsqrt: 0,
            lut_silu: 0,
            end: 0,
        };
        let mut l = l;
        for i in 0..LAYERS {
            l.wq[i] = take((D * D) as u64, 64);
            l.wk[i] = take((D * D) as u64, 64);
            l.wv[i] = take((D * D) as u64, 64);
            l.wo[i] = take((D * D) as u64, 64);
            l.w1[i] = take((FFN * D) as u64, 64);
            l.w2[i] = take((D * FFN) as u64, 64);
            l.g1[i] = take((D * 4) as u64, 64);
            l.g2[i] = take((D * 4) as u64, 64);
        }
        l.gf = take((D * 4) as u64, 64);
        l.head = take((VOCAB * D) as u64, 64);
        for i in 0..LAYERS {
            for h in 0..HEADS {
                l.kc[i][h] = take((MAX_SEQ * D_HEAD * 2) as u64, 64);
                l.vt[i][h] = take((D_HEAD * MAX_SEQ) as u64, 64);
            }
        }
        l.lut_exp = take(2 * luts::LUT_ENTRIES as u64, pg);
        l.lut_rsqrt = take(2 * luts::LUT_ENTRIES as u64, pg);
        l.lut_silu = take(2 * luts::LUT_ENTRIES as u64, pg);
        l.end = at;
        assert!(l.end <= (1u64 << MEM_DEPTH) * pg, "layout exceeds memory");
        l
    }
}

fn put_i8(img: &mut [u8], at: u64, vals: &[i8]) {
    for (i, v) in vals.iter().enumerate() {
        img[at as usize + i] = *v as u8;
    }
}

fn put_i32(img: &mut [u8], at: u64, vals: &[i32]) {
    for (i, v) in vals.iter().enumerate() {
        img[at as usize + 4 * i..at as usize + 4 * i + 4].copy_from_slice(&v.to_le_bytes());
    }
}

/// Genesis memory image: constants + weights + LUTs + input tokens.
/// Everything else (scratch, KV, output) is zero (SPEC §7.2 ZERO regions).
pub fn genesis_image(lay: &Layout, model: &ToyModel, prompt: &[u32]) -> Vec<u8> {
    assert!(!prompt.is_empty() && prompt.len() <= 255, "prompt size");
    let mut img = vec![0u8; ((1u64 << MEM_DEPTH) * PAGE_SIZE as u64) as usize];
    img[lay.c_one_i8 as usize] = 1;
    put_i32(&mut img, lay.c_d, &[D as i32]);
    put_i32(&mut img, lay.c_2p14, &[16384]);
    put_i32(&mut img, lay.c_neg1, &[-1]);
    put_i8(&mut img, lay.emb, &model.emb);
    put_i8(&mut img, lay.pos, &model.pos);
    for (i, lw) in model.layers.iter().enumerate() {
        put_i8(&mut img, lay.wq[i], &lw.wq);
        put_i8(&mut img, lay.wk[i], &lw.wk);
        put_i8(&mut img, lay.wv[i], &lw.wv);
        put_i8(&mut img, lay.wo[i], &lw.wo);
        put_i8(&mut img, lay.w1[i], &lw.w1);
        put_i8(&mut img, lay.w2[i], &lw.w2);
        put_i32(&mut img, lay.g1[i], &lw.g1);
        put_i32(&mut img, lay.g2[i], &lw.g2);
    }
    put_i32(&mut img, lay.gf, &model.gf);
    put_i8(&mut img, lay.head, &model.head);
    let exp = luts::to_bytes(&luts::gen_exp());
    let rsq = luts::to_bytes(&luts::gen_rsqrt());
    let sil = luts::to_bytes(&luts::gen_silu());
    img[lay.lut_exp as usize..lay.lut_exp as usize + exp.len()].copy_from_slice(&exp);
    img[lay.lut_rsqrt as usize..lay.lut_rsqrt as usize + rsq.len()].copy_from_slice(&rsq);
    img[lay.lut_silu as usize..lay.lut_silu as usize + sil.len()].copy_from_slice(&sil);
    // Input region: [n u32][token ids u32 …] (SPEC §7.2).
    put_i32(&mut img, lay.input, &[prompt.len() as i32]);
    for (i, t) in prompt.iter().enumerate() {
        let at = lay.input as usize + 4 + 4 * i;
        img[at..at + 4].copy_from_slice(&t.to_le_bytes());
    }
    img
}
