//! Genesis image builder: lay the quantized model, LUTs, rotary tables,
//! constants, and input tokens into the committed 1 GiB memory (SPEC §7.2).

use crate::layout::{QwenLayout, MAX_SEQ, MEM_DEPTH};
use crate::quant::IntModel;
use crate::trig;
use toy_model::luts;
use vm::PAGE_SIZE;

pub struct Tables {
    pub exp: Vec<i16>,
    pub rsqrt: Vec<i16>,
    pub silu: Vec<i16>,
    pub cos: Vec<i16>,
    pub sin: Vec<i16>,
    pub nsin: Vec<i16>,
}

impl Tables {
    pub fn generate(rope_theta: u64, head_dim: usize) -> Self {
        let (cos, sin, nsin) = trig::rope_tables(rope_theta, head_dim / 2, MAX_SEQ);
        Self {
            exp: luts::gen_exp(),
            rsqrt: luts::gen_rsqrt(),
            silu: luts::gen_silu(),
            cos,
            sin,
            nsin,
        }
    }
}

fn put_i8(img: &mut [u8], at: u64, v: &[i8]) {
    let at = at as usize;
    for (i, x) in v.iter().enumerate() {
        img[at + i] = *x as u8;
    }
}

fn put_i16(img: &mut [u8], at: u64, v: &[i16]) {
    let at = at as usize;
    for (i, x) in v.iter().enumerate() {
        img[at + 2 * i..at + 2 * i + 2].copy_from_slice(&x.to_le_bytes());
    }
}

fn put_i32(img: &mut [u8], at: u64, v: &[i32]) {
    let at = at as usize;
    for (i, x) in v.iter().enumerate() {
        img[at + 4 * i..at + 4 * i + 4].copy_from_slice(&x.to_le_bytes());
    }
}

pub fn genesis_image(
    lay: &QwenLayout,
    im: &IntModel,
    tables: &Tables,
    prompt: &[u32],
) -> Vec<u8> {
    assert!(!prompt.is_empty() && prompt.len() < MAX_SEQ);
    let mut img = vec![0u8; (1usize << MEM_DEPTH) * PAGE_SIZE];
    // Constants.
    img[lay.c_one_i8 as usize] = 1;
    put_i32(&mut img, lay.c_h, &[im.cfg.hidden_size as i32]);
    put_i32(&mut img, lay.c_dh, &[im.cfg.head_dim as i32]);
    put_i32(&mut img, lay.c_2p14, &[16384]);
    put_i32(&mut img, lay.c_neg1, &[-1]);
    put_i32(&mut img, lay.c_2p30, &[1 << 30]);
    // c_m_logit / c_m_h are superseded by per-layer cells; left zero.
    put_i32(&mut img, lay.m_emb_arr, &im.m_emb);
    put_i32(&mut img, lay.c_i32min, &[i32::MIN]);
    // Weights.
    put_i8(&mut img, lay.emb, &im.emb);
    put_i8(&mut img, lay.head_w, &im.head_w);
    put_i32(&mut img, lay.m_head, &im.m_head.0);
    put_i32(&mut img, lay.gf, &im.norm_f.gamma_m);
    for (l, lw) in im.layers.iter().enumerate() {
        let a = &lay.layers[l];
        put_i8(&mut img, a.wq, &lw.wq);
        put_i8(&mut img, a.wk, &lw.wk);
        put_i8(&mut img, a.wv, &lw.wv);
        put_i8(&mut img, a.wo, &lw.wo);
        put_i8(&mut img, a.w_gate, &lw.w_gate);
        put_i8(&mut img, a.w_up, &lw.w_up);
        put_i8(&mut img, a.w_down, &lw.w_down);
        put_i32(&mut img, a.mq, &lw.mq.0);
        put_i32(&mut img, a.mk, &lw.mk.0);
        put_i32(&mut img, a.mv, &lw.mv.0);
        put_i32(&mut img, a.mo, &lw.mo.0);
        put_i32(&mut img, a.m_gate, &lw.m_gate.0);
        put_i32(&mut img, a.m_up, &lw.m_up.0);
        put_i32(&mut img, a.m_down, &lw.m_down.0);
        put_i32(&mut img, a.g1, &lw.norm1.gamma_m);
        put_i32(&mut img, a.g2, &lw.norm2.gamma_m);
        put_i32(&mut img, a.gq, &lw.qnorm.gamma_m);
        put_i32(&mut img, a.gk, &lw.knorm.gamma_m);
        put_i32(&mut img, a.m_logit_c, &[lw.m_logit]);
        put_i32(&mut img, a.m_sig_c, &[lw.m_sig]);
        put_i32(&mut img, a.m_h_arr, &lw.m_h);
    }
    // Tables.
    put_i16(&mut img, lay.rope_cos, &tables.cos);
    put_i16(&mut img, lay.rope_sin, &tables.sin);
    put_i16(&mut img, lay.rope_nsin, &tables.nsin);
    put_i16(&mut img, lay.lut_exp, &tables.exp);
    put_i16(&mut img, lay.lut_rsqrt, &tables.rsqrt);
    put_i16(&mut img, lay.lut_silu, &tables.silu);
    // Input region [n][ids…] (SPEC §7.2).
    put_i32(&mut img, lay.input, &[prompt.len() as i32]);
    for (i, t) in prompt.iter().enumerate() {
        let at = lay.input as usize + 4 + 4 * i;
        img[at..at + 4].copy_from_slice(&t.to_le_bytes());
    }
    img
}
