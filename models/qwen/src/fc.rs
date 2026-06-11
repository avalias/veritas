//! FW-6 committed-float native runtime: fmodel's forward (hardware floats,
//! proven bit-equal to softfloat over 8M fuzz cases per op) executed with
//! every committed value written into the FLayout image — the write-for-
//! write twin of the float compiler (compiler/src/fqwen.rs). C-14-float
//! asserts its boundary memory roots equal the VM oracle's exactly.
//!
//! The mirroring discipline (proven on the integer path): every cell the
//! compiled program writes — state arrays AND scratch (facc/t1..t4/flag/
//! fm/fsum/fr/v_cell/tok/saved_max/logit_buf) — is written here with the
//! same final value, in the same order.
#![allow(clippy::float_arithmetic)] // FW-6: floats are the committed semantics

use crate::flayout::{FLayout, FMAX_SEQ};
use crate::fmath::cexp;
use crate::fmodel::FModel;
use kernels::fkernels::bf16_to_f32;
use toy_model::forward::FlatMem;

fn rf(mem: &FlatMem, at: u64) -> f32 {
    f32::from_bits(mem.r32i(at) as u32)
}

fn wf(mem: &mut FlatMem, at: u64, v: f32) {
    mem.w32(at, v.to_bits());
}

/// Committed CEXP macro twin (compiler CEXP): reads t1, leaves the result
/// in t2, staging in t3; mirrors every scratch write. Returns exp value.
fn cexp_committed(mem: &mut FlatMem, lay: &FLayout) -> f32 {
    let mut x = rf(mem, lay.t1);
    // clamp lo
    let f1 = u32::from(-87.0f32 > x);
    mem.w32(lay.flag, f1);
    if f1 == 1 {
        x = -87.0;
        wf(mem, lay.t1, x);
    }
    // clamp hi
    let f2 = u32::from(x > 88.0f32);
    mem.w32(lay.flag, f2);
    if f2 == 1 {
        x = 88.0;
        wf(mem, lay.t1, x);
    }
    // kf (VM writes t3 thrice: mul, add, floor — finals overwrite)
    let t3a = x * core::f32::consts::LOG2_E;
    wf(mem, lay.t3, t3a);
    let t3b = t3a + 0.5;
    wf(mem, lay.t3, t3b);
    let kf = t3b.floor();
    wf(mem, lay.t3, kf);
    // r staged in t1 (two fused steps)
    let r1 = kf.mul_add(-0.693_359_4_f32, x);
    wf(mem, lay.t1, r1);
    let r = kf.mul_add(2.121_944_4e-4_f32, r1);
    wf(mem, lay.t1, r);
    // two_k via integer ops into t3
    let ki = kf as i32;
    let two_k_bits = (((ki + 127) as u32) << 23) as u32;
    mem.w32(lay.t3, two_k_bits);
    // poly in t2 (Horner, 6 mul/add pairs after the seed)
    let mut p = 1.0f32 / 720.0;
    wf(mem, lay.t2, p);
    for c in [1.0f32 / 120.0, 1.0 / 24.0, 1.0 / 6.0, 0.5, 1.0, 1.0] {
        p *= r;
        wf(mem, lay.t2, p);
        p += c;
        wf(mem, lay.t2, p);
    }
    let out = p * f32::from_bits(two_k_bits);
    wf(mem, lay.t2, out);
    // The macro must equal the committed scalar cexp (clamp already applied,
    // so cexp's own clamp is a no-op here).
    debug_assert_eq!(out.to_bits(), cexp(x).to_bits());
    out
}

/// The committed row dot: facc ← 0; 16 (or cols/64) committed block dots
/// accumulate sequentially — identical to fkernels::fdot_row.
fn gemv_row(mem: &FlatMem, w_base: u64, row: u64, cols: u64, x: &[f32]) -> f32 {
    let off = (w_base + row * cols * 2) as usize;
    let wrow = mem.slice(off as u64, (cols * 2) as usize);
    let w16: Vec<u16> =
        wrow.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    kernels::fkernels::fdot_row(&w16, x)
}

fn rmsnorm_committed(
    mem: &mut FlatMem,
    lay: &FLayout,
    src: u64,
    gamma: &[f32],
    dst: u64,
    n: usize,
    eps: f32,
) {
    // facc ← 0; sequential ffma chain
    let mut ss = 0f32;
    for c in 0..n {
        let v = rf(mem, src + 4 * c as u64);
        ss = v.mul_add(v, ss);
    }
    wf(mem, lay.facc, ss);
    let mean = ss / n as f32;
    wf(mem, lay.t1, mean);
    let me = mean + eps;
    wf(mem, lay.t1, me);
    let sq = me.sqrt();
    wf(mem, lay.t1, sq);
    let r = 1.0 / sq;
    wf(mem, lay.fr, r);
    for c in 0..n {
        let v = rf(mem, src + 4 * c as u64);
        let t = v * r;
        wf(mem, dst + 4 * c as u64, t);
        wf(mem, dst + 4 * c as u64, t * gamma[c]);
    }
}

/// One committed position. Mirrors compiler/src/fqwen.rs block-for-block.
#[allow(clippy::too_many_lines)]
pub fn fc_position(
    m: &FModel,
    lay: &FLayout,
    mem: &mut FlatMem,
    pos: usize,
    tok: u32,
    decide: bool,
) -> Option<u32> {
    let cfg = &m.cfg;
    let (h, dh, f) = (cfg.hidden_size, cfg.head_dim, cfg.intermediate_size);
    let (nh, nkv) = (cfg.num_attention_heads, cfg.num_key_value_heads);
    let group = nh / nkv;
    let half = dh / 2;
    let eps = 1.0 / (cfg.rms_norm_eps_recip as f32);
    let isqdh = 1.0 / (dh as f32).sqrt();

    // E: embedding — widen(emb bf16) into x cells (ST32 truncation of
    // sext16(bits)·65536 == bits<<16 == exact widening).
    for c in 0..h {
        let bits = m.emb[tok as usize * h + c];
        mem.w32(lay.x + 4 * c as u64, (bits as u32) << 16);
    }

    let read_vec = |mem: &FlatMem, base: u64, n: usize| -> Vec<f32> {
        (0..n).map(|i| f32::from_bits(mem.r32i(base + 4 * i as u64) as u32)).collect()
    };

    for (l, a) in lay.layers.iter().enumerate() {
        let lw = &m.layers[l];
        // N1
        rmsnorm_committed(mem, lay, lay.x, &lw.ln1, lay.xn, h, eps);
        let xn = read_vec(mem, lay.xn, h);
        // G1..G3: q/k/v GEMVs (committed row chain; facc final = last row)
        for r in 0..nh * dh {
            let v = gemv_row(mem, a.wq, r as u64, h as u64, &xn);
            wf(mem, lay.facc, v);
            wf(mem, lay.q + 4 * r as u64, v);
        }
        let krow = a.kc + (pos * nkv * dh * 4) as u64;
        for r in 0..nkv * dh {
            let v = gemv_row(mem, a.wk, r as u64, h as u64, &xn);
            wf(mem, lay.facc, v);
            wf(mem, krow + 4 * r as u64, v);
        }
        let vrow = a.vc + (pos * nkv * dh * 4) as u64;
        for r in 0..nkv * dh {
            let v = gemv_row(mem, a.wv, r as u64, h as u64, &xn);
            wf(mem, lay.facc, v);
            wf(mem, vrow + 4 * r as u64, v);
        }
        // QK-norm + rope (q heads, then k heads)
        let rope = |mem: &mut FlatMem, base: u64, lay: &FLayout| {
            for p2 in 0..half {
                let aa = rf(mem, base + 4 * p2 as u64);
                let bb = rf(mem, base + 4 * (p2 + half) as u64);
                let c = m.rope_cos[pos * half + p2];
                let s = m.rope_sin[pos * half + p2];
                let t = bb * s;
                wf(mem, lay.t4, t);
                let nt = t * -1.0;
                wf(mem, lay.t4, nt);
                wf(mem, lay.t1, nt);
                let na = aa.mul_add(c, nt);
                wf(mem, lay.t1, na);
                let t2v = aa * s;
                wf(mem, lay.t4, t2v);
                let nb = bb.mul_add(c, t2v);
                wf(mem, lay.t4, nb);
                wf(mem, base + 4 * (p2 + half) as u64, nb);
                wf(mem, base + 4 * p2 as u64, na);
            }
        };
        for hd in 0..nh {
            let base = lay.q + 4 * (hd * dh) as u64;
            // qknorm in place (gamma = q_norm), n = dh
            let qs = read_vec(mem, base, dh);
            let mut ss = 0f32;
            for &v in &qs {
                ss = v.mul_add(v, ss);
            }
            wf(mem, lay.facc, ss);
            let mean = ss / dh as f32;
            wf(mem, lay.t1, mean);
            let me = mean + eps;
            wf(mem, lay.t1, me);
            let sq = me.sqrt();
            wf(mem, lay.t1, sq);
            let r = 1.0 / sq;
            wf(mem, lay.fr, r);
            for (d, &v) in qs.iter().enumerate() {
                let t = v * r;
                wf(mem, base + 4 * d as u64, t);
                wf(mem, base + 4 * d as u64, t * lw.q_norm[d]);
            }
            rope(mem, base, lay);
        }
        for kvh in 0..nkv {
            let base = krow + 4 * (kvh * dh) as u64;
            let ks = read_vec(mem, base, dh);
            let mut ss = 0f32;
            for &v in &ks {
                ss = v.mul_add(v, ss);
            }
            wf(mem, lay.facc, ss);
            let mean = ss / dh as f32;
            wf(mem, lay.t1, mean);
            let me = mean + eps;
            wf(mem, lay.t1, me);
            let sq = me.sqrt();
            wf(mem, lay.t1, sq);
            let r = 1.0 / sq;
            wf(mem, lay.fr, r);
            for (d, &v) in ks.iter().enumerate() {
                let t = v * r;
                wf(mem, base + 4 * d as u64, t);
                wf(mem, base + 4 * d as u64, t * lw.k_norm[d]);
            }
            rope(mem, base, lay);
        }
        // ATT: per (kvh, g)
        for kvh in 0..nkv {
            for g in 0..group {
                let hd = kvh * group + g;
                let qbase = lay.q + 4 * (hd * dh) as u64;
                let qs = read_vec(mem, qbase, dh);
                // scores
                for j in 0..=pos {
                    let kb = a.kc + ((j * nkv * dh + kvh * dh) * 4) as u64;
                    let ks = read_vec(mem, kb, dh);
                    let mut acc = 0f32;
                    for d in 0..dh {
                        acc = qs[d].mul_add(ks[d], acc);
                    }
                    wf(mem, lay.facc, acc);
                    wf(mem, lay.scores + 4 * j as u64, acc * isqdh);
                }
                // max scan (FGT/JEQ twin)
                let s0 = rf(mem, lay.scores);
                wf(mem, lay.fm, s0);
                let mut mx = s0;
                for j in 0..=pos {
                    let sj = rf(mem, lay.scores + 4 * j as u64);
                    let fl = u32::from(sj > mx);
                    mem.w32(lay.flag, fl);
                    if fl == 1 {
                        mx = sj;
                        wf(mem, lay.fm, mx);
                    }
                }
                // exps in place + sum
                mem.w32(lay.fsum, 0);
                let mut sum = 0f32;
                for j in 0..=pos {
                    let sj = rf(mem, lay.scores + 4 * j as u64);
                    let nm = mx * -1.0;
                    wf(mem, lay.t1, nm);
                    let arg = sj + nm;
                    wf(mem, lay.t1, arg);
                    let e = cexp_committed(mem, lay);
                    wf(mem, lay.scores + 4 * j as u64, e);
                    sum += e;
                    wf(mem, lay.fsum, sum);
                }
                // normalize
                for j in 0..=pos {
                    let e = rf(mem, lay.scores + 4 * j as u64);
                    wf(mem, lay.scores + 4 * j as u64, e / sum);
                }
                // ctx into attnx
                for d in 0..dh {
                    let cell = lay.attnx + 4 * (hd * dh + d) as u64;
                    mem.w32(cell, 0);
                    let mut acc = 0f32;
                    for j in 0..=pos {
                        let p = rf(mem, lay.scores + 4 * j as u64);
                        let vv =
                            rf(mem, a.vc + ((j * nkv * dh + kvh * dh + d) * 4) as u64);
                        acc = p.mul_add(vv, acc);
                        wf(mem, cell, acc);
                    }
                }
            }
        }
        // O: gemv(wo, attnx) + residual into x
        let ax = read_vec(mem, lay.attnx, nh * dh);
        for r in 0..h {
            let v = gemv_row(mem, a.wo, r as u64, (nh * dh) as u64, &ax);
            wf(mem, lay.facc, v);
            let xr = rf(mem, lay.x + 4 * r as u64);
            wf(mem, lay.x + 4 * r as u64, xr + v);
        }
        // N2 + FFN
        rmsnorm_committed(mem, lay, lay.x, &lw.ln2, lay.xn, h, eps);
        let xn2 = read_vec(mem, lay.xn, h);
        for r in 0..f {
            let v = gemv_row(mem, a.w_gate, r as u64, h as u64, &xn2);
            wf(mem, lay.facc, v);
            wf(mem, lay.gate + 4 * r as u64, v);
        }
        for r in 0..f {
            let v = gemv_row(mem, a.w_up, r as u64, h as u64, &xn2);
            wf(mem, lay.facc, v);
            wf(mem, lay.up + 4 * r as u64, v);
        }
        // SI: silu per row
        for r in 0..f {
            let g = rf(mem, lay.gate + 4 * r as u64);
            let u = rf(mem, lay.up + 4 * r as u64);
            let ng = g * -1.0;
            wf(mem, lay.t1, ng);
            let e = cexp_committed(mem, lay);
            let den = e + 1.0;
            wf(mem, lay.t2, den);
            let sig = g / den;
            wf(mem, lay.t3, sig);
            wf(mem, lay.h_ffn + 4 * r as u64, sig * u);
        }
        // FD: down + residual
        let hf = read_vec(mem, lay.h_ffn, f);
        for r in 0..h {
            let v = gemv_row(mem, a.w_down, r as u64, f as u64, &hf);
            wf(mem, lay.facc, v);
            let xr = rf(mem, lay.x + 4 * r as u64);
            wf(mem, lay.x + 4 * r as u64, xr + v);
        }
    }

    if !decide {
        return None;
    }
    // NF + streaming head with FGT argmax + cycled logit page.
    rmsnorm_committed(mem, lay, lay.x, &m.ln_f, lay.xn, h, eps);
    let xnf = read_vec(mem, lay.xn, h);
    mem.w32(lay.saved_max, 0xFF80_0000); // −inf
    mem.w32(lay.v_cell, 0);
    let vocab = cfg.vocab_size;
    let mut best = f32::NEG_INFINITY;
    let mut win = 0u32;
    for v in 0..vocab {
        let logit = gemv_row(mem, lay.emb, v as u64, h as u64, &xnf);
        wf(mem, lay.facc, logit);
        wf(mem, lay.logit_buf + 4 * (v % 256) as u64, logit);
        let fl = u32::from(logit > best);
        mem.w32(lay.flag, fl);
        if fl == 1 {
            best = logit;
            wf(mem, lay.saved_max, best);
            mem.w32(lay.tok, v as u32);
            win = v as u32;
        }
        mem.w32(lay.v_cell, v as u32 + 1);
    }
    Some(win)
}
