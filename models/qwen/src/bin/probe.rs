//! Divergence probe: float vs integer forward in LOCKSTEP, per layer, per
//! site, position 0 only. Debug tool — floats allowed (offline).
#![allow(clippy::float_arithmetic)]

use qwen::config::QwenConfig;
use qwen::forward::Native;
use qwen::image::{genesis_image, Tables};
use qwen::layout::QwenLayout;
use qwen::quant::{matvec, quantize, rmsnorm_f, Calib, FloatModel, FloatState};
use qwen::tensors::SafeTensors;
use toy_model::forward::FlatMem;

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/artifacts");

/// Relative L2 error between a float vector and a dequantized int vector.
fn rel(f: &[f32], qv: &[f32]) -> f32 {
    let mut num = 0f64;
    let mut den = 0f64;
    for (a, b) in f.iter().zip(qv) {
        num += ((a - b) as f64).powi(2);
        den += (*a as f64).powi(2);
    }
    ((num / den.max(1e-12)).sqrt()) as f32
}

fn main() {
    let cfg = QwenConfig::load(&format!("{DIR}/config.json"));
    let tokenizer =
        tokenizers::Tokenizer::from_file(format!("{DIR}/tokenizer.json")).expect("tok");
    let prompt: Vec<u32> = tokenizer
        .encode("Question: Will it rain tomorrow? Answer:", false)
        .expect("enc")
        .get_ids()
        .to_vec();
    let st = SafeTensors::load(&format!("{DIR}/model.safetensors"));
    let fm = FloatModel::load(&cfg, &st);
    drop(st);
    let tables = Tables::generate(cfg.rope_theta, cfg.head_dim);

    // Calibrate over the prompt as usual.
    let mut calib = Calib::new(cfg.num_hidden_layers);
    let mut fs = FloatState::new(&cfg, qwen::layout::MAX_SEQ);
    let mut tok = prompt[0];
    for pos in 0..prompt.len() {
        let next = qwen::quant::float_forward(
            &fm, &mut fs, qwen::layout::MAX_SEQ, &tables.cos, &tables.sin, tok, &mut calib,
        );
        tok = if pos + 1 < prompt.len() { prompt[pos + 1] } else { next };
    }
    let im = quantize(&fm, &calib);
    let lay = QwenLayout::new(&cfg);
    let image = genesis_image(&lay, &im, &tables, &prompt);
    let native = Native { lay: &lay, im: &im, tables: &tables, threads: 8 };
    let mut mem = FlatMem::new(image);

    // Scales (same formulas as quantize).
    let nl = cfg.num_hidden_layers;
    let res_max = calib.res.iter().fold(1e-6f32, |a, &b| a.max(b));
    let s_res_g = res_max / (1u64 << 29) as f32;
    let s_res: Vec<f32> = calib.res.iter().map(|_| s_res_g).collect();
    let s_xn1: Vec<f32> = calib.xn1.iter().map(|&v| v.max(1e-6) / 32767.0).collect();
    let s_qk: Vec<f32> = calib.qk.iter().map(|&v| v.max(1e-6) / 16384.0).collect();
    let s_v: Vec<f32> = calib.v.iter().map(|&v| v.max(1e-6) / 127.0).collect();
    let s_h: Vec<f32> = calib.ffn_h.iter().map(|&v| v.max(1e-6) / 32767.0).collect();

    // ---- lockstep position 0, token prompt[0] ----
    let (h, dh) = (cfg.hidden_size, cfg.head_dim);
    let (nh, nkv) = (cfg.num_attention_heads, cfg.num_key_value_heads);
    let t0 = prompt[0] as usize;

    // INTEGER side runs whole position first (writes scratch + caches).
    native.position(&mut mem, 0, prompt[0], false);

    // FLOAT side, instrumented, comparing after each stage.
    let x: Vec<f32> = fm.emb[t0 * h..(t0 + 1) * h].to_vec();
    // int x after embedding is NOT recoverable post-run (overwritten by
    // layer residuals) — compare end-of-position only, plus per-layer
    // recomputation below for layer 0 internals.
    let mut xn = vec![0f32; h];
    let lw = &fm.layers[0];
    rmsnorm_f(&x, &lw.ln1, &mut xn);

    // Integer xn for layer 0 is ALSO overwritten by later layers. So for
    // internals we re-run integer layer 0 pieces on a fresh image.
    let image2 = genesis_image(&lay, &im, &tables, &prompt);
    let mut mem2 = FlatMem::new(image2);
    // embedding
    {
        use vm::exec::{rnd, sat16};
        for c in 0..h as u64 {
            let e = mem2.r8i(lay.emb + t0 as u64 * h as u64 + c);
            let v = rnd(e.wrapping_mul(im.m_emb as i64), 20)
                .clamp(i32::MIN as i64, i32::MAX as i64);
            mem2.w32(lay.x + 4 * c, v as i32 as u32);
        }
        let _ = sat16(0);
    }
    let xq: Vec<f32> = (0..h).map(|c| mem2.r32i(lay.x + 4 * c as u64) as f32 * s_res[0]).collect();
    println!("emb rel err: {:.4}", rel(&x, &xq));

    // layer 0 norm1
    native_norm(&native, &mut mem2, lay.x, lay.xn, &im.layers[0].norm1, h as u64);
    let xnq: Vec<f32> =
        (0..h).map(|c| mem2.r16i(lay.xn + 2 * c as u64) as f32 * s_xn1[0]).collect();
    println!("L0 xn1 rel err: {:.4}", rel(&xn, &xnq));

    // layer 0 q projection + qknorm + rope, head 0
    let mut qf = vec![0f32; nh * dh];
    matvec(&lw.wq, &xn, nh * dh, h, &mut qf);
    let mut qh0 = vec![0f32; dh];
    {
        let mut tmp = vec![0f32; dh];
        rmsnorm_f(&qf[0..dh], &lw.q_norm, &mut tmp);
        for p2 in 0..dh / 2 {
            let (c, s) = (
                tables.cos[p2] as f32 / 16384.0,
                tables.sin[p2] as f32 / 16384.0,
            );
            let (a, b) = (tmp[p2], tmp[p2 + dh / 2]);
            qh0[p2] = a * c - b * s;
            qh0[p2 + dh / 2] = a * s + b * c;
        }
    }
    // integer q (proj into scratch then norm+rope) — reuse Native pieces via
    // a single full layer-0-only model? Simplest: full position on mem2
    // would run all layers; instead compare against mem (already ran) is
    // impossible for layer>0 internals. For layer 0, q scratch in `mem` was
    // overwritten by later layers too. So: run the q pipeline manually:
    {
        use vm::exec::{rnd, sat16};
        let a0 = &lay.layers[0];
        let qv = native_proj(&native, &mem2, a0.wq, &im.layers[0].mq, (nh * dh) as u64, h as u64, lay.xn);
        for (r, v) in qv.iter().enumerate() {
            mem2.w16(lay.q + 2 * r as u64, sat16(*v) as u16);
        }
        native_qknorm(&native, &mut mem2, lay.q, &im.layers[0].qnorm, dh as u64);
        native_rope(&native, &mut mem2, lay.q, 0, dh as u64);
        let _ = rnd(0, 0);
    }
    let qq: Vec<f32> = (0..dh).map(|d| mem2.r16i(lay.q + 2 * d as u64) as f32 * s_qk[0]).collect();
    println!("L0 q head0 (norm+rope) rel err: {:.4}", rel(&qh0, &qq));

    // Whole-position landmarks: residual AFTER each layer, float vs int.
    // Rerun integer fresh, capturing x at each layer exit.
    let image3 = genesis_image(&lay, &im, &tables, &prompt);
    let mut mem3 = FlatMem::new(image3);
    let xs_int = native_position_capture(&native, &mut mem3, 0, prompt[0]);
    // float layer-exits (with the integer-matching rescale points: capture
    // BEFORE entering next layer, real units).
    let mut fs2 = FloatState::new(&cfg, qwen::layout::MAX_SEQ);
    let xs_f = float_position_capture(&fm, &mut fs2, &tables, prompt[0]);
    for l in 0..nl {
        let deq: Vec<f32> =
            xs_int[l].iter().map(|&q| q as f32 * s_res[l + 1]).collect();
        println!(
            "layer {:>2} exit residual rel err: {:.4}   (|f| max {:.2})",
            l,
            rel(&xs_f[l], &deq),
            xs_f[l].iter().fold(0f32, |a, &b| a.max(b.abs()))
        );
    }
    // ---- drill into layer 2 (the sink-formation layer) ----
    let l2 = 2usize;
    let s_xn2: Vec<f32> = calib.xn2.iter().map(|&v| v.max(1e-6) / 32767.0).collect();
    let s_gate: Vec<f32> = calib.gate.iter().map(|&v| v.max(1e-6) / 32767.0).collect();
    let s_up: Vec<f32> = calib.up.iter().map(|&v| v.max(1e-6) / 32767.0).collect();
    // float entry to layer 2 = exit of layer 1.
    let xf2 = &xs_f[l2 - 1];
    let lw2 = &fm.layers[l2];
    let fdim = cfg.intermediate_size;
    let mut xn2f = vec![0f32; h];
    // (the float path entering ffn: after attn-add of layer 2 — recompute
    // attn quickly: pos0 ⇒ ctx = v.)
    let mut xattn = xf2.clone();
    {
        let mut xnt = vec![0f32; h];
        rmsnorm_f(&xattn, &lw2.ln1, &mut xnt);
        let kv_per = nkv * dh;
        let mut q2 = vec![0f32; nh * dh];
        let mut kc2 = vec![0f32; kv_per];
        let mut vc2 = vec![0f32; kv_per];
        matvec(&lw2.wq, &xnt, nh * dh, h, &mut q2);
        matvec(&lw2.wk, &xnt, kv_per, h, &mut kc2);
        matvec(&lw2.wv, &xnt, kv_per, h, &mut vc2);
        let mut attn = vec![0f32; nh * dh];
        for hd in 0..nh {
            let kvh = hd / (nh / nkv);
            attn[hd * dh..(hd + 1) * dh].copy_from_slice(&vc2[kvh * dh..(kvh + 1) * dh]);
        }
        let mut o = vec![0f32; h];
        matvec(&lw2.wo, &attn, h, nh * dh, &mut o);
        for i in 0..h {
            xattn[i] += o[i];
        }
    }
    rmsnorm_f(&xattn, &lw2.ln2, &mut xn2f);
    let mut gatef = vec![0f32; fdim];
    let mut upf = vec![0f32; fdim];
    matvec(&lw2.w_gate, &xn2f, fdim, h, &mut gatef);
    matvec(&lw2.w_up, &xn2f, fdim, h, &mut upf);
    let mut hf = vec![0f32; fdim];
    for i in 0..fdim {
        hf[i] = (gatef[i] / (1.0 + (-gatef[i]).exp())) * upf[i];
    }
    let spike = hf
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
        .unwrap()
        .0;
    println!(
        "float L2: |xn2|max {:.2}  gate[spike] {:.2}  up[spike] {:.2}  h[spike] {:.2} (ch {})",
        xn2f.iter().fold(0f32, |a, &b| a.max(b.abs())),
        gatef[spike],
        upf[spike],
        hf[spike],
        spike
    );
    println!(
        "calib L2: xn2max {:.2} gatemax {:.2} upmax {:.2} hmax {:.2}",
        calib.xn2[l2], calib.gate[l2], calib.up[l2], calib.ffn_h[l2]
    );

    // integer: prefix to 2 layers (exit of layer 1 = entry of 2 at s_res[2]),
    // then run layer-2 pieces manually up to h.
    let image4 = genesis_image(&lay, &im, &tables, &prompt);
    let mut m4 = FlatMem::new(image4);
    native.position_prefix(&mut m4, 0, prompt[0], 2);
    // attn part of layer 2 — reuse full prefix(3) for the residual compare,
    // but for ffn internals re-run norm2/gate/up on a prefix-2.5 state.
    // Cheat: run prefix(3) on a clone to get the committed ffn scratch.
    let image5 = genesis_image(&lay, &im, &tables, &prompt);
    let mut m5 = FlatMem::new(image5);
    native.position_prefix(&mut m5, 0, prompt[0], 3);
    let hq_spike = m5.r16i(lay.h_ffn + 2 * spike as u64) as f32 * s_h[l2];
    println!("int   L2: h[spike] deq {:.2}  (raw {})", hq_spike, m5.r16i(lay.h_ffn + 2 * spike as u64));
    // prefix(3) leaves layer-2 ffn scratch behind: xn == layer-2 norm2 out.
    let xn2_int: Vec<f32> =
        (0..h).map(|c| m5.r16i(lay.xn + 2 * c as u64) as f32 * s_xn2[l2]).collect();
    println!("int   L2: xn2 rel err {:.4}", rel(&xn2f, &xn2_int));
    let a2 = &lay.layers[l2];
    let il2 = &im.layers[l2];
    let gq = native_proj(&native, &m5, a2.w_gate, &il2.m_gate, fdim as u64, h as u64, lay.xn);
    let uq = native_proj(&native, &m5, a2.w_up, &il2.m_up, fdim as u64, h as u64, lay.xn);
    use vm::exec::{rnd as vrnd, sat16 as vsat16, trunc_div as vdiv};
    let g = vsat16(gq[spike]) as i64;
    let u = vsat16(uq[spike]) as i64;
    let x411 = vrnd(g.wrapping_mul(il2.m_sig as i64), 20);
    let em = native.tables_exp_at(if x411 >= 0 { -x411 } else { x411 });
    let sig = if x411 >= 0 { vdiv(1i64 << 28, 16384 + em) } else { vdiv(em << 14, 16384 + em) };
    let hpre = vrnd(g.wrapping_mul(sig), 14);
    let prod = hpre.wrapping_mul(u);
    let hq = vrnd(prod.wrapping_mul(il2.m_h as i64), 20);
    println!(
        "int   L2 spike chain: g_q {} ({:.2} real)  u_q {} ({:.2} real)  x411 {}  em {}  sig {}  hpre {}  m_sig {}  m_h {}  m_gate[55] {}  m_up[55] {}  → h_q {}",
        g, g as f32 * s_gate[l2], u, u as f32 * s_up[l2], x411, em, sig, hpre,
        il2.m_sig, il2.m_h, il2.m_gate[spike], il2.m_up[spike], hq
    );
    let _ = s_v;
}

// -- small wrappers around Native internals (private fns re-exposed by
//    copying their logic where needed) --

fn native_norm(n: &Native, mem: &mut FlatMem, src: u64, dst: u64, site: &qwen::quant::NormSite, nn: u64) {
    use vm::exec::{rnd, sat16, trunc_div};
    let mut ss = 0i64;
    for c in 0..nn {
        let xp = rnd(mem.r32i(src + 4 * c), site.pre_shift);
        ss = ss.wrapping_add(xp.wrapping_mul(xp));
    }
    let mean = trunc_div(ss, nn as i64);
    let r = n.tables.rsqrt[(sat16(mean) as i64 + 32768) as usize] as i64;
    for c in 0..nn {
        let t = rnd(mem.r32i(src + 4 * c).wrapping_mul(r), 14);
        let v = rnd(t.wrapping_mul(site.gamma_m[c as usize] as i64), site.shift - 14);
        mem.w16(dst + 2 * c, sat16(v) as u16);
    }
}

fn native_proj(_n: &Native, mem: &FlatMem, w: u64, m: &[i32], rows: u64, cols: u64, x16: u64) -> Vec<i64> {
    use vm::exec::rnd;
    let x = mem.slice(x16, 2 * cols as usize);
    let wb = mem.slice(w, (rows * cols) as usize);
    (0..rows as usize)
        .map(|r| {
            let mut acc = 0i64;
            for (wc, xc) in wb[r * cols as usize..(r + 1) * cols as usize]
                .chunks(64)
                .zip(x.chunks(128))
            {
                let mut part = 0i32;
                for (a, b) in wc.iter().zip(xc.chunks_exact(2)) {
                    part += (*a as i8 as i32) * (i16::from_le_bytes([b[0], b[1]]) as i32);
                }
                acc += part as i64;
            }
            rnd(acc.wrapping_mul(m[r] as i64), 20)
        })
        .collect()
}

fn native_qknorm(n: &Native, mem: &mut FlatMem, base: u64, site: &qwen::quant::NormSite, dh: u64) {
    use vm::exec::{rnd, sat16, trunc_div};
    let mut ss = 0i64;
    for d in 0..dh {
        let v = mem.r16i(base + 2 * d);
        ss = ss.wrapping_add(v.wrapping_mul(v));
    }
    let mean = rnd(trunc_div(ss, dh as i64), site.pre_shift);
    let r = n.tables.rsqrt[(sat16(mean) as i64 + 32768) as usize] as i64;
    for d in 0..dh {
        let v = rnd(
            mem.r16i(base + 2 * d).wrapping_mul(r).wrapping_mul(site.gamma_m[d as usize] as i64),
            site.shift,
        );
        mem.w16(base + 2 * d, sat16(v) as u16);
    }
}

fn native_rope(n: &Native, mem: &mut FlatMem, base: u64, pos: usize, dh: u64) {
    use vm::exec::{rnd, sat16};
    let half = (dh / 2) as usize;
    for p2 in 0..half {
        let c = n.tables.cos[pos * half + p2] as i64;
        let s = n.tables.sin[pos * half + p2] as i64;
        let ns = n.tables.nsin[pos * half + p2] as i64;
        let a = mem.r16i(base + 2 * p2 as u64);
        let b = mem.r16i(base + 2 * (p2 + half) as u64);
        let na = rnd(a.wrapping_mul(c).wrapping_add(b.wrapping_mul(ns)), 14);
        let nb = rnd(a.wrapping_mul(s).wrapping_add(b.wrapping_mul(c)), 14);
        mem.w16(base + 2 * p2 as u64, sat16(na) as u16);
        mem.w16(base + 2 * (p2 + half) as u64, sat16(nb) as u16);
    }
}

/// Integer position 0 capturing the residual (raw i16 quanta) at each
/// layer exit (post-rescale, i.e. at s_res[l+1]).
fn native_position_capture(n: &Native, mem: &mut FlatMem, pos: usize, tok: u32) -> Vec<Vec<i64>> {
    // Run the real thing layer by layer by calling position() ONCE is not
    // segmented; instead replicate: easiest correct path — run position()
    // fully on a scratch image is NOT segmentable, so we re-run the layer
    // loop here using the same Native helpers. To avoid drift this probe
    // reuses Native::position itself with a capture hook… simplest: rerun
    // full position with `capture` recompute per prefix is O(L²) — fine
    // for one position: for each l, fresh image, run layers 0..=l.
    let h = n.im.cfg.hidden_size as u64;
    let nl = n.im.cfg.num_hidden_layers;
    let mut out = Vec::new();
    for upto in 0..nl {
        let img = mem.bytes.clone(); // genesis copy
        let mut m2 = FlatMem::new(img);
        n.position_prefix(&mut m2, pos, tok, upto + 1);
        out.push((0..h).map(|c| m2.r32i(n.lay.x + 4 * c)).collect());
    }
    out
}

/// Float position 0 capturing the residual at each layer exit.
fn float_position_capture(
    fm: &FloatModel,
    st: &mut FloatState,
    tables: &Tables,
    token: u32,
) -> Vec<Vec<f32>> {
    let mut calib = Calib::new(fm.cfg.num_hidden_layers);
    // run instrumented float by reusing float_forward? It doesn't capture.
    // Cheap trick: float layers are deterministic; replicate with capture
    // via repeated truncated runs like the integer side — but float_forward
    // has no prefix mode either. Implement inline single-position float
    // with capture (mirrors quant::float_forward, attention pos 0 only).
    let cfg = &fm.cfg;
    let (h, dh) = (cfg.hidden_size, cfg.head_dim);
    let (nh, nkv) = (cfg.num_attention_heads, cfg.num_key_value_heads);
    let t0 = token as usize;
    let mut x: Vec<f32> = fm.emb[t0 * h..(t0 + 1) * h].to_vec();
    let mut xn = vec![0f32; h];
    let mut q = vec![0f32; nh * dh];
    let mut outs = Vec::new();
    let kv_per = nkv * dh;
    let mut kc = vec![0f32; kv_per];
    let mut vc = vec![0f32; kv_per];
    for lw in &fm.layers {
        rmsnorm_f(&x, &lw.ln1, &mut xn);
        matvec(&lw.wq, &xn, nh * dh, h, &mut q);
        matvec(&lw.wk, &xn, kv_per, h, &mut kc);
        matvec(&lw.wv, &xn, kv_per, h, &mut vc);
        let rot = |vecv: &mut [f32]| {
            for p2 in 0..dh / 2 {
                let (c, s) = (
                    tables.cos[p2] as f32 / 16384.0,
                    tables.sin[p2] as f32 / 16384.0,
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
            let ks = &mut kc[kv * dh..(kv + 1) * dh];
            let mut tmp = vec![0f32; dh];
            rmsnorm_f(ks, &lw.k_norm, &mut tmp);
            ks.copy_from_slice(&tmp);
            rot(ks);
        }
        // pos 0 attention: softmax over one position = weight 1 ⇒ ctx = v.
        let mut attn = vec![0f32; nh * dh];
        for hd in 0..nh {
            let kvh = hd / (nh / nkv);
            attn[hd * dh..(hd + 1) * dh].copy_from_slice(&vc[kvh * dh..(kvh + 1) * dh]);
        }
        let mut o = vec![0f32; h];
        matvec(&lw.wo, &attn, h, nh * dh, &mut o);
        for i in 0..h {
            x[i] += o[i];
        }
        rmsnorm_f(&x, &lw.ln2, &mut xn);
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
        let mut down = vec![0f32; h];
        matvec(&lw.w_down, &hb, h, f, &mut down);
        for i in 0..h {
            x[i] += down[i];
        }
        outs.push(x.clone());
    }
    let _ = (&mut calib, st);
    outs
}
