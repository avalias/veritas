//! Quality harness: perplexity + top-1 agreement, replicating llama.cpp's
//! convention (512-token chunks, score the second half of each chunk) so
//! our numbers are directly comparable to `llama-perplexity -c 512`.
//!
//!   cargo run -p qwen --release --bin qwen_eval -- int|float [n_chunks]
//!
//! Calibration uses the tail remainder of the eval file (tokens beyond the
//! scored chunks) — no leakage into scored text. Floats here are eval-only
//! measurement (softmax/log over integer logits), not committed semantics.
#![allow(clippy::float_arithmetic)]
#![allow(clippy::needless_range_loop)] // eval favors index clarity

use qwen::config::QwenConfig;
use qwen::forward::Native;
use qwen::image::{genesis_image, Tables};
use qwen::layout::{QwenLayout, MAX_SEQ};
use qwen::quant::{float_forward_logits, quantize, Calib, FloatModel, FloatState};
use qwen::tensors::SafeTensors;
use std::io::Write as _;
use toy_model::forward::FlatMem;

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/artifacts");
const EVAL: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../benches/eval_text.txt");
const CALIB: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../benches/calib_text.txt");

fn logprob_of(logits_real: &[f64], target: usize) -> f64 {
    let mx = logits_real.iter().cloned().fold(f64::MIN, f64::max);
    let lse = mx + logits_real.iter().map(|&v| (v - mx).exp()).sum::<f64>().ln();
    logits_real[target] - lse
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mode = args.next().unwrap_or_else(|| "int".into());
    let n_chunks: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(2);
    // Third arg "agree" enables the (slow) float twin for top-1 agreement.
    let with_agree = args.next().as_deref() == Some("agree");

    let cfg = QwenConfig::load(&format!("{DIR}/config.json"));
    let tokenizer =
        tokenizers::Tokenizer::from_file(format!("{DIR}/tokenizer.json")).expect("tok");
    let text = std::fs::read_to_string(EVAL).expect("eval text");
    let tokens: Vec<u32> = tokenizer.encode(text.as_str(), false).expect("enc").get_ids().to_vec();
    println!("eval tokens: {} (chunks of {MAX_SEQ}, scoring second half)", tokens.len());
    assert!(tokens.len() >= n_chunks * MAX_SEQ, "need ≥ {} tokens", n_chunks * MAX_SEQ);

    let st = SafeTensors::load(&format!("{DIR}/model.safetensors"));
    let fm = FloatModel::load(&cfg, &st);
    drop(st);
    let tables = Tables::generate(cfg.rope_theta, cfg.head_dim);

    // Calibrate on a DEDICATED corpus (disjoint from eval text): richer
    // max statistics than the eval-file tail; two fresh-state passes of
    // MAX_SEQ tokens each.
    let calib_text = std::fs::read_to_string(CALIB).expect("calib text");
    let calib_toks: Vec<u32> =
        tokenizer.encode(calib_text.as_str(), false).expect("enc").get_ids().to_vec();
    println!("calibrating on {} dedicated tokens…", calib_toks.len());
    let mut calib = Calib::new(cfg.num_hidden_layers);
    for pass in 0..2usize {
        let start = pass * MAX_SEQ;
        if start + 8 > calib_toks.len() {
            break;
        }
        let mut fs = FloatState::new(&cfg, MAX_SEQ);
        for &t in calib_toks[start..].iter().take(MAX_SEQ) {
            let _ = float_forward_logits(&fm, &mut fs, MAX_SEQ, &tables.cos, &tables.sin, t, &mut calib);
        }
    }

    if mode == "diag" {
        // Single-position logit diagnosis: float vs int·K at position P.
        let im = quantize(&fm, &calib);
        let lay = QwenLayout::new(&cfg);
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let native = Native { lay: &lay, im: &im, tables: &tables, threads, pool: kernels::Pool::new(threads) };
        let k = im.s_logit_eval as f64;
        let pp = 48usize; // prefix length
        let chunk = &tokens[0..pp + 1];
        let image = genesis_image(&lay, &im, &tables, &chunk[..1]);
        let mut mem = FlatMem::new(image);
        let mut fs = FloatState::new(&cfg, MAX_SEQ);
        let mut dummy = Calib::new(cfg.num_hidden_layers);
        let mut fl = vec![];
        for p in 0..=pp {
            native.position(&mut mem, p, chunk[p], true);
            fl = float_forward_logits(&fm, &mut fs, MAX_SEQ, &tables.cos, &tables.sin, chunk[p], &mut dummy);
        }
        let mut il = vec![0i64; cfg.vocab_size];
        native.head_logits(&mem, &mut il);
        let ir: Vec<f64> = il.iter().map(|&v| v as f64 * k).collect();
        let mut fi: Vec<(usize, f32)> = fl.iter().cloned().enumerate().collect();
        fi.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        println!("float top5: {:?}", &fi[..5]);
        let mut ii: Vec<(usize, f64)> = ir.iter().cloned().enumerate().collect();
        ii.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        println!("int·K top5: {:?}", &ii[..5]);
        let mean_f: f64 = fl.iter().map(|&v| v as f64).sum::<f64>() / fl.len() as f64;
        let mean_i: f64 = ir.iter().sum::<f64>() / ir.len() as f64;
        let mut dsum = 0f64;
        let mut dmax = 0f64;
        for (a, b) in fl.iter().zip(&ir) {
            let d = (*a as f64 - b).abs();
            dsum += d;
            if d > dmax { dmax = d; }
        }
        println!("mean float {mean_f:.3} | mean int·K {mean_i:.3} | mean|Δ| {:.3} | max|Δ| {dmax:.3} | K {k:.3e}", dsum / fl.len() as f64);
        return;
    }
    let half = MAX_SEQ / 2;
    let mut nll = 0f64;
    let mut scored = 0usize;
    let mut agree = 0usize;

    if mode == "float" {
        // The "true Qwen" bar: float reference, same convention.
        let mut dummy = Calib::new(cfg.num_hidden_layers);
        for c in 0..n_chunks {
            let chunk = &tokens[c * MAX_SEQ..(c + 1) * MAX_SEQ];
            let mut fs = FloatState::new(&cfg, MAX_SEQ);
            for p in 0..MAX_SEQ - 1 {
                let logits =
                    float_forward_logits(&fm, &mut fs, MAX_SEQ, &tables.cos, &tables.sin, chunk[p], &mut dummy);
                if p + 1 >= half {
                    let lr: Vec<f64> = logits.iter().map(|&v| v as f64).collect();
                    nll -= logprob_of(&lr, chunk[p + 1] as usize);
                    scored += 1;
                }
                print!("\r  chunk {c} pos {p}    ");
                let _ = std::io::stdout().flush();
            }
        }
        println!();
    } else {
        let im = quantize(&fm, &calib);
        let lay = QwenLayout::new(&cfg);
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let native = Native { lay: &lay, im: &im, tables: &tables, threads, pool: kernels::Pool::new(threads) };
        // Real-scale factor: one common logit quantum from the per-row head.
        let k = im.s_logit_eval as f64;
        let mut int_logits = vec![0i64; cfg.vocab_size];
        // Float twin runs alongside for top-1 agreement (slow but bounded).
        for c in 0..n_chunks {
            let chunk = &tokens[c * MAX_SEQ..(c + 1) * MAX_SEQ];
            let image = genesis_image(&lay, &im, &tables, &chunk[..1]);
            let mut mem = FlatMem::new(image);
            let mut fs = FloatState::new(&cfg, MAX_SEQ);
            let mut dummy = Calib::new(cfg.num_hidden_layers);
            for p in 0..MAX_SEQ - 1 {
                native.position(&mut mem, p, chunk[p], true);
                if p + 1 >= half {
                    native.head_logits(&mem, &mut int_logits);
                    let lr: Vec<f64> = int_logits.iter().map(|&v| v as f64 * k).collect();
                    nll -= logprob_of(&lr, chunk[p + 1] as usize);
                    scored += 1;
                    if with_agree {
                        let fl = float_forward_logits(
                            &fm, &mut fs, MAX_SEQ, &tables.cos, &tables.sin, chunk[p], &mut dummy,
                        );
                        let fa = fl
                            .iter()
                            .enumerate()
                            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                            .unwrap()
                            .0;
                        let ia = int_logits.iter().enumerate().max_by_key(|(_, &v)| v).unwrap().0;
                        if fa == ia {
                            agree += 1;
                        }
                    }
                } else if with_agree {
                    // keep float twin's KV in sync on unscored prefix
                    let _ = float_forward_logits(
                        &fm, &mut fs, MAX_SEQ, &tables.cos, &tables.sin, chunk[p], &mut dummy,
                    );
                }
                print!("\r  chunk {c} pos {p}    ");
                let _ = std::io::stdout().flush();
            }
        }
        println!();
        if with_agree {
            println!(
                "top-1 agreement int vs float: {}/{} = {:.1}%",
                agree,
                scored,
                100.0 * agree as f64 / scored as f64
            );
        }
    }
    let ppl = (nll / scored as f64).exp();
    println!("== {mode} PPL over {scored} scored tokens: {ppl:.4} ==");
    println!("(llama.cpp Q8_0 same text, -c 512: PPL = 34.99 ± 5.11)");
}
