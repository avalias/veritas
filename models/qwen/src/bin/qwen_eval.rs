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

    // Calibrate on the remainder tail (outside scored chunks).
    let tail = &tokens[n_chunks * MAX_SEQ..];
    let calib_toks: &[u32] = if tail.len() >= 16 { tail } else { &tokens[..MAX_SEQ] };
    println!("calibrating on {} tail tokens…", calib_toks.len().min(MAX_SEQ));
    let mut calib = Calib::new(cfg.num_hidden_layers);
    {
        let mut fs = FloatState::new(&cfg, MAX_SEQ);
        for &t in calib_toks.iter().take(MAX_SEQ) {
            let _ = float_forward_logits(&fm, &mut fs, MAX_SEQ, &tables.cos, &tables.sin, t, &mut calib);
        }
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
        // Real-scale factor: real_logit = int_logit · s_emb · s_xnf · 2^11.
        let k = im.s_emb_eval as f64 * im.s_xnf_eval as f64 * 2048.0;
        let mut int_logits = vec![0i64; cfg.vocab_size];
        // Float twin runs alongside for top-1 agreement (slow but bounded).
        for c in 0..n_chunks {
            let chunk = &tokens[c * MAX_SEQ..(c + 1) * MAX_SEQ];
            let image = genesis_image(&lay, &im, &tables, &chunk[..1].to_vec());
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
