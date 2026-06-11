//! FW-6 proof harness: Qwen3-0.6B "as published" under COMMITTED float
//! semantics (pinned reduction tree, committed exp, no runtime libm).
//!
//!   cargo run -p qwen --release --bin fqwen -- demo  [prompt] [n_gen]
//!   cargo run -p qwen --release --bin fqwen -- ppl   [n_chunks]
//!   cargo run -p qwen --release --bin fqwen -- stable [prompt] [n_gen]
//!
//! demo   — greedy decode + tokens/s (the predictor).
//! ppl    — llama-perplexity convention (512-token chunks, second half
//!          scored) on the committed eval corpus: the QUALITY proof. Bars:
//!          our libm float reference 34.60, llama.cpp Q8_0 34.99.
//! stable — decodes TWICE with different thread counts (1 vs all) and
//!          asserts every emitted token AND every final logit bit is
//!          IDENTICAL: the determinism proof (float edition of §9.1).
#![allow(clippy::float_arithmetic)] // FW-6 + f64 eval-side scoring

use qwen::config::QwenConfig;
use qwen::fmodel::{fposition, FModel, FState};
use qwen::tensors::SafeTensors;
use std::io::Write as _;
use std::time::Instant;

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/artifacts");
const EVAL: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../benches/eval_text.txt");
const MAX_SEQ: usize = 512;

fn logprob_of(logits: &[f32], target: usize) -> f64 {
    let lr: Vec<f64> = logits.iter().map(|&v| v as f64).collect();
    let mx = lr.iter().cloned().fold(f64::MIN, f64::max);
    let lse = mx + lr.iter().map(|&v| (v - mx).exp()).sum::<f64>().ln();
    lr[target] - lse
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mode = args.next().unwrap_or_else(|| "demo".into());

    let cfg = QwenConfig::load(&format!("{DIR}/config.json"));
    let tokenizer =
        tokenizers::Tokenizer::from_file(format!("{DIR}/tokenizer.json")).expect("tokenizer");
    println!("· loading bf16 weights (resident — published form)…");
    let t0 = Instant::now();
    let st = SafeTensors::load(&format!("{DIR}/model.safetensors"));
    let m = FModel::load(&cfg, &st, MAX_SEQ);
    drop(st);
    println!("  loaded in {:.1?}", t0.elapsed());
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);

    match mode.as_str() {
        "demo" | "stable" => {
            let prompt_text = args.next().unwrap_or_else(|| {
                "Question: Will it rain in Paris tomorrow? Evidence: The forecast shows heavy clouds. Answer (yes or no):".into()
            });
            let n_gen: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(24);
            let prompt: Vec<u32> =
                tokenizer.encode(prompt_text.as_str(), false).expect("enc").get_ids().to_vec();

            let decode = |nthreads: usize| -> (Vec<u32>, Vec<u32>, u128) {
                let pool = kernels::Pool::new(nthreads);
                let mut fs = FState::new(&cfg, MAX_SEQ);
                let mut logits = vec![0f32; cfg.vocab_size];
                let mut out = Vec::new();
                let mut tok = prompt[0];
                let t0 = Instant::now();
                let n_pos = prompt.len() + n_gen - 1;
                for pos in 0..n_pos {
                    let decide = pos >= prompt.len() - 1;
                    match fposition(&m, &mut fs, &pool, tok, decide, &mut logits) {
                        Some(t) => {
                            out.push(t);
                            if t == cfg.eos_token_id {
                                break;
                            }
                            tok = t;
                        }
                        None => tok = prompt[pos + 1],
                    }
                }
                let us = t0.elapsed().as_micros();
                (out, logits.iter().map(|v| v.to_bits()).collect(), us)
            };

            if mode == "stable" {
                println!("· determinism proof: decode with 1 thread vs {threads} threads…");
                let (t1, l1, _) = decode(1);
                let (tn, ln, _) = decode(threads);
                assert_eq!(t1, tn, "tokens must be identical across thread counts");
                assert_eq!(l1, ln, "ALL {} final logits bit-identical", ln.len());
                println!(
                    "  ✓ {} tokens AND all {} final-position logits BIT-IDENTICAL (1 vs {} threads)",
                    t1.len(),
                    ln.len(),
                    threads
                );
                println!("  output: {:?}", tokenizer.decode(&t1, false).unwrap_or_default());
            } else {
                let (toks, _, us) = decode(threads);
                let text = tokenizer.decode(&toks, false).unwrap_or_default();
                println!("  output ({} tokens): {:?}", toks.len(), text);
                let n = toks.len().max(1) as u128;
                println!(
                    "  tokens/s: {}.{:01}   ({} µs/token)",
                    (n * 1_000_000) / us.max(1),
                    ((n * 10_000_000) / us.max(1)) % 10,
                    us / n
                );
            }
        }
        "ppl" => {
            let n_chunks: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(2);
            let text = std::fs::read_to_string(EVAL).expect("eval text");
            let tokens: Vec<u32> =
                tokenizer.encode(text.as_str(), false).expect("enc").get_ids().to_vec();
            assert!(tokens.len() >= n_chunks * MAX_SEQ);
            let pool = kernels::Pool::new(threads);
            let half = MAX_SEQ / 2;
            let mut logits = vec![0f32; cfg.vocab_size];
            let mut nll = 0f64;
            let mut scored = 0usize;
            for c in 0..n_chunks {
                let chunk = &tokens[c * MAX_SEQ..(c + 1) * MAX_SEQ];
                let mut fs = FState::new(&cfg, MAX_SEQ);
                for p in 0..MAX_SEQ - 1 {
                    fposition(&m, &mut fs, &pool, chunk[p], true, &mut logits);
                    if p + 1 >= half {
                        nll -= logprob_of(&logits, chunk[p + 1] as usize);
                        scored += 1;
                    }
                    print!("\r  chunk {c} pos {p}    ");
                    let _ = std::io::stdout().flush();
                }
            }
            println!();
            println!("== committed-float PPL over {scored} scored tokens: {:.4} ==", (nll / scored as f64).exp());
            println!("(bars: libm float reference 34.60, llama.cpp Q8_0 34.99 ± 5.11)");
        }
        other => panic!("unknown mode {other}"),
    }
}
