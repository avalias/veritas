//! First integer Qwen3-0.6B inference under committed semantics.
//!
//!   cargo run -p qwen --release --bin qwen_demo -- [prompt] [n_gen]
//!
//! Loads the real weights, calibrates on the prompt (float, offline),
//! quantizes to W8A8, builds the 1 GiB committed image, and decodes with
//! per-token checkpoint commitments — reporting tokens/s and the honest-
//! path overhead split (compute vs hashing), plus agreement against the
//! float reference.

use qwen::config::QwenConfig;
use qwen::forward::{run_committed, run_pure, Native};
use qwen::image::{genesis_image, Tables};
use qwen::layout::QwenLayout;
use qwen::quant::{quantize, Calib, FloatModel, FloatState};
use qwen::tensors::SafeTensors;
use std::time::Instant;

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/artifacts");

fn main() {
    let mut args = std::env::args().skip(1);
    let prompt_text = args
        .next()
        .unwrap_or_else(|| "Question: Will it rain in Paris tomorrow? Evidence: The forecast shows heavy clouds. Answer (yes or no):".into());
    let n_gen: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(12);

    println!("· loading config + tokenizer + weights…");
    let cfg = QwenConfig::load(&format!("{DIR}/config.json"));
    let tokenizer = tokenizers::Tokenizer::from_file(format!("{DIR}/tokenizer.json"))
        .expect("vendored tokenizer");
    let prompt: Vec<u32> = tokenizer
        .encode(prompt_text.as_str(), false)
        .expect("encode")
        .get_ids()
        .to_vec();
    println!("  prompt: {} tokens", prompt.len());
    let t0 = Instant::now();
    let st = SafeTensors::load(&format!("{DIR}/model.safetensors"));
    let fm = FloatModel::load(&cfg, &st);
    drop(st);
    println!("  weights loaded in {:.1?}", t0.elapsed());

    println!("· generating integer tables (pure integer — no libm)…");
    let tables = Tables::generate(cfg.rope_theta, cfg.head_dim);

    println!("· calibrating on the prompt (float, offline)…");
    let t0 = Instant::now();
    let mut calib = Calib::new(cfg.num_hidden_layers);
    let mut fs = FloatState::new(&cfg, qwen::layout::MAX_SEQ);
    let mut float_tokens = Vec::new();
    let mut tok = prompt[0];
    for pos in 0..prompt.len() + 5 {
        let next = qwen::quant::float_forward(
            &fm, &mut fs, qwen::layout::MAX_SEQ, &tables.cos, &tables.sin, tok, &mut calib,
        );
        if pos >= prompt.len() - 1 {
            float_tokens.push(next);
            tok = next;
        } else {
            tok = prompt[pos + 1];
        }
    }
    println!("  calib {:?} in {:.1?}", calib, t0.elapsed());

    println!("· quantizing to W8A8 (per-channel weights, static activations)…");
    let t0 = Instant::now();
    let im = quantize(&fm, &calib);
    drop(fm);
    println!("  quantized in {:.1?}", t0.elapsed());

    println!("· building committed 1 GiB genesis image…");
    let lay = QwenLayout::new(&cfg);
    let image = genesis_image(&lay, &im, &tables, &prompt);

    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let native = Native { lay: &lay, im: &im, tables: &tables, threads };

    println!("· pure native decode ({threads} threads)…");
    let (toks_pure, us_pure) = run_pure(&native, image.clone(), &prompt, n_gen);
    let text = tokenizer.decode(&toks_pure, false).unwrap_or_default();
    println!("  integer output: {:?} = {:?}", toks_pure, text);
    println!(
        "  float reference (first {}): {:?} = {:?}",
        float_tokens.len(),
        float_tokens,
        tokenizer.decode(&float_tokens, false).unwrap_or_default()
    );
    let agree = toks_pure
        .iter()
        .zip(&float_tokens)
        .take_while(|(a, b)| a == b)
        .count();
    println!("  int/float greedy agreement: first {agree} tokens");

    println!("· committed decode (per-token checkpoints)…");
    let stats = run_committed(&native, image, &prompt, n_gen);
    assert_eq!(stats.tokens, toks_pure, "commitment must not change results");
    let n_tok = stats.tokens.len().max(1) as u128;
    // integer-only percentages (basis points) to respect the float ban here
    let bp = (stats.hash_us * 10_000) / stats.compute_us.max(1);
    println!("== results ==");
    println!(
        "tokens/s (pure):       {}.{:01}",
        (n_tok * 1_000_000) / us_pure.max(1),
        ((n_tok * 10_000_000) / us_pure.max(1)) % 10
    );
    println!(
        "compute per token:     {} µs   hash per token: {} µs",
        stats.compute_us / n_tok,
        stats.hash_us / n_tok
    );
    println!(
        "commitment overhead:   {}.{:02}% of compute ({} dirty pages total)",
        bp / 100,
        bp % 100,
        stats.dirty_pages
    );
    println!("genesis tree (one-off): {} ms", stats.genesis_us / 1000);
}
