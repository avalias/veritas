//! resolver — the off-chain AI judge, served over HTTP so the dApp can show
//! the REAL Qwen-0.6B reading evidence and typing its verdict live.
//!
//!   cargo run -p qwen --release --bin resolver        # listens on :8899
//!
//! This is who runs Qwen: a resolver, off-chain, on its own hardware (here,
//! your laptop). The chain never runs the model — it only re-runs ONE
//! micro-op if a verdict is disputed (the Fraud Lab). The model is the
//! committed-float Qwen3-0.6B (perplexity 34.60, bit-identical on any CPU).
//!
//! GET /judge?q=<question>&e=<evidence>  → Server-Sent Events:
//!   data:{"t":"<token text delta>"}        (one per decoded token — typing)
//!   data:{"verdict":"YES|NO","done":true}  (final)
#![allow(clippy::float_arithmetic)]

use qwen::config::QwenConfig;
use qwen::fmodel::{fposition, FModel, FScratch, FState};
use qwen::tensors::SafeTensors;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Mutex;

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/artifacts");
const MAX_SEQ: usize = 512;
const N_GEN: usize = 64;

fn main() {
    let cfg = QwenConfig::load(&format!("{DIR}/config.json"));
    let tok = tokenizers::Tokenizer::from_file(format!("{DIR}/tokenizer.json")).expect("tokenizer");
    eprintln!("· loading committed-float Qwen3-0.6B (published weights)…");
    let st = SafeTensors::load(&format!("{DIR}/model.safetensors"));
    let model = FModel::load(&cfg, &st, MAX_SEQ);
    drop(st);
    let threads = std::env::var("QWEN_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(4);
    eprintln!("· judge ready. POST/GET http://127.0.0.1:8899/judge?q=…&e=…");

    let lock = Mutex::new(()); // one inference at a time (bounded memory)
    let listener = TcpListener::bind("127.0.0.1:8899").expect("bind :8899");
    for stream in listener.incoming() {
        let Ok(mut s) = stream else { continue };
        let _g = lock.lock().unwrap();
        if let Err(e) = handle(&mut s, &cfg, &tok, &model, threads) {
            eprintln!("  req error: {e}");
        }
    }
}

fn handle(
    s: &mut TcpStream,
    cfg: &QwenConfig,
    tok: &tokenizers::Tokenizer,
    model: &FModel,
    threads: usize,
) -> std::io::Result<()> {
    // read the request line
    let mut line = String::new();
    BufReader::new(s.try_clone()?).read_line(&mut line)?;
    let path = line.split_whitespace().nth(1).unwrap_or("/");

    if line.starts_with("OPTIONS") {
        return s.write_all(cors_preflight().as_bytes());
    }
    if !path.starts_with("/judge") {
        return s.write_all(b"HTTP/1.1 200 OK\r\nAccess-Control-Allow-Origin: *\r\nContent-Type: text/plain\r\n\r\nqwen judge up");
    }

    let (q, e) = parse_qe(path);
    let prompt_text = format!(
        "Question: {q}\nA trusted news source reported: \"{e}\"\nBased only on this report, is the answer to the question YES? Reply with YES or NO, then one short reason.\nAnswer:"
    );

    // SSE headers
    s.write_all(
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n",
    )?;
    s.flush()?;

    let pool = kernels::Pool::new(threads);
    let mut fs = FState::new(cfg, MAX_SEQ);
    let mut sc = FScratch::new(cfg, MAX_SEQ);
    let mut logits = vec![0f32; cfg.vocab_size];
    let prompt: Vec<u32> = tok.encode(prompt_text.as_str(), false).expect("enc").get_ids().to_vec();

    let mut out: Vec<u32> = Vec::new();
    let mut emitted = String::new();
    let mut cur = prompt[0];
    let n_pos = prompt.len() + N_GEN - 1;
    for pos in 0..n_pos {
        let decide = pos >= prompt.len() - 1;
        match fposition(model, &mut fs, &mut sc, &pool, cur, decide, &mut logits) {
            Some(t) => {
                if t == cfg.eos_token_id {
                    break;
                }
                out.push(t);
                // incremental detokenize: decode all, emit the new suffix
                let full = tok.decode(&out, false).unwrap_or_default();
                if full.len() > emitted.len() && full.starts_with(&emitted) {
                    let delta = full[emitted.len()..].to_string();
                    let _ = write!(s, "data:{}\n\n", json_str("t", &delta));
                    let _ = s.flush();
                    emitted = full;
                    // one clean sentence: stop at the newline after the answer
                    if out.len() >= 4 && emitted.trim_end().contains('\n') {
                        break;
                    }
                }
                cur = t;
            }
            None => cur = prompt[pos + 1],
        }
    }

    let verdict = extract_verdict(&emitted);
    let _ = write!(s, "data:{{\"verdict\":\"{verdict}\",\"done\":true}}\n\n");
    s.flush()
}

/// First standalone YES/NO in the model's answer (defaults to NO — silence
/// is the occurrence null hypothesis).
fn extract_verdict(text: &str) -> &'static str {
    let up = text.to_uppercase();
    let yi = find_word(&up, "YES");
    let ni = find_word(&up, "NO");
    match (yi, ni) {
        (Some(y), Some(n)) => if y <= n { "YES" } else { "NO" },
        (Some(_), None) => "YES",
        _ => "NO",
    }
}

fn find_word(hay: &str, w: &str) -> Option<usize> {
    let b = hay.as_bytes();
    let wb = w.as_bytes();
    let mut i = 0;
    while i + wb.len() <= b.len() {
        if &b[i..i + wb.len()] == wb {
            let before = i == 0 || !b[i - 1].is_ascii_alphabetic();
            let after = i + wb.len() == b.len() || !b[i + wb.len()].is_ascii_alphabetic();
            if before && after {
                return Some(i);
            }
        }
        i += 1;
    }
    None
}

fn parse_qe(path: &str) -> (String, String) {
    let mut q = String::new();
    let mut e = String::new();
    if let Some(qs) = path.split_once('?').map(|x| x.1) {
        for kv in qs.split('&') {
            let (k, v) = kv.split_once('=').unwrap_or((kv, ""));
            let dec = urldecode(v);
            match k {
                "q" => q = dec,
                "e" => e = dec,
                _ => {}
            }
        }
    }
    (q, e)
}

fn urldecode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'%' if i + 2 < b.len() => {
                let h = hexv(b[i + 1]) * 16 + hexv(b[i + 2]);
                out.push(h);
                i += 3;
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn hexv(c: u8) -> u8 {
    match c {
        b'0'..=b'9' => c - b'0',
        b'a'..=b'f' => c - b'a' + 10,
        b'A'..=b'F' => c - b'A' + 10,
        _ => 0,
    }
}

/// minimal JSON string field (escapes the bare minimum for token text)
fn json_str(key: &str, val: &str) -> String {
    let mut e = String::new();
    for ch in val.chars() {
        match ch {
            '"' => e.push_str("\\\""),
            '\\' => e.push_str("\\\\"),
            '\n' => e.push_str("\\n"),
            '\r' => {}
            '\t' => e.push_str("\\t"),
            c if (c as u32) < 0x20 => {}
            c => e.push(c),
        }
    }
    format!("{{\"{key}\":\"{e}\"}}")
}

fn cors_preflight() -> String {
    "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, OPTIONS\r\nAccess-Control-Allow-Headers: *\r\nContent-Length: 0\r\n\r\n".to_string()
}
