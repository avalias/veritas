//! C-14 at Qwen scale: the native checkpoint runtime and the VM oracle,
//! executing the SAME committed program over real Qwen3-0.6B weights, must
//! produce identical memory roots at every token boundary — and the VM's
//! register file must land exactly on the compiler's static prediction.
//!
//!   cargo run -p compiler --release --bin qwen_c14 -- [n_prompt] [n_gen]
//!
//! On a root mismatch the named layout regions are diffed byte-for-byte to
//! localize the first divergent write (the probe methodology, VM edition).

use compiler::qwen::compile_qwen;
use qwen::config::QwenConfig;
use qwen::forward::Native;
use qwen::image::{genesis_image, Tables};
use qwen::layout::{QwenLayout, MEM_DEPTH};
use qwen::quant::{quantize, Calib, FloatModel, FloatState, IntModel};
use qwen::tensors::SafeTensors;
use std::time::Instant;
use toy_model::forward::FlatMem;
use vm::exec::Machine;
use vm::hash::page_leaf_hash;
use vm::merkle::MerkleTree;
use vm::PAGE_SIZE;

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/qwen/artifacts");

fn load_model(cfg: &QwenConfig, tables: &Tables, prompt: &[u32]) -> IntModel {
    let st = SafeTensors::load(&format!("{DIR}/model.safetensors"));
    let fm = FloatModel::load(cfg, &st);
    drop(st);
    let mut calib = Calib::new(cfg.num_hidden_layers);
    let mut fs = FloatState::new(cfg, qwen::layout::MAX_SEQ);
    let mut tok = prompt[0];
    for pos in 0..prompt.len() + 4 {
        let next = qwen::quant::float_forward(
            &fm, &mut fs, qwen::layout::MAX_SEQ, &tables.cos, &tables.sin, tok, &mut calib,
        );
        if pos >= prompt.len() - 1 {
            tok = next;
        } else {
            tok = prompt[pos + 1];
        }
    }
    quantize(&fm, &calib)
}

/// Layout regions for divergence localization.
fn regions(lay: &QwenLayout, im: &IntModel) -> Vec<(String, u64, usize)> {
    let cfg = &im.cfg;
    let (h, dh, f) = (cfg.hidden_size as u64, cfg.head_dim as u64, cfg.intermediate_size as u64);
    let (nh, nkv) = (cfg.num_attention_heads as u64, cfg.num_key_value_heads as u64);
    let mut r = vec![
        ("x".into(), lay.x, (h * 4) as usize),
        ("xp".into(), lay.xp, (h * 2) as usize),
        ("xn".into(), lay.xn, (h * 2) as usize),
        ("q".into(), lay.q, (nh * dh * 2) as usize),
        ("attnx".into(), lay.attnx, (nh * dh * 2) as usize),
        ("att32".into(), lay.att32, 64),
        ("e32".into(), lay.e32, 64),
        ("probs".into(), lay.probs, 64),
        ("cells r32..tok".into(), lay.r32, 16),
        ("cells silu32..v_cell".into(), lay.silu32, 40),
        ("h_ffn".into(), lay.h_ffn, (f * 2) as usize),
        ("logit_buf".into(), lay.logit_buf, PAGE_SIZE),
        ("saved_max".into(), lay.saved_max, 4),
    ];
    for (l, a) in lay.layers.iter().enumerate() {
        r.push((format!("kc[{l}]"), a.kc, (8 * nkv * dh * 2) as usize));
        r.push((format!("vc[{l}]"), a.vc, (8 * nkv * dh * 2) as usize));
    }
    r
}

fn main() {
    let mut args = std::env::args().skip(1);
    let n_prompt: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(3);
    let n_gen: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(2);
    let n_pos = n_prompt + n_gen - 1;

    println!("· loading + calibrating + quantizing…");
    let cfg = QwenConfig::load(&format!("{DIR}/config.json"));
    let tables = Tables::generate(cfg.rope_theta, cfg.head_dim);
    // Fixed prompt ids (real BPE ids; content is irrelevant to C-14).
    let prompt: Vec<u32> = (0..n_prompt as u32).map(|i| 9707 + 13 * i).collect();
    let t0 = Instant::now();
    let im = load_model(&cfg, &tables, &prompt);
    println!("  ready in {:.1?}", t0.elapsed());

    let lay = QwenLayout::new(&cfg);
    let image = genesis_image(&lay, &im, &tables, &prompt);

    println!("· compiling Qwen → VM program ({n_prompt}+{n_gen} tokens)…");
    let c = compile_qwen(&lay, &im, n_prompt, n_gen);
    println!(
        "  {} instrs (p = {}), {} total steps, boundaries at {:?}",
        c.program.len(),
        c.p,
        c.total_steps,
        c.token_boundaries
    );

    // ---- native run, boundary mem roots ----
    println!("· native checkpoint run…");
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let native =
        Native { lay: &lay, im: &im, tables: &tables, threads, pool: kernels::Pool::new(threads) };
    let mut nm = FlatMem::new(image.clone());
    let leaves: Vec<_> = nm.bytes.chunks_exact(PAGE_SIZE).map(page_leaf_hash).collect();
    let mut tree = MerkleTree::from_leaf_hashes(MEM_DEPTH, leaves, page_leaf_hash(&[0u8; PAGE_SIZE]));
    let mut native_roots = Vec::new();
    let mut tok = prompt[0];
    for pos in 0..n_pos {
        let decide = pos >= n_prompt - 1;
        let next = native.position(&mut nm, pos, tok, decide);
        let updates: Vec<_> = nm
            .take_dirty()
            .iter()
            .map(|pg| (*pg, page_leaf_hash(nm.slice(*pg * PAGE_SIZE as u64, PAGE_SIZE))))
            .collect();
        tree.update_leaf_hashes_bulk(&updates);
        native_roots.push(tree.root());
        tok = match next {
            Some(t) => t,
            None => prompt[pos + 1],
        };
    }
    println!("  native tokens fed: …, boundaries committed: {}", native_roots.len());

    // ---- VM oracle run ----
    println!("· VM oracle run ({} steps)…", c.total_steps);
    let mut m = Machine::with_image(MEM_DEPTH, c.p, c.program.clone(), &image);
    let t0 = Instant::now();
    let mut all_ok = true;
    // Probe walk: localize any step-prediction drift to a single block.
    for (label, psteps, ppc) in &c.probes {
        while m.regs.step < *psteps {
            m.step().expect("probe walk");
        }
        if m.regs.pc != *ppc {
            panic!(
                "probe '{label}': predicted pc {ppc} at step {psteps}, VM at pc {} — \
                 prediction drift inside the preceding block",
                m.regs.pc
            );
        }
    }
    let mut m = Machine::with_image(MEM_DEPTH, c.p, c.program.clone(), &image);
    for (i, (&bstep, &bpc)) in c.token_boundaries.iter().zip(&c.boundary_pcs).enumerate() {
        while m.regs.step < bstep {
            match m.step() {
                Ok(vm::exec::StepOutcome::Ran) => {}
                other => panic!(
                    "VM left RUNNING at step {} (pc {}): {:?}",
                    m.regs.step, m.regs.pc, other
                ),
            }
        }
        // Register file must be the compiler's static prediction.
        assert_eq!(m.regs.pc, bpc, "boundary {i}: pc");
        assert_eq!(m.regs.step, bstep, "boundary {i}: step");
        assert_eq!((m.regs.acc, m.regs.aux), (0, 0), "boundary {i}: acc/aux");
        assert_eq!(m.regs.idx, [0; 4], "boundary {i}: idx");
        let vm_root = m.mem.root();
        if vm_root == native_roots[i] {
            println!("  boundary {i}: mem roots EQUAL ({:.1?} elapsed)", t0.elapsed());
        } else {
            all_ok = false;
            println!("  boundary {i}: MEM ROOT MISMATCH — diffing regions…");
            for (name, base, len) in regions(&lay, &im) {
                let vm_bytes = m.mem.read(base, len);
                let nat = nm_at(&nm, base, len);
                if vm_bytes != nat {
                    let off = vm_bytes.iter().zip(&nat).position(|(a, b)| a != b).unwrap();
                    println!(
                        "    {name}: first diff at +{off} (addr {}): vm {:02x?} native {:02x?}",
                        base + off as u64,
                        &vm_bytes[off..(off + 8).min(len)],
                        &nat[off..(off + 8).min(len)]
                    );
                }
            }
            break;
        }
    }
    assert!(all_ok, "C-14 at Qwen scale FAILED");
    println!("✓ C-14 at Qwen scale: {} boundaries bit-identical, {} steps", n_pos, c.total_steps);
}

/// NOTE: diffing against the FINAL native memory — valid for the FIRST
/// mismatching boundary only (later boundaries overwrite scratch).
fn nm_at(nm: &FlatMem, base: u64, len: usize) -> Vec<u8> {
    nm.slice(base, len).to_vec()
}
