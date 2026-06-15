//! C-14 FLOAT: the committed-float native runtime (hardware floats, fc.rs)
//! and the VM oracle (pure-integer softfloat) execute the SAME committed
//! program over the real bf16 Qwen3-0.6B — memory roots must be
//! bit-identical at every token boundary. This is the final-product check:
//! "Qwen as published", deterministic, and fraud-provable.
//!
//!   cargo run -p compiler --release --bin fqwen_c14 -- [n_prompt] [n_gen]
#![allow(clippy::float_arithmetic)] // host-side reporting only

use compiler::fqwen::compile_fqwen;
use qwen::config::QwenConfig;
use qwen::fc::fc_position;
use qwen::flayout::{fgenesis, FLayout, FMEM_DEPTH};
use qwen::fmodel::FModel;
use qwen::tensors::SafeTensors;
use std::time::Instant;
use toy_model::forward::FlatMem;
use vm::exec::Machine;
use vm::hash::page_leaf_hash;
use vm::merkle::MerkleTree;
use vm::PAGE_SIZE;

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/qwen/artifacts");

fn regions(lay: &FLayout, cfg: &QwenConfig) -> Vec<(String, u64, usize)> {
    let (h, dh, f) = (cfg.hidden_size, cfg.head_dim, cfg.intermediate_size);
    let (nh, nkv) = (cfg.num_attention_heads, cfg.num_key_value_heads);
    let mut r = vec![
        ("x".into(), lay.x, h * 4),
        ("xn".into(), lay.xn, h * 4),
        ("q".into(), lay.q, nh * dh * 4),
        ("attnx".into(), lay.attnx, nh * dh * 4),
        ("gate".into(), lay.gate, f * 4),
        ("up".into(), lay.up, f * 4),
        ("h_ffn".into(), lay.h_ffn, f * 4),
        ("scores".into(), lay.scores, 64),
        ("scratch facc..saved_max".into(), lay.facc, (lay.saved_max - lay.facc) as usize + 4),
        ("logit_buf".into(), lay.logit_buf, PAGE_SIZE),
    ];
    for (l, a) in lay.layers.iter().enumerate().take(2) {
        r.push((format!("kc[{l}]"), a.kc, 4 * nkv * dh * 4));
        r.push((format!("vc[{l}]"), a.vc, 4 * nkv * dh * 4));
    }
    r
}

fn main() {
    let mut args = std::env::args().skip(1);
    let n_prompt: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(2);
    let n_gen: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(1);
    let n_pos = n_prompt + n_gen - 1;

    println!("· loading bf16 Qwen (no quantization — the published model)…");
    let cfg = QwenConfig::load(&format!("{DIR}/config.json"));
    let st = SafeTensors::load(&format!("{DIR}/model.safetensors"));
    let m = FModel::load(&cfg, &st, qwen::flayout::FMAX_SEQ);
    drop(st);

    let lay = FLayout::new(&cfg);
    let prompt: Vec<u32> = (0..n_prompt as u32).map(|i| 9707 + 13 * i).collect();
    println!("· building 2 GiB committed genesis…");
    let t0 = Instant::now();
    let image = fgenesis(&lay, &m, &prompt);
    println!("  image in {:.1?}", t0.elapsed());

    println!("· compiling committed-float Qwen → VM program ({n_prompt}+{n_gen})…");
    let c = compile_fqwen(&lay, &cfg, n_prompt, n_gen);
    println!(
        "  {} instrs (p = {}), {} steps, boundaries {:?}",
        c.program.len(),
        c.p,
        c.total_steps,
        c.token_boundaries
    );

    // Native committed run (hardware floats) + Merkle boundaries.
    println!("· native committed-float run…");
    let t0 = Instant::now();
    let mut nm = FlatMem::new(image.clone());
    let leaves: Vec<_> = nm.bytes.chunks_exact(PAGE_SIZE).map(page_leaf_hash).collect();
    let mut tree =
        MerkleTree::from_leaf_hashes(FMEM_DEPTH, leaves, page_leaf_hash(&[0u8; PAGE_SIZE]));
    println!("  genesis tree in {:.1?}", t0.elapsed());
    let mut native_roots = Vec::new();
    let mut tok = prompt[0];
    for pos in 0..n_pos {
        let decide = pos >= n_prompt - 1;
        let next = fc_position(&m, &lay, &mut nm, pos, tok, decide);
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
    println!("  {} boundaries committed", native_roots.len());

    // VM oracle (pure-integer softfloat) with probe walk.
    println!("· VM oracle ({} steps)…", c.total_steps);
    let mut mp = Machine::with_image(FMEM_DEPTH, c.p, c.program.clone(), &image);
    for (label, psteps, ppc) in &c.probes {
        while mp.regs.step < *psteps {
            mp.step().expect("probe walk");
        }
        assert_eq!(
            mp.regs.pc, *ppc,
            "probe '{label}': step-prediction drift (VM pc {} vs {ppc})",
            mp.regs.pc
        );
    }
    drop(mp);
    let mut vm = Machine::with_image(FMEM_DEPTH, c.p, c.program.clone(), &image);
    let t0 = Instant::now();
    for (i, (&bstep, &bpc)) in c.token_boundaries.iter().zip(&c.boundary_pcs).enumerate() {
        while vm.regs.step < bstep {
            vm.step().expect("running");
        }
        assert_eq!(vm.regs.pc, bpc, "boundary {i}: pc");
        assert_eq!((vm.regs.acc, vm.regs.aux), (0, 0), "boundary {i}: acc/aux");
        assert_eq!(vm.regs.idx, [0; 4], "boundary {i}: idx");
        let vm_root = vm.mem.root();
        if vm_root == native_roots[i] {
            println!("  boundary {i}: mem roots EQUAL ({:.1?})", t0.elapsed());
        } else {
            println!("  boundary {i}: MISMATCH — diffing regions…");
            for (name, base, len) in regions(&lay, &cfg) {
                let v = vm.mem.read(base, len).to_vec();
                let n = nm.slice(base, len);
                if v != n {
                    let off = v.iter().zip(n).position(|(a, b)| a != b).unwrap();
                    println!(
                        "    {name} +{off} (addr {}): vm {:02x?} native {:02x?}",
                        base + off as u64,
                        &v[off..(off + 8).min(len)],
                        &n[off..(off + 8).min(len)]
                    );
                }
            }
            panic!("C-14-float failed at boundary {i}");
        }
    }
    println!(
        "✓ C-14 FLOAT: {} boundaries bit-identical — published-quality Qwen, fraud-provable",
        n_pos
    );
}
