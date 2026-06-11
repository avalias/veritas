//! FW-6 float compiler: lowers the committed-float Qwen3 forward (the
//! fc.rs native runtime, which is itself fmodel.rs with committed writes)
//! to the float ISA (FDOT/FOP) plus the integer control ops.
//!
//! Same discipline as the integer compiler: one segment per position,
//! bottom-tested LOOPs (registers zero at entry — learned the hard way),
//! JEQ branches with EQUAL-LENGTH arms so the step count stays a
//! compile-time fact, canonicalizing tails (boundary regs fully static).
//! fc.rs mirrors this file's writes block-for-block; fqwen_c14 asserts the
//! memory roots agree at every token boundary.

use qwen::config::QwenConfig;
use qwen::flayout::{FLayout, FMAX_SEQ};
use vm::isa::{Instr, Opcode, Operand};

const K0: u8 = 0; // token id (embedding); kv-head loops
const K1: u8 = 1; // element/pair/group/in-chunk-row loops
const K2: u8 = 2; // row / score-position / dim / chunk loops
const K3: u8 = 3; // innermost gathers; absolute vocab row (head)

const CHUNK: u64 = 256;

pub struct FCompiled {
    pub program: Vec<Instr>,
    pub p: u8,
    pub probes: Vec<(String, u64, u32)>,
    pub total_steps: u64,
    pub token_boundaries: Vec<u64>,
    pub boundary_pcs: Vec<u32>,
    pub n_prompt: usize,
    pub n_gen: usize,
}

struct Em {
    prog: Vec<Instr>,
    steps: u64,
    mult: u64,
    probes: Vec<(String, u64, u32)>,
}

impl Em {
    fn emit(&mut self, i: Instr) {
        self.prog.push(i);
        self.steps += self.mult;
    }

    fn pc(&self) -> u32 {
        self.prog.len() as u32
    }

    fn probe(&mut self, label: String) {
        let (s, p) = (self.steps, self.pc());
        self.probes.push((label, s, p));
    }

    fn loop_over(&mut self, k: u8, trips: u32, body: impl FnOnce(&mut Em)) {
        assert!(trips >= 1);
        let start = self.pc();
        self.mult *= trips as u64;
        body(self);
        self.emit(Instr { k, target: start, imm: trips, ..Instr::op(Opcode::Loop) });
        self.mult /= trips as u64;
    }

    /// JEQ(flag == 0 → skip-arm); both arms MUST have equal step counts.
    /// `update` runs when flag == 1 (gets a trailing JMP emitted here);
    /// the skip arm is `pad` LDC-0 fillers.
    fn ifflag(&mut self, flag: u64, update: impl FnOnce(&mut Em), pad: u32) {
        let jeq_at = self.prog.len();
        self.emit(Instr { a: at(flag), imm: 0, ..Instr::op(Opcode::Jeq) });
        let upd_start = self.prog.len();
        update(self);
        let jmp_at = self.prog.len();
        self.emit(Instr::op(Opcode::Jmp));
        let upd_len = (self.prog.len() - upd_start) as u32;
        assert_eq!(upd_len, pad + 1, "update arm = pad body ops + trailing JMP");
        self.prog[jeq_at].target = self.pc();
        // Skip arm: pad+1 fillers — taken path executes JEQ + pad body ops
        // + JMP = pad+2 steps; the skip path must match: JEQ + (pad+1)
        // fillers = pad+2. Fillers are 0-counted (the taken arm carries
        // the step count; exactly one arm executes).
        let saved = self.mult;
        self.mult = 0;
        for _ in 0..=pad {
            self.emit(ldc(0));
        }
        self.mult = saved;
        self.prog[jmp_at].target = self.pc();
    }
}

fn at(base: u64) -> Operand {
    Operand::at(base)
}

fn st(base: u64, k: u8, stride: u32) -> Operand {
    let mut o = Operand::at(base);
    o.stride[k as usize] = stride;
    o
}

fn st2(base: u64, k1: u8, s1: u32, k2: u8, s2: u32) -> Operand {
    let mut o = Operand::at(base);
    o.stride[k1 as usize] = s1;
    o.stride[k2 as usize] = s2;
    o
}

fn st3(base: u64, k1: u8, s1: u32, k2: u8, s2: u32, k3: u8, s3: u32) -> Operand {
    let mut o = Operand::at(base);
    o.stride[k1 as usize] = s1;
    o.stride[k2 as usize] = s2;
    o.stride[k3 as usize] = s3;
    o
}

fn ldc(v: i32) -> Instr {
    Instr { imm: v as u32, ..Instr::op(Opcode::Ldc) }
}

fn ld16(a: Operand) -> Instr {
    Instr { a, ..Instr::op(Opcode::Ld16) }
}

fn ld32(a: Operand) -> Instr {
    Instr { a, ..Instr::op(Opcode::Ld32) }
}

fn add32(a: Operand) -> Instr {
    Instr { a, ..Instr::op(Opcode::Add32) }
}

fn mul32(a: Operand) -> Instr {
    Instr { a, ..Instr::op(Opcode::Mul32) }
}

fn st32_acc(w: Operand) -> Instr {
    Instr { k: 0, w, ..Instr::op(Opcode::St32) }
}

fn st32_idx3(w: Operand) -> Instr {
    Instr { k: 5, w, ..Instr::op(Opcode::St32) } // k=5 stores idx[3]
}

fn ldidx(k: u8, a: Operand) -> Instr {
    Instr { k, a, ..Instr::op(Opcode::Ldidx) }
}

fn argmax(k: u8, a: Operand) -> Instr {
    Instr { k, a, ..Instr::op(Opcode::ArgmaxStep) }
}

fn fdot(a: Operand, b: Operand, w: Operand) -> Instr {
    Instr { imm: 64, a, b, w, ..Instr::op(Opcode::Fdot) }
}

fn fop(k: u8, a: Operand, b: Operand, w: Operand) -> Instr {
    Instr { k, a, b, w, ..Instr::op(Opcode::Fop) }
}

/// Copy a 4-byte cell (bit-exact, integer path).
fn copy4(em: &mut Em, from: Operand, to: Operand) {
    em.emit(ld32(from));
    em.emit(st32_acc(to));
}

/// The committed CEXP macro: in = t1 (clamped in place), out = t2,
/// staging t3/flag. EXACTLY mirrors fc::cexp_committed.
fn cexp(em: &mut Em, lay: &FLayout) {
    // clamp lo: if (−87 > t1) t1 ← −87
    em.emit(fop(8, at(lay.c_n87), at(lay.t1), at(lay.flag)));
    em.ifflag(lay.flag, |em| copy4(em, at(lay.c_n87), at(lay.t1)), 2);
    // clamp hi: if (t1 > 88) t1 ← 88
    em.emit(fop(8, at(lay.t1), at(lay.c_p88), at(lay.flag)));
    em.ifflag(lay.flag, |em| copy4(em, at(lay.c_p88), at(lay.t1)), 2);
    // kf in t3
    em.emit(fop(1, at(lay.t1), at(lay.c_log2e), at(lay.t3)));
    em.emit(fop(0, at(lay.t3), at(lay.c_half), at(lay.t3)));
    em.emit(fop(5, at(lay.t3), at(lay.c_izero), at(lay.t3))); // floor (B unused)
    // r in t1 (two fused steps; W is read+written)
    em.emit(fop(2, at(lay.t3), at(lay.c_nln2hi), at(lay.t1)));
    em.emit(fop(2, at(lay.t3), at(lay.c_nln2lo), at(lay.t1)));
    // two_k bits into t3: ftoi, then integer (ki+127)<<23
    em.emit(fop(6, at(lay.t3), at(lay.c_izero), at(lay.t3)));
    em.emit(ldc(127));
    em.emit(add32(at(lay.t3)));
    em.emit(mul32(at(lay.c_i2p23)));
    em.emit(st32_acc(at(lay.t3)));
    // poly in t2: seed c_exp6, then 6 × (mul r, add C)
    copy4(em, at(lay.c_exp6), at(lay.t2));
    for c in [lay.c_exp5, lay.c_exp4, lay.c_exp3, lay.c_half, lay.c_one, lay.c_one] {
        em.emit(fop(1, at(lay.t2), at(lay.t1), at(lay.t2)));
        em.emit(fop(0, at(lay.t2), at(c), at(lay.t2)));
    }
    // out: t2 ← t2 · two_k
    em.emit(fop(1, at(lay.t2), at(lay.t3), at(lay.t2)));
}

/// Committed RMS-norm src → dst (n elements), gamma at `gamma`:
/// EXACTLY mirrors fc::rmsnorm_committed.
fn rmsnorm(em: &mut Em, lay: &FLayout, src: u64, gamma: u64, dst: u64, n: u32, nf_cell: u64) {
    em.emit(ldc(0));
    em.emit(st32_acc(at(lay.facc)));
    em.loop_over(K1, n, |em| {
        em.emit(fop(2, st(src, K1, 4), st(src, K1, 4), at(lay.facc)));
    });
    em.emit(fop(3, at(lay.facc), at(nf_cell), at(lay.t1))); // mean
    em.emit(fop(0, at(lay.t1), at(lay.c_eps), at(lay.t1)));
    em.emit(fop(4, at(lay.t1), at(lay.c_izero), at(lay.t1))); // sqrt
    em.emit(fop(3, at(lay.c_one), at(lay.t1), at(lay.fr)));
    em.loop_over(K1, n, |em| {
        em.emit(fop(1, st(src, K1, 4), at(lay.fr), st(dst, K1, 4)));
        em.emit(fop(1, st(dst, K1, 4), st(gamma, K1, 4), st(dst, K1, 4)));
    });
}

/// Committed GEMV: bf16 matrix at `w` (rows × cols), input f32 line array
/// at `x` (256-aligned), each row's value lands via `store`. Row chain:
/// facc ← 0; cols/64 FDOTs; store reads facc.
fn gemv(
    em: &mut Em,
    lay: &FLayout,
    w: u64,
    rows: u32,
    cols: u64,
    x: u64,
    store: impl Fn(&mut Em),
) {
    let blocks = cols / 64;
    em.loop_over(K2, rows, |em| {
        em.emit(ldc(0));
        em.emit(st32_acc(at(lay.facc)));
        for b in 0..blocks {
            em.emit(fdot(
                st(w + 128 * b, K2, (cols * 2) as u32),
                at(x + 256 * b),
                at(lay.facc),
            ));
        }
        store(em);
    });
}

/// In-place per-head RMS-norm at `base + idx[K0]·(dh·4)`, then rope.
fn qknorm_rope(
    em: &mut Em,
    lay: &FLayout,
    base: u64,
    head_stride: u32,
    gamma: u64,
    dh: u32,
    pos: usize,
) {
    let cell = |off: u64| st2(base + off, K0, head_stride, K1, 4);
    // sum of squares (sequential ffma over the head)
    em.emit(ldc(0));
    em.emit(st32_acc(at(lay.facc)));
    em.loop_over(K1, dh, |em| {
        em.emit(fop(2, cell(0), cell(0), at(lay.facc)));
    });
    em.emit(fop(3, at(lay.facc), at(lay.c_dhf), at(lay.t1)));
    em.emit(fop(0, at(lay.t1), at(lay.c_eps), at(lay.t1)));
    em.emit(fop(4, at(lay.t1), at(lay.c_izero), at(lay.t1)));
    em.emit(fop(3, at(lay.c_one), at(lay.t1), at(lay.fr)));
    em.loop_over(K1, dh, |em| {
        em.emit(fop(1, cell(0), at(lay.fr), cell(0)));
        em.emit(fop(1, cell(0), st(gamma, K1, 4), cell(0)));
    });
    // rope (pairs in K1; tables static at pos)
    let half = dh as u64 / 2;
    let tcos = lay.rope_cos + pos as u64 * half * 4;
    let tsin = lay.rope_sin + pos as u64 * half * 4;
    let lo = st2(base, K0, head_stride, K1, 4);
    let hi = st2(base + half * 4, K0, head_stride, K1, 4);
    em.loop_over(K1, half as u32, |em| {
        em.emit(fop(1, hi, st(tsin, K1, 4), at(lay.t4))); // t = b·s
        em.emit(fop(1, at(lay.t4), at(lay.c_neg1f), at(lay.t4))); // −t
        copy4(em, at(lay.t4), at(lay.t1));
        em.emit(fop(2, lo, st(tcos, K1, 4), at(lay.t1))); // na = a·c + (−t)
        em.emit(fop(1, lo, st(tsin, K1, 4), at(lay.t4))); // a·s
        em.emit(fop(2, hi, st(tcos, K1, 4), at(lay.t4))); // nb = b·c + a·s
        copy4(em, at(lay.t4), hi);
        copy4(em, at(lay.t1), lo);
    });
}

pub fn compile_fqwen(
    lay: &FLayout,
    cfg: &QwenConfig,
    n_prompt: usize,
    n_gen: usize,
) -> FCompiled {
    assert!(n_prompt >= 1 && n_gen >= 1);
    let (h, dh, f) = (cfg.hidden_size as u64, cfg.head_dim as u64, cfg.intermediate_size as u64);
    let (nh, nkv) = (cfg.num_attention_heads as u64, cfg.num_key_value_heads as u64);
    let group = (nh / nkv) as u32;
    let vocab = cfg.vocab_size as u64;
    let n_pos = n_prompt + n_gen - 1;
    assert!(n_pos <= FMAX_SEQ);

    let mut em = Em { prog: Vec::new(), steps: 0, mult: 1, probes: Vec::new() };
    let mut token_boundaries = Vec::with_capacity(n_pos);
    let mut boundary_pcs = Vec::with_capacity(n_pos);

    for pos in 0..n_pos {
        let tok_addr =
            if pos < n_prompt { lay.input + 4 + 4 * pos as u64 } else { lay.tok };
        // E: embedding (bf16 widen via sext16·65536 + ST32 truncation)
        em.emit(ldidx(K0, at(tok_addr)));
        em.loop_over(K1, h as u32, |em| {
            em.emit(ld16(st2(lay.emb, K0, (h * 2) as u32, K1, 2)));
            em.emit(mul32(at(lay.c_i65536)));
            em.emit(st32_acc(st(lay.x, K1, 4)));
        });
        em.emit(ldidx(K0, at(lay.c_izero))); // LOOP needs zeroed registers
        em.probe(format!("f{pos} emb"));

        for (l, a) in lay.layers.iter().enumerate() {
            let krow = a.kc + pos as u64 * nkv * dh * 4;
            let vrow = a.vc + pos as u64 * nkv * dh * 4;
            rmsnorm(&mut em, lay, lay.x, a.ln1, lay.xn, h as u32, lay.c_hf);
            em.probe(format!("f{pos} l{l} n1"));
            gemv(&mut em, lay, a.wq, (nh * dh) as u32, h, lay.xn, |em| {
                copy4(em, at(lay.facc), st(lay.q, K2, 4));
            });
            gemv(&mut em, lay, a.wk, (nkv * dh) as u32, h, lay.xn, |em| {
                copy4(em, at(lay.facc), st(krow, K2, 4));
            });
            gemv(&mut em, lay, a.wv, (nkv * dh) as u32, h, lay.xn, |em| {
                copy4(em, at(lay.facc), st(vrow, K2, 4));
            });
            em.probe(format!("f{pos} l{l} qkv"));
            em.loop_over(K0, nh as u32, |em| {
                qknorm_rope(em, lay, lay.q, (dh * 4) as u32, a.q_norm, dh as u32, pos);
            });
            em.loop_over(K0, nkv as u32, |em| {
                qknorm_rope(em, lay, krow, (dh * 4) as u32, a.k_norm, dh as u32, pos);
            });
            em.probe(format!("f{pos} l{l} qkrope"));

            // ATT: kv-head (K0) × group member (K1)
            let t1pos = (pos + 1) as u32;
            em.loop_over(K0, nkv as u32, |em| {
                em.loop_over(K1, group, |em| {
                    // scores
                    em.loop_over(K2, t1pos, |em| {
                        em.emit(ldc(0));
                        em.emit(st32_acc(at(lay.facc)));
                        em.loop_over(K3, dh as u32, |em| {
                            em.emit(fop(
                                2,
                                st3(lay.q, K0, (group as u64 * dh * 4) as u32, K1, (dh * 4) as u32, K3, 4),
                                st3(a.kc, K2, (nkv * dh * 4) as u32, K0, (dh * 4) as u32, K3, 4),
                                at(lay.facc),
                            ));
                        });
                        em.emit(fop(1, at(lay.facc), at(lay.c_isqdh), st(lay.scores, K2, 4)));
                    });
                    // max scan
                    copy4(em, at(lay.scores), at(lay.fm));
                    em.loop_over(K2, t1pos, |em| {
                        em.emit(fop(8, st(lay.scores, K2, 4), at(lay.fm), at(lay.flag)));
                        em.ifflag(lay.flag, |em| copy4(em, st(lay.scores, K2, 4), at(lay.fm)), 2);
                    });
                    // exps + sum
                    em.emit(ldc(0));
                    em.emit(st32_acc(at(lay.fsum)));
                    em.loop_over(K2, t1pos, |em| {
                        em.emit(fop(1, at(lay.fm), at(lay.c_neg1f), at(lay.t1)));
                        em.emit(fop(0, st(lay.scores, K2, 4), at(lay.t1), at(lay.t1)));
                        cexp(em, lay);
                        copy4(em, at(lay.t2), st(lay.scores, K2, 4));
                        em.emit(fop(0, at(lay.t2), at(lay.fsum), at(lay.fsum)));
                    });
                    // normalize
                    em.loop_over(K2, t1pos, |em| {
                        em.emit(fop(3, st(lay.scores, K2, 4), at(lay.fsum), st(lay.scores, K2, 4)));
                    });
                    // ctx into attnx
                    em.loop_over(K2, dh as u32, |em| {
                        let cell = st3(lay.attnx, K0, (group as u64 * dh * 4) as u32, K1, (dh * 4) as u32, K2, 4);
                        em.emit(ldc(0));
                        em.emit(st32_acc(cell));
                        em.loop_over(K3, t1pos, |em| {
                            em.emit(fop(
                                2,
                                st(lay.scores, K3, 4),
                                st3(a.vc, K3, (nkv * dh * 4) as u32, K0, (dh * 4) as u32, K2, 4),
                                cell,
                            ));
                        });
                    });
                });
            });
            em.probe(format!("f{pos} l{l} att"));
            // O-proj + residual
            gemv(&mut em, lay, a.wo, h as u32, nh * dh, lay.attnx, |em| {
                em.emit(fop(0, st(lay.x, K2, 4), at(lay.facc), st(lay.x, K2, 4)));
            });
            // FFN
            rmsnorm(&mut em, lay, lay.x, a.ln2, lay.xn, h as u32, lay.c_hf);
            gemv(&mut em, lay, a.w_gate, f as u32, h, lay.xn, |em| {
                copy4(em, at(lay.facc), st(lay.gate, K2, 4));
            });
            gemv(&mut em, lay, a.w_up, f as u32, h, lay.xn, |em| {
                copy4(em, at(lay.facc), st(lay.up, K2, 4));
            });
            em.probe(format!("f{pos} l{l} gateup"));
            em.loop_over(K2, f as u32, |em| {
                em.emit(fop(1, st(lay.gate, K2, 4), at(lay.c_neg1f), at(lay.t1)));
                cexp(em, lay);
                em.emit(fop(0, at(lay.t2), at(lay.c_one), at(lay.t2)));
                em.emit(fop(3, st(lay.gate, K2, 4), at(lay.t2), at(lay.t3)));
                em.emit(fop(1, at(lay.t3), st(lay.up, K2, 4), st(lay.h_ffn, K2, 4)));
            });
            gemv(&mut em, lay, a.w_down, h as u32, f, lay.h_ffn, |em| {
                em.emit(fop(0, st(lay.x, K2, 4), at(lay.facc), st(lay.x, K2, 4)));
            });
            em.probe(format!("f{pos} l{l} ffn"));
        }

        if pos >= n_prompt - 1 {
            rmsnorm(&mut em, lay, lay.x, lay.ln_f, lay.xn, h as u32, lay.c_hf);
            em.emit(Instr { imm: 0xFF80_0000, ..Instr::op(Opcode::Ldc) }); // −inf bits
            em.emit(st32_acc(at(lay.saved_max)));
            em.emit(ldc(0));
            em.emit(st32_acc(at(lay.v_cell)));
            let blocks = h / 64;
            let row_body = |em: &mut Em, w_base: u64, k2_strided: bool| {
                em.emit(ldidx(K3, at(lay.v_cell)));
                em.emit(ldc(0));
                em.emit(st32_acc(at(lay.facc)));
                for b in 0..blocks {
                    let wa = if k2_strided {
                        st2(w_base + 128 * b, K2, (CHUNK * h * 2) as u32, K1, (h * 2) as u32)
                    } else {
                        st(w_base + 128 * b, K1, (h * 2) as u32)
                    };
                    em.emit(fdot(wa, at(lay.xn + 256 * b), at(lay.facc)));
                }
                copy4(em, at(lay.facc), st(lay.logit_buf, K1, 4));
                em.emit(fop(8, st(lay.logit_buf, K1, 4), at(lay.saved_max), at(lay.flag)));
                em.ifflag(
                    lay.flag,
                    |em| {
                        copy4(em, st(lay.logit_buf, K1, 4), at(lay.saved_max));
                        em.emit(st32_idx3(at(lay.tok)));
                    },
                    3,
                );
                em.emit(ldc(1));
                em.emit(add32(at(lay.v_cell)));
                em.emit(st32_acc(at(lay.v_cell)));
            };
            let full_chunks = (vocab / CHUNK) as u32;
            let tail = (vocab % CHUNK) as u32;
            em.loop_over(K2, full_chunks, |em| {
                em.loop_over(K1, CHUNK as u32, |em| row_body(em, lay.emb, true));
            });
            if tail > 0 {
                let base_rows = full_chunks as u64 * CHUNK;
                em.loop_over(K1, tail, |em| {
                    row_body(em, lay.emb + base_rows * h * 2, false)
                });
            }
            em.probe(format!("f{pos} head"));
        }

        // canonicalizing tail
        em.emit(ldc(i32::MIN));
        em.emit(argmax(K1, at(lay.c_izero)));
        em.emit(ldidx(K3, at(lay.c_izero)));
        token_boundaries.push(em.steps);
        boundary_pcs.push(em.pc());
    }
    em.emit(Instr::op(Opcode::Halt));

    let len = em.prog.len() as u64;
    let p = (0..=vm::MAX_PROG_DEPTH).find(|&p| len <= 1u64 << p).unwrap();
    FCompiled {
        program: em.prog,
        p,
        probes: em.probes,
        total_steps: em.steps,
        token_boundaries,
        boundary_pcs,
        n_prompt,
        n_gen,
    }
}
