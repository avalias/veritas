//! One-step fraud verification (SPEC §8.4) — the Move contract's twin.
//!
//! `verify_step` is a DELIBERATELY independent implementation of the step
//! relation: it executes one micro-op from Merkle openings alone, never
//! touching a `Machine`. The Phase 2 Move contract is a line-by-line port
//! of this file, and the property test in `tests/onestep.rs` (every verdict
//! matches `Machine::step` across randomized states, valid and trapping)
//! is the local stand-in for the Rust↔Move equivalence suite — the single
//! most important test in the project (brief Phase 2.3).
//!
//! Verdict logic is submitter-independent: either party may submit; the
//! comparison decides (SPEC §8.3).

use crate::exec::{rnd, sat16, sat8, sext16, sext32, sext8, trunc_div};
use crate::hash::{page_leaf_hash, prog_leaf_hash, state_root, Hash};
use crate::isa::{Instr, Opcode, Operand, INSTR_ENC_LEN};
use crate::merkle::{fold_proof, MerkleTree};
use crate::state::{Registers, REG_ENC_LEN, TRAPPED};
use crate::{DOT_LINE, PAGE_SIZE};

/// The on-chain judge parameters a dispute is verified against (SPEC §8.1).
#[derive(Clone, Debug)]
pub struct JudgeParams {
    pub d: u8,
    pub p: u8,
    pub program_root: Hash,
}

#[derive(Clone, Debug)]
pub struct PageOpening {
    pub page: Vec<u8>, // exactly PAGE_SIZE bytes
    pub siblings: Vec<Hash>,
}

/// Everything `verify_step` consumes (SPEC §8.4 payload).
#[derive(Clone, Debug)]
pub struct StepProof {
    pub regs: [u8; REG_ENC_LEN],
    pub mem_root: Hash,
    /// Instruction bytes + program-tree opening at index `regs.pc`.
    /// Ignored (may be empty) when pc ≥ 2^p — V3 traps without a fetch.
    pub instr: [u8; INSTR_ENC_LEN],
    pub instr_siblings: Vec<Hash>,
    pub open_a: Option<PageOpening>,
    pub open_b: Option<PageOpening>,
    pub open_w: Option<PageOpening>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    ResolverWins,
    ChallengerWins,
}

/// Malformed proof ⇒ transaction aborts, state unchanged (SPEC §8.2):
/// distinct from a verdict — garbage neither wins nor stalls the clock.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProofError {
    PreRootMismatch,
    BadInstrProof,
    MissingOpening,
    BadOpening,
}

/// Effective address (SPEC §4.2) — pure twin of `Machine::ea`.
fn ea(regs: &Registers, o: &Operand) -> u64 {
    let mut a = o.base;
    for j in 0..4 {
        a = a.wrapping_add((regs.idx[j] as u64).wrapping_mul(o.stride[j] as u64));
    }
    a
}

#[allow(clippy::manual_is_multiple_of)] // spec-literal form (SPEC §4.4 T3)
fn ok_access(ea: u64, size: u64, mem_bytes: u64) -> bool {
    ea % size == 0 && ea <= mem_bytes - size
}

#[allow(clippy::manual_is_multiple_of)] // spec-literal form (SPEC §4.4 T7)
fn ok_line(ea: u64, mem_bytes: u64) -> bool {
    ea % DOT_LINE as u64 == 0 && ea <= mem_bytes - DOT_LINE as u64
}

/// T7 wide variant (SPEC §5.2 DOT8X16/DOTBM): 128-byte i16 line. 128 | 1024
/// so an aligned wide line never straddles a page — one opening suffices.
#[allow(clippy::manual_is_multiple_of)]
fn ok_line128(ea: u64, mem_bytes: u64) -> bool {
    ea % 128 == 0 && ea <= mem_bytes - 128
}

fn trap_regs(r: &Registers) -> Registers {
    Registers {
        halted: TRAPPED,
        // Wrapping: a CRAFTED pre-state may carry step = u64::MAX; the
        // verifier must stay total (no panic/abort) on adversarial states.
        step: r.step.wrapping_add(1),
        ..*r
    }
}

/// Verify an opening against `mem_root` and return the page bytes.
fn open<'a>(
    slot: &'a Option<PageOpening>,
    page_index: u64,
    d: u8,
    mem_root: &Hash,
) -> Result<&'a [u8], ProofError> {
    let o = slot.as_ref().ok_or(ProofError::MissingOpening)?;
    if o.page.len() != PAGE_SIZE || o.siblings.len() != d as usize {
        return Err(ProofError::BadOpening);
    }
    if fold_proof(page_leaf_hash(&o.page), page_index, &o.siblings) != *mem_root {
        return Err(ProofError::BadOpening);
    }
    Ok(&o.page)
}

fn page_of(a: u64) -> u64 {
    a / PAGE_SIZE as u64
}

fn off_of(a: u64) -> usize {
    (a % PAGE_SIZE as u64) as usize
}

fn read_n(page: &[u8], a: u64, n: usize) -> &[u8] {
    &page[off_of(a)..off_of(a) + n]
}

fn read_u32_at(page: &[u8], a: u64) -> u32 {
    u32::from_le_bytes(read_n(page, a, 4).try_into().unwrap())
}

/// SPEC §8.4, checks V1–V9. `pre_root` is the agreed root at `lo`;
/// `claimed_post` is the disputed root at `lo + 1`.
pub fn verify_step(
    pre_root: &Hash,
    claimed_post: &Hash,
    judge: &JudgeParams,
    proof: &StepProof,
) -> Result<Verdict, ProofError> {
    let mem_bytes = (1u64 << judge.d) * PAGE_SIZE as u64;

    // V1: the revealed (mem_root, regs) must BE the agreed pre-state.
    if state_root(&proof.mem_root, &proof.regs) != *pre_root {
        return Err(ProofError::PreRootMismatch);
    }
    let regs = Registers::decode(&proof.regs);

    // V2: terminality — halted/trapped states have no successor.
    if regs.halted != 0 {
        return Ok(Verdict::ChallengerWins);
    }

    let verdict = |post_mem_root: Hash, post_regs: Registers| {
        if state_root(&post_mem_root, &post_regs.encode()) == *claimed_post {
            Verdict::ResolverWins
        } else {
            Verdict::ChallengerWins
        }
    };
    let trap = |r: &Registers| Ok(verdict(proof.mem_root, trap_regs(r)));

    // V3: T1 — pc outside the program tree; trap without any opening.
    if (regs.pc as u64) >= (1u64 << judge.p) {
        return trap(&regs);
    }

    // V4: fetch — instruction inclusion at index pc.
    if proof.instr_siblings.len() != judge.p as usize
        || fold_proof(prog_leaf_hash(&proof.instr), regs.pc as u64, &proof.instr_siblings)
            != judge.program_root
    {
        return Err(ProofError::BadInstrProof);
    }
    let instr = Instr::decode(&proof.instr);
    let op = match Opcode::from_u8(instr.opcode) {
        Some(o) => o,
        None => return trap(&regs), // T2
    };

    // V5–V7: per-op. `post` starts as pc+1/step+1 and arms adjust.
    let mut post = Registers {
        pc: regs.pc.wrapping_add(1),
        step: regs.step.wrapping_add(1),
        ..regs
    };
    let mut write: Option<(u64, Vec<u8>)> = None; // (ea, bytes)

    match op {
        Opcode::Mac8 | Opcode::Mac16 => {
            let size = if op == Opcode::Mac8 { 1u64 } else { 2 };
            let (ea_a, ea_b) = (ea(&regs, &instr.a), ea(&regs, &instr.b));
            if !ok_access(ea_a, size, mem_bytes) || !ok_access(ea_b, size, mem_bytes) {
                return trap(&regs);
            }
            let pa = open(&proof.open_a, page_of(ea_a), judge.d, &proof.mem_root)?;
            let pb = open(&proof.open_b, page_of(ea_b), judge.d, &proof.mem_root)?;
            let (av, bv) = if op == Opcode::Mac8 {
                (sext8(read_n(pa, ea_a, 1)[0]), sext8(read_n(pb, ea_b, 1)[0]))
            } else {
                (
                    sext16(u16::from_le_bytes(read_n(pa, ea_a, 2).try_into().unwrap())),
                    sext16(u16::from_le_bytes(read_n(pb, ea_b, 2).try_into().unwrap())),
                )
            };
            post.acc = regs.acc.wrapping_add(av.wrapping_mul(bv));
        }
        Opcode::Dot8 | Opcode::Dot16 => {
            let cap = if op == Opcode::Dot8 { 64 } else { 32 };
            if instr.imm == 0 || instr.imm > cap {
                return trap(&regs); // T7 lanes
            }
            let (ea_a, ea_b) = (ea(&regs, &instr.a), ea(&regs, &instr.b));
            if !ok_line(ea_a, mem_bytes) || !ok_line(ea_b, mem_bytes) {
                return trap(&regs); // T7 line
            }
            let pa = open(&proof.open_a, page_of(ea_a), judge.d, &proof.mem_root)?;
            let pb = open(&proof.open_b, page_of(ea_b), judge.d, &proof.mem_root)?;
            let lanes = instr.imm as usize;
            let mut acc = regs.acc;
            if op == Opcode::Dot8 {
                let a = read_n(pa, ea_a, lanes);
                let b = read_n(pb, ea_b, lanes);
                for j in 0..lanes {
                    acc = acc.wrapping_add(sext8(a[j]).wrapping_mul(sext8(b[j])));
                }
            } else {
                let a = read_n(pa, ea_a, 2 * lanes);
                let b = read_n(pb, ea_b, 2 * lanes);
                for j in 0..lanes {
                    let av = sext16(u16::from_le_bytes([a[2 * j], a[2 * j + 1]]));
                    let bv = sext16(u16::from_le_bytes([b[2 * j], b[2 * j + 1]]));
                    acc = acc.wrapping_add(av.wrapping_mul(bv));
                }
            }
            post.acc = acc;
        }
        Opcode::Fdot => {
            // FW-6 committed block dot (see exec.rs): imm must be 64; A a
            // 128-aligned 128-byte bf16 line, B a 256-aligned 256-byte f32
            // line, W a 4-aligned f32 accumulator cell (read + WRITE).
            if instr.imm != 64 {
                return trap(&regs); // T7
            }
            let (ea_a, ea_b, ea_w) =
                (ea(&regs, &instr.a), ea(&regs, &instr.b), ea(&regs, &instr.w));
            if ea_a % 128 != 0
                || ea_a > mem_bytes - 128
                || ea_b % 256 != 0
                || ea_b > mem_bytes - 256
                || !ok_access(ea_w, 4, mem_bytes)
            {
                return trap(&regs);
            }
            let pa = open(&proof.open_a, page_of(ea_a), judge.d, &proof.mem_root)?;
            let pb = open(&proof.open_b, page_of(ea_b), judge.d, &proof.mem_root)?;
            let pw = open(&proof.open_w, page_of(ea_w), judge.d, &proof.mem_root)?;
            let wb = read_n(pa, ea_a, 128);
            let xb = read_n(pb, ea_b, 256);
            let mut w16 = [0u16; 64];
            let mut x32 = [0u32; 64];
            for j in 0..64 {
                w16[j] = u16::from_le_bytes([wb[2 * j], wb[2 * j + 1]]);
                x32[j] = u32::from_le_bytes([
                    xb[4 * j],
                    xb[4 * j + 1],
                    xb[4 * j + 2],
                    xb[4 * j + 3],
                ]);
            }
            let cell = read_u32_at(pw, ea_w);
            let out =
                crate::softfloat::fadd(cell, crate::softfloat::block_dot_bf16(&w16, &x32));
            write = Some((ea_w, out.to_le_bytes().to_vec()));
        }
        Opcode::Fop => {
            if instr.k > 7 {
                return trap(&regs); // T6
            }
            let (ea_a, ea_b, ea_w) =
                (ea(&regs, &instr.a), ea(&regs, &instr.b), ea(&regs, &instr.w));
            let binary = instr.k <= 3;
            if !ok_access(ea_a, 4, mem_bytes)
                || (binary && !ok_access(ea_b, 4, mem_bytes))
                || !ok_access(ea_w, 4, mem_bytes)
            {
                return trap(&regs);
            }
            let pa = open(&proof.open_a, page_of(ea_a), judge.d, &proof.mem_root)?;
            let a = read_u32_at(pa, ea_a);
            use crate::softfloat as sf;
            let out = match instr.k {
                0 | 1 | 3 => {
                    let pb = open(&proof.open_b, page_of(ea_b), judge.d, &proof.mem_root)?;
                    let b = read_u32_at(pb, ea_b);
                    match instr.k {
                        0 => sf::fadd(a, b),
                        1 => sf::fmul(a, b),
                        _ => sf::fdiv(a, b),
                    }
                }
                2 => {
                    let pb = open(&proof.open_b, page_of(ea_b), judge.d, &proof.mem_root)?;
                    let b = read_u32_at(pb, ea_b);
                    let pw = open(&proof.open_w, page_of(ea_w), judge.d, &proof.mem_root)?;
                    let c = read_u32_at(pw, ea_w);
                    sf::ffma(a, b, c)
                }
                4 => sf::fsqrt(a),
                5 => sf::ffloor(a),
                6 => sf::ftoi(a) as u32,
                _ => sf::itof(a as i32),
            };
            write = Some((ea_w, out.to_le_bytes().to_vec()));
        }
        Opcode::Ld8 => {
            let ea_a = ea(&regs, &instr.a);
            if !ok_access(ea_a, 1, mem_bytes) {
                return trap(&regs);
            }
            let pa = open(&proof.open_a, page_of(ea_a), judge.d, &proof.mem_root)?;
            post.acc = sext8(read_n(pa, ea_a, 1)[0]);
        }
        Opcode::Ld16 => {
            let ea_a = ea(&regs, &instr.a);
            if !ok_access(ea_a, 2, mem_bytes) {
                return trap(&regs);
            }
            let pa = open(&proof.open_a, page_of(ea_a), judge.d, &proof.mem_root)?;
            post.acc = sext16(u16::from_le_bytes(read_n(pa, ea_a, 2).try_into().unwrap()));
        }
        Opcode::Dot8x16 | Opcode::Dotbm => {
            if instr.imm == 0 || instr.imm > 64 {
                return trap(&regs); // T7 lanes
            }
            let (ea_a, ea_b) = (ea(&regs, &instr.a), ea(&regs, &instr.b));
            if !ok_line(ea_a, mem_bytes) || !ok_line128(ea_b, mem_bytes) {
                return trap(&regs); // T7 line
            }
            // DOTBM: the W slot is a READ — the per-block multiplier cell
            // (the one ISA asymmetry, SPEC §5.2).
            let ea_w = ea(&regs, &instr.w);
            if op == Opcode::Dotbm && !ok_access(ea_w, 4, mem_bytes) {
                return trap(&regs);
            }
            let pa = open(&proof.open_a, page_of(ea_a), judge.d, &proof.mem_root)?;
            let pb = open(&proof.open_b, page_of(ea_b), judge.d, &proof.mem_root)?;
            let lanes = instr.imm as usize;
            let a = read_n(pa, ea_a, lanes);
            let b = read_n(pb, ea_b, 2 * lanes);
            let mut part = 0i64;
            for j in 0..lanes {
                let av = sext8(a[j]);
                let bv = sext16(u16::from_le_bytes([b[2 * j], b[2 * j + 1]]));
                part = part.wrapping_add(av.wrapping_mul(bv));
            }
            post.acc = if op == Opcode::Dotbm {
                let pw = open(&proof.open_w, page_of(ea_w), judge.d, &proof.mem_root)?;
                let mv = sext32(read_u32_at(pw, ea_w));
                regs.acc.wrapping_add(part.wrapping_mul(mv))
            } else {
                regs.acc.wrapping_add(part)
            };
        }
        Opcode::Ld32 | Opcode::Add32 | Opcode::Mul32 | Opcode::Div32 => {
            let ea_a = ea(&regs, &instr.a);
            if !ok_access(ea_a, 4, mem_bytes) {
                return trap(&regs);
            }
            let pa = open(&proof.open_a, page_of(ea_a), judge.d, &proof.mem_root)?;
            let v = sext32(read_u32_at(pa, ea_a));
            match op {
                Opcode::Ld32 => post.acc = v,
                Opcode::Add32 => post.acc = regs.acc.wrapping_add(v),
                Opcode::Mul32 => post.acc = regs.acc.wrapping_mul(v),
                Opcode::Div32 => {
                    if v <= 0 {
                        return trap(&regs); // T5
                    }
                    post.acc = trunc_div(regs.acc, v);
                }
                _ => unreachable!(),
            }
        }
        Opcode::Ldc => post.acc = sext32(instr.imm),
        Opcode::ShiftRndn => {
            if instr.s > 63 {
                return trap(&regs); // T4
            }
            post.acc = rnd(regs.acc, instr.s);
        }
        Opcode::Clamp8 | Opcode::Clamp16 => {
            let size = if op == Opcode::Clamp8 { 1u64 } else { 2 };
            let ea_w = ea(&regs, &instr.w);
            if !ok_access(ea_w, size, mem_bytes) {
                return trap(&regs);
            }
            let bytes = if op == Opcode::Clamp8 {
                sat8(regs.acc).to_le_bytes().to_vec()
            } else {
                sat16(regs.acc).to_le_bytes().to_vec()
            };
            write = Some((ea_w, bytes));
        }
        Opcode::Lut16 => {
            let index = (sat16(regs.acc) as i64 + 32768) as u64;
            let ea_t = instr.a.base.wrapping_add(2 * index);
            if !ok_access(ea_t, 2, mem_bytes) {
                return trap(&regs);
            }
            let pa = open(&proof.open_a, page_of(ea_t), judge.d, &proof.mem_root)?;
            post.acc = sext16(u16::from_le_bytes(read_n(pa, ea_t, 2).try_into().unwrap()));
        }
        Opcode::St32 => {
            let src: u32 = match instr.k {
                0 => regs.acc as u32,
                1 => regs.aux as u32,
                2..=5 => regs.idx[(instr.k - 2) as usize],
                _ => return trap(&regs), // T6
            };
            let ea_w = ea(&regs, &instr.w);
            if !ok_access(ea_w, 4, mem_bytes) {
                return trap(&regs);
            }
            write = Some((ea_w, src.to_le_bytes().to_vec()));
        }
        Opcode::Ldidx => {
            if instr.k > 3 {
                return trap(&regs); // T6
            }
            let ea_a = ea(&regs, &instr.a);
            if !ok_access(ea_a, 4, mem_bytes) {
                return trap(&regs);
            }
            let pa = open(&proof.open_a, page_of(ea_a), judge.d, &proof.mem_root)?;
            post.idx[instr.k as usize] = read_u32_at(pa, ea_a);
        }
        Opcode::ArgmaxStep | Opcode::ArgmaxOff => {
            if instr.k > 3 {
                return trap(&regs); // T6
            }
            let ea_a = ea(&regs, &instr.a);
            if !ok_access(ea_a, 4, mem_bytes) {
                return trap(&regs);
            }
            let pa = open(&proof.open_a, page_of(ea_a), judge.d, &proof.mem_root)?;
            let v = sext32(read_u32_at(pa, ea_a));
            if v > regs.acc {
                post.acc = v;
                // ARGMAX_OFF: aux ← imm +w idx[k] (SPEC §5.2 streaming head).
                let base = if op == Opcode::ArgmaxOff { instr.imm as u64 } else { 0 };
                post.aux = base.wrapping_add(regs.idx[instr.k as usize] as u64) as i64;
            }
        }
        Opcode::Jmp => post.pc = instr.target,
        Opcode::Jeq => {
            let ea_a = ea(&regs, &instr.a);
            if !ok_access(ea_a, 4, mem_bytes) {
                return trap(&regs);
            }
            let pa = open(&proof.open_a, page_of(ea_a), judge.d, &proof.mem_root)?;
            if read_u32_at(pa, ea_a) == instr.imm {
                post.pc = instr.target;
            }
        }
        Opcode::Loop => {
            if instr.k > 3 {
                return trap(&regs); // T6
            }
            let k = instr.k as usize;
            let nxt = regs.idx[k].wrapping_add(1);
            if nxt < instr.imm {
                post.idx[k] = nxt;
                post.pc = instr.target;
            } else {
                post.idx[k] = 0;
            }
        }
        Opcode::Halt => {
            post.pc = regs.pc; // pc unchanged
            post.halted = crate::state::HALTED;
        }
    }

    // V7 write application + V8 post-root.
    let post_mem_root = if let Some((ea_w, bytes)) = write {
        let pw = open(&proof.open_w, page_of(ea_w), judge.d, &proof.mem_root)?;
        let mut page = pw.to_vec();
        page[off_of(ea_w)..off_of(ea_w) + bytes.len()].copy_from_slice(&bytes);
        // Re-fold the modified leaf with the SAME siblings (SPEC §3.4).
        fold_proof(
            page_leaf_hash(&page),
            page_of(ea_w),
            &proof.open_w.as_ref().unwrap().siblings,
        )
    } else {
        proof.mem_root
    };

    // V9.
    Ok(verdict(post_mem_root, post))
}

// ---------------------------------------------------------------------------
// Proof building (off-chain side)
// ---------------------------------------------------------------------------

/// Merkle tree over instruction leaves (SPEC §3.5). Dense — fine for the
/// program sizes of Phases 1–3 (≤ 2^20 instructions).
pub struct ProgramTree {
    tree: MerkleTree,
}

impl ProgramTree {
    pub fn new(program: &[Instr], p: u8) -> Self {
        assert!(p <= 20, "dense program tree capped at 2^20 leaves");
        let leaves = program.iter().map(|i| prog_leaf_hash(&i.encode())).collect();
        let pad = prog_leaf_hash(&[0u8; INSTR_ENC_LEN]); // zero-instruction padding
        Self {
            tree: MerkleTree::from_leaf_hashes(p, leaves, pad),
        }
    }

    pub fn root(&self) -> Hash {
        self.tree.root()
    }
}

/// Build the §8.4 payload from a live machine paused at the pre-state.
/// Mirrors the verifier's own structural checks to decide which openings
/// the op will consume (a trapping access needs no opening).
pub fn build_step_proof(m: &crate::exec::Machine, prog_tree: &ProgramTree) -> StepProof {
    let regs = m.regs;
    let mem_bytes = m.mem.mem_bytes();
    let mut proof = StepProof {
        regs: regs.encode(),
        mem_root: m.mem.root(),
        instr: [0u8; INSTR_ENC_LEN],
        instr_siblings: vec![],
        open_a: None,
        open_b: None,
        open_w: None,
    };
    if regs.halted != 0 || (regs.pc as u64) >= (1u64 << m.p) {
        return proof; // V2 / V3 — nothing else consumed
    }
    let instr = m.program.get(regs.pc as usize).copied().unwrap_or_else(Instr::zero);
    proof.instr = instr.encode();
    proof.instr_siblings = prog_tree.tree.prove(regs.pc as u64);

    let opening = |addr: u64| -> Option<PageOpening> {
        Some(PageOpening {
            page: m.mem.page(page_of(addr)).to_vec(),
            siblings: m.mem.prove_page(page_of(addr)),
        })
    };
    let op = match Opcode::from_u8(instr.opcode) {
        Some(o) => o,
        None => return proof, // T2 — no openings
    };
    let (ea_a, ea_b, ea_w) = (ea(&regs, &instr.a), ea(&regs, &instr.b), ea(&regs, &instr.w));
    match op {
        Opcode::Mac8 => {
            if ok_access(ea_a, 1, mem_bytes) && ok_access(ea_b, 1, mem_bytes) {
                proof.open_a = opening(ea_a);
                proof.open_b = opening(ea_b);
            }
        }
        Opcode::Mac16 => {
            if ok_access(ea_a, 2, mem_bytes) && ok_access(ea_b, 2, mem_bytes) {
                proof.open_a = opening(ea_a);
                proof.open_b = opening(ea_b);
            }
        }
        Opcode::Dot8 | Opcode::Dot16 => {
            let cap = if op == Opcode::Dot8 { 64 } else { 32 };
            if instr.imm >= 1
                && instr.imm <= cap
                && ok_line(ea_a, mem_bytes)
                && ok_line(ea_b, mem_bytes)
            {
                proof.open_a = opening(ea_a);
                proof.open_b = opening(ea_b);
            }
        }
        Opcode::Fdot => {
            if instr.imm == 64
                && ea_a % 128 == 0
                && ea_a <= mem_bytes - 128
                && ea_b % 256 == 0
                && ea_b <= mem_bytes - 256
                && ok_access(ea_w, 4, mem_bytes)
            {
                proof.open_a = opening(ea_a);
                proof.open_b = opening(ea_b);
                proof.open_w = opening(ea_w); // read + write
            }
        }
        Opcode::Fop => {
            let binary = instr.k <= 3;
            if instr.k <= 7
                && ok_access(ea_a, 4, mem_bytes)
                && (!binary || ok_access(ea_b, 4, mem_bytes))
                && ok_access(ea_w, 4, mem_bytes)
            {
                proof.open_a = opening(ea_a);
                if binary {
                    proof.open_b = opening(ea_b);
                }
                proof.open_w = opening(ea_w);
            }
        }
        Opcode::Ld8 => {
            if ok_access(ea_a, 1, mem_bytes) {
                proof.open_a = opening(ea_a);
            }
        }
        Opcode::Ld16 => {
            if ok_access(ea_a, 2, mem_bytes) {
                proof.open_a = opening(ea_a);
            }
        }
        Opcode::Dot8x16 | Opcode::Dotbm => {
            if instr.imm >= 1
                && instr.imm <= 64
                && ok_line(ea_a, mem_bytes)
                && ok_line128(ea_b, mem_bytes)
                && (op == Opcode::Dot8x16 || ok_access(ea_w, 4, mem_bytes))
            {
                proof.open_a = opening(ea_a);
                proof.open_b = opening(ea_b);
                if op == Opcode::Dotbm {
                    proof.open_w = opening(ea_w); // READ opening (multiplier)
                }
            }
        }
        Opcode::Ld32 | Opcode::Add32 | Opcode::Mul32 | Opcode::Div32 | Opcode::Jeq => {
            if ok_access(ea_a, 4, mem_bytes) {
                proof.open_a = opening(ea_a);
            }
        }
        Opcode::Ldidx | Opcode::ArgmaxStep | Opcode::ArgmaxOff => {
            if instr.k <= 3 && ok_access(ea_a, 4, mem_bytes) {
                proof.open_a = opening(ea_a);
            }
        }
        Opcode::Lut16 => {
            let ea_t = instr.a.base.wrapping_add(2 * (sat16(regs.acc) as i64 + 32768) as u64);
            if ok_access(ea_t, 2, mem_bytes) {
                proof.open_a = opening(ea_t);
            }
        }
        Opcode::Clamp8 => {
            if ok_access(ea_w, 1, mem_bytes) {
                proof.open_w = opening(ea_w);
            }
        }
        Opcode::Clamp16 => {
            if ok_access(ea_w, 2, mem_bytes) {
                proof.open_w = opening(ea_w);
            }
        }
        Opcode::St32 => {
            if instr.k <= 5 && ok_access(ea_w, 4, mem_bytes) {
                proof.open_w = opening(ea_w);
            }
        }
        Opcode::Ldc | Opcode::ShiftRndn | Opcode::Jmp | Opcode::Loop | Opcode::Halt => {}
    }
    proof
}
