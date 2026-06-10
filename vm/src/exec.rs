//! Reference interpreter: micro-op step relation (SPEC §5) and the numeric
//! helpers shared with future Move cross-tests (SPEC §5.1, §9.3).
//!
//! This is the per-step conformance oracle (SPEC §9.1): single-threaded,
//! fixed order, every transition total — any adversarial state either traps
//! or steps deterministically.

use crate::hash::Hash;
use crate::isa::{Instr, Opcode, Operand};
use crate::state::{CommittedMemory, Registers, HALTED, RUNNING, TRAPPED};
use crate::DOT_LINE;

// ---------------------------------------------------------------------------
// Numeric helpers (SPEC §5.1) — normative, cross-tested against Move later.
// ---------------------------------------------------------------------------

pub fn sext8(b: u8) -> i64 {
    b as i8 as i64
}

pub fn sext16(v: u16) -> i64 {
    v as i16 as i64
}

pub fn sext32(v: u32) -> i64 {
    v as i32 as i64
}

/// Saturating requantization to i8 (SPEC §5.1): stores saturate, never wrap.
pub fn sat8(x: i64) -> i8 {
    x.clamp(-128, 127) as i8
}

pub fn sat16(x: i64) -> i16 {
    x.clamp(-32768, 32767) as i16
}

/// THE rounding rule (SPEC §5.1): arithmetic shift right by `s` with
/// round-half-to-even. Single rule for the whole system.
pub fn rnd(x: i64, s: u8) -> i64 {
    if s == 0 {
        return x;
    }
    debug_assert!(s <= 63, "caller must trap T4 first");
    let q = x >> s; // arithmetic shift: floor(x / 2^s)
    // r = x − q·2^s ∈ [0, 2^s). q·2^s ≤ x with the same sign behavior, so
    // the wrapping subtraction yields the exact nonnegative remainder even
    // at the i64 extremes (e.g. x = −1, s = 63).
    let r = x.wrapping_sub(q.wrapping_shl(s as u32)) as u64;
    let half = 1u64 << (s - 1);
    if r > half {
        q + 1 // cannot overflow: |q| ≤ 2^62 when s ≥ 1 (SPEC §5.1)
    } else if r == half {
        q + (q & 1) // half rounds to even
    } else {
        q
    }
}

/// Truncating division toward zero, divisor > 0 (SPEC §5.1). i128 makes
/// i64::MIN / 1 exact and matches the spec's u64-magnitude formulation.
pub fn trunc_div(a: i64, dv: i64) -> i64 {
    debug_assert!(dv > 0, "caller must trap T5 first");
    ((a as i128) / (dv as i128)) as i64
}

// ---------------------------------------------------------------------------
// Machine
// ---------------------------------------------------------------------------

/// Outcome of one executed transition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepOutcome {
    Ran,
    Halted,
    Trapped,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepError {
    /// Halted/trapped states have no successor (SPEC §4.4 terminality).
    AlreadyTerminal,
}

#[derive(Clone)]
pub struct Machine {
    pub regs: Registers,
    pub mem: CommittedMemory,
    pub program: Vec<Instr>,
    /// Program tree depth (SPEC §3.5): fetch space is 2^p slots; slots past
    /// `program.len()` are zero instructions (padding ⇒ trap T2).
    pub p: u8,
}

impl Machine {
    pub fn new(d: u8, p: u8, program: Vec<Instr>) -> Self {
        assert!(p <= crate::MAX_PROG_DEPTH, "program depth {p} out of range");
        assert!(
            (program.len() as u64) <= (1u64 << p),
            "program does not fit in 2^{p} slots"
        );
        Self {
            regs: Registers::default(), // genesis registers: all zero (SPEC §7.2)
            mem: CommittedMemory::new_zero(d),
            program,
            p,
        }
    }

    /// Build a machine with genesis memory installed from a full image
    /// (zero pages skipped — they're already the tree default).
    pub fn with_image(d: u8, p: u8, program: Vec<Instr>, image: &[u8]) -> Self {
        let mut m = Self::new(d, p, program);
        assert_eq!(image.len() as u64, m.mem.mem_bytes(), "image size mismatch");
        for (i, page) in image.chunks_exact(crate::PAGE_SIZE).enumerate() {
            if page.iter().any(|&b| b != 0) {
                m.mem.set_page(i as u64, page.try_into().unwrap());
            }
        }
        m
    }

    /// state_root = H(0x02 ‖ mem_root ‖ regs) (SPEC §3.3).
    pub fn state_root(&self) -> Hash {
        crate::hash::state_root(&self.mem.root(), &self.regs.encode())
    }

    /// Effective address (SPEC §4.2): base +w Σ idx[j] ·w stride[j], mod 2^64.
    fn ea(&self, o: &Operand) -> u64 {
        let mut a = o.base;
        for j in 0..4 {
            a = a.wrapping_add((self.regs.idx[j] as u64).wrapping_mul(o.stride[j] as u64));
        }
        a
    }

    /// T3: in-bounds and naturally aligned (size ∈ {1, 2, 4}).
    // `%` kept over `is_multiple_of`: this line must read exactly like
    // SPEC §4.4 T3 for audit, and like its Move twin in Phase 2.
    #[allow(clippy::manual_is_multiple_of)]
    fn ok_access(&self, ea: u64, size: u64) -> bool {
        ea % size == 0 && ea <= self.mem.mem_bytes() - size
    }

    /// T7 address part: 64-aligned full line in bounds (SPEC §5.2 DOT ops).
    #[allow(clippy::manual_is_multiple_of)]
    fn ok_line(&self, ea: u64) -> bool {
        ea % DOT_LINE as u64 == 0 && ea <= self.mem.mem_bytes() - DOT_LINE as u64
    }

    /// Trap transition (SPEC §4.4): halted ← 2, step ← step+1, everything
    /// else (pc, acc, aux, idx, memory) frozen.
    fn trap(&mut self) -> StepOutcome {
        self.regs.halted = TRAPPED;
        self.regs.step += 1;
        StepOutcome::Trapped
    }

    /// Execute exactly one micro-op (SPEC §5.2 semantics, §8.4 check order).
    pub fn step(&mut self) -> Result<StepOutcome, StepError> {
        if self.regs.halted != RUNNING {
            return Err(StepError::AlreadyTerminal);
        }

        // T1: pc outside the program tree — no leaf exists to fetch.
        if (self.regs.pc as u64) >= (1u64 << self.p) {
            return Ok(self.trap());
        }
        // Fetch; slots past program.len() are zero-padding (opcode 0x00).
        let instr = self
            .program
            .get(self.regs.pc as usize)
            .copied()
            .unwrap_or_else(Instr::zero);
        // T2: unknown opcode (includes padding).
        let op = match Opcode::from_u8(instr.opcode) {
            Some(o) => o,
            None => return Ok(self.trap()),
        };

        // pc ← pc+1 unless an arm overrides; wraps mod 2^32 (only reachable
        // at p = 32 — kept total for adversarial completeness).
        let mut next_pc = self.regs.pc.wrapping_add(1);

        match op {
            Opcode::Mac8 => {
                let (ea_a, ea_b) = (self.ea(&instr.a), self.ea(&instr.b));
                if !self.ok_access(ea_a, 1) || !self.ok_access(ea_b, 1) {
                    return Ok(self.trap());
                }
                let prod = sext8(self.mem.read_u8(ea_a)).wrapping_mul(sext8(self.mem.read_u8(ea_b)));
                self.regs.acc = self.regs.acc.wrapping_add(prod);
            }
            Opcode::Mac16 => {
                let (ea_a, ea_b) = (self.ea(&instr.a), self.ea(&instr.b));
                if !self.ok_access(ea_a, 2) || !self.ok_access(ea_b, 2) {
                    return Ok(self.trap());
                }
                let prod =
                    sext16(self.mem.read_u16(ea_a)).wrapping_mul(sext16(self.mem.read_u16(ea_b)));
                self.regs.acc = self.regs.acc.wrapping_add(prod);
            }
            Opcode::Dot8 | Opcode::Dot16 => {
                // T7: lane count and full-line alignment/bounds — checked on
                // the whole 64-byte line even when imm < cap (SPEC §5.2).
                let cap = if op == Opcode::Dot8 { 64 } else { 32 };
                if instr.imm == 0 || instr.imm > cap {
                    return Ok(self.trap());
                }
                let (ea_a, ea_b) = (self.ea(&instr.a), self.ea(&instr.b));
                if !self.ok_line(ea_a) || !self.ok_line(ea_b) {
                    return Ok(self.trap());
                }
                let lanes = instr.imm as usize;
                let mut acc = self.regs.acc;
                if op == Opcode::Dot8 {
                    let a = self.mem.read(ea_a, lanes);
                    let b = self.mem.read(ea_b, lanes);
                    for j in 0..lanes {
                        acc = acc.wrapping_add(sext8(a[j]).wrapping_mul(sext8(b[j])));
                    }
                } else {
                    let a = self.mem.read(ea_a, 2 * lanes);
                    let b = self.mem.read(ea_b, 2 * lanes);
                    for j in 0..lanes {
                        let av = sext16(u16::from_le_bytes([a[2 * j], a[2 * j + 1]]));
                        let bv = sext16(u16::from_le_bytes([b[2 * j], b[2 * j + 1]]));
                        acc = acc.wrapping_add(av.wrapping_mul(bv));
                    }
                }
                self.regs.acc = acc;
            }
            Opcode::Ld8 => {
                let ea_a = self.ea(&instr.a);
                if !self.ok_access(ea_a, 1) {
                    return Ok(self.trap());
                }
                self.regs.acc = sext8(self.mem.read_u8(ea_a));
            }
            Opcode::Ld32 => {
                let ea_a = self.ea(&instr.a);
                if !self.ok_access(ea_a, 4) {
                    return Ok(self.trap());
                }
                self.regs.acc = sext32(self.mem.read_u32(ea_a));
            }
            Opcode::Ldc => {
                self.regs.acc = sext32(instr.imm);
            }
            Opcode::Add32 => {
                let ea_a = self.ea(&instr.a);
                if !self.ok_access(ea_a, 4) {
                    return Ok(self.trap());
                }
                self.regs.acc = self.regs.acc.wrapping_add(sext32(self.mem.read_u32(ea_a)));
            }
            Opcode::Mul32 => {
                let ea_a = self.ea(&instr.a);
                if !self.ok_access(ea_a, 4) {
                    return Ok(self.trap());
                }
                self.regs.acc = self.regs.acc.wrapping_mul(sext32(self.mem.read_u32(ea_a)));
            }
            Opcode::Div32 => {
                let ea_a = self.ea(&instr.a);
                if !self.ok_access(ea_a, 4) {
                    return Ok(self.trap());
                }
                let dv = sext32(self.mem.read_u32(ea_a));
                // T5: value-dependent trap, after the read (SPEC §8.4 V6).
                if dv <= 0 {
                    return Ok(self.trap());
                }
                self.regs.acc = trunc_div(self.regs.acc, dv);
            }
            Opcode::ShiftRndn => {
                // T4
                if instr.s > 63 {
                    return Ok(self.trap());
                }
                self.regs.acc = rnd(self.regs.acc, instr.s);
            }
            Opcode::Clamp8 => {
                let ea_w = self.ea(&instr.w);
                if !self.ok_access(ea_w, 1) {
                    return Ok(self.trap());
                }
                let v = sat8(self.regs.acc);
                self.mem.write(ea_w, &v.to_le_bytes());
            }
            Opcode::Clamp16 => {
                let ea_w = self.ea(&instr.w);
                if !self.ok_access(ea_w, 2) {
                    return Ok(self.trap());
                }
                let v = sat16(self.regs.acc);
                self.mem.write(ea_w, &v.to_le_bytes());
            }
            Opcode::Lut16 => {
                // Index = sat16(acc) + 32768 ∈ [0, 65535]; tables are stored
                // from most-negative input. Strides of opA are ignored —
                // address is base + 2·index (SPEC §5.2).
                let index = (sat16(self.regs.acc) as i64 + 32768) as u64;
                let ea = instr.a.base.wrapping_add(2 * index);
                if !self.ok_access(ea, 2) {
                    return Ok(self.trap());
                }
                self.regs.acc = sext16(self.mem.read_u16(ea));
            }
            Opcode::St32 => {
                // Source selector in k (SPEC §5.2): 0 acc, 1 aux, 2..=5 idx.
                let src: u32 = match instr.k {
                    0 => self.regs.acc as u32, // low32 truncation
                    1 => self.regs.aux as u32,
                    2..=5 => self.regs.idx[(instr.k - 2) as usize],
                    _ => return Ok(self.trap()), // T6
                };
                let ea_w = self.ea(&instr.w);
                if !self.ok_access(ea_w, 4) {
                    return Ok(self.trap());
                }
                self.mem.write(ea_w, &src.to_le_bytes());
            }
            Opcode::Ldidx => {
                if instr.k > 3 {
                    return Ok(self.trap()); // T6
                }
                let ea_a = self.ea(&instr.a);
                if !self.ok_access(ea_a, 4) {
                    return Ok(self.trap());
                }
                self.regs.idx[instr.k as usize] = self.mem.read_u32(ea_a);
            }
            Opcode::ArgmaxStep | Opcode::ArgmaxOff => {
                if instr.k > 3 {
                    return Ok(self.trap()); // T6
                }
                let ea_a = self.ea(&instr.a);
                if !self.ok_access(ea_a, 4) {
                    return Ok(self.trap());
                }
                let v = sext32(self.mem.read_u32(ea_a));
                // Strictly greater ⇒ first maximum wins ⇒ ties break to the
                // lowest index under an ascending scan (SPEC §5.2).
                if v > self.regs.acc {
                    self.regs.acc = v;
                    // ARGMAX_OFF: aux gets imm + idx[k] — a chunk-local scan
                    // records a global row index (SPEC §5.2, streaming head).
                    let base = if op == Opcode::ArgmaxOff { instr.imm as u64 } else { 0 };
                    self.regs.aux =
                        base.wrapping_add(self.regs.idx[instr.k as usize] as u64) as i64;
                }
            }
            Opcode::Jmp => {
                next_pc = instr.target;
            }
            Opcode::Jeq => {
                let ea_a = self.ea(&instr.a);
                if !self.ok_access(ea_a, 4) {
                    return Ok(self.trap());
                }
                if self.mem.read_u32(ea_a) == instr.imm {
                    next_pc = instr.target;
                }
            }
            Opcode::Loop => {
                if instr.k > 3 {
                    return Ok(self.trap()); // T6
                }
                let k = instr.k as usize;
                let nxt = self.regs.idx[k].wrapping_add(1);
                if nxt < instr.imm {
                    self.regs.idx[k] = nxt;
                    next_pc = instr.target;
                } else {
                    self.regs.idx[k] = 0; // auto-reset for clean nesting
                }
            }
            Opcode::Ld16 => {
                let ea_a = self.ea(&instr.a);
                if !self.ok_access(ea_a, 2) {
                    return Ok(self.trap());
                }
                self.regs.acc = sext16(self.mem.read_u16(ea_a));
            }
            Opcode::Dot8x16 | Opcode::Dotbm => {
                // T7 variant: A = 64-B i8 line, B = 128-B i16 line.
                if instr.imm == 0 || instr.imm > 64 {
                    return Ok(self.trap());
                }
                let (ea_a, ea_b) = (self.ea(&instr.a), self.ea(&instr.b));
                let mem_bytes = self.mem.mem_bytes();
                if ea_a % 64 != 0
                    || ea_a > mem_bytes - 64
                    || ea_b % 128 != 0
                    || ea_b > mem_bytes - 128
                {
                    return Ok(self.trap());
                }
                let lanes = instr.imm as usize;
                let a = self.mem.read(ea_a, lanes);
                let b = self.mem.read(ea_b, 2 * lanes);
                let mut p = 0i64;
                for j in 0..lanes {
                    let av = sext8(a[j]);
                    let bv = sext16(u16::from_le_bytes([b[2 * j], b[2 * j + 1]]));
                    p = p.wrapping_add(av.wrapping_mul(bv));
                }
                if op == Opcode::Dotbm {
                    // W slot is a READ: the per-block multiplier cell.
                    let ea_w = self.ea(&instr.w);
                    if !self.ok_access(ea_w, 4) {
                        return Ok(self.trap());
                    }
                    let m = sext32(self.mem.read_u32(ea_w));
                    self.regs.acc = self.regs.acc.wrapping_add(p.wrapping_mul(m));
                } else {
                    self.regs.acc = self.regs.acc.wrapping_add(p);
                }
            }
            Opcode::Halt => {
                // halted ← 1, pc unchanged, step counts (SPEC §5.2).
                self.regs.halted = HALTED;
                self.regs.step += 1;
                return Ok(StepOutcome::Halted);
            }
        }

        self.regs.pc = next_pc;
        self.regs.step += 1;
        Ok(StepOutcome::Ran)
    }
}
