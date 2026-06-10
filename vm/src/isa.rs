//! Instruction set: opcodes and the fixed 96-byte encoding (SPEC §4.1, §5.2).

/// Fixed instruction encoding length (SPEC §4.1).
pub const INSTR_ENC_LEN: usize = 96;

/// Opcode numbers are consensus constants and APPEND-ONLY across spec
/// versions (SPEC §5.2). 0x00 is reserved: program-tree padding decodes to
/// it and traps (T2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Opcode {
    Mac8 = 0x01,
    Mac16 = 0x02,
    Ld8 = 0x03,
    Ld32 = 0x04,
    Ldc = 0x05,
    Add32 = 0x06,
    Mul32 = 0x07,
    Div32 = 0x08,
    ShiftRndn = 0x09,
    Clamp8 = 0x0A,
    Clamp16 = 0x0B,
    Lut16 = 0x0C,
    St32 = 0x0D,
    Ldidx = 0x0E,
    ArgmaxStep = 0x0F,
    Jmp = 0x10,
    Jeq = 0x11,
    Loop = 0x12,
    Halt = 0x13,
    Dot8 = 0x14,
    Dot16 = 0x15,
    ArgmaxOff = 0x16,
}

impl Opcode {
    /// Exhaustive decode; `None` ⇒ trap T2 at execution.
    pub fn from_u8(b: u8) -> Option<Opcode> {
        use Opcode::*;
        Some(match b {
            0x01 => Mac8,
            0x02 => Mac16,
            0x03 => Ld8,
            0x04 => Ld32,
            0x05 => Ldc,
            0x06 => Add32,
            0x07 => Mul32,
            0x08 => Div32,
            0x09 => ShiftRndn,
            0x0A => Clamp8,
            0x0B => Clamp16,
            0x0C => Lut16,
            0x0D => St32,
            0x0E => Ldidx,
            0x0F => ArgmaxStep,
            0x10 => Jmp,
            0x11 => Jeq,
            0x12 => Loop,
            0x13 => Halt,
            0x14 => Dot8,
            0x15 => Dot16,
            0x16 => ArgmaxOff,
            _ => return None,
        })
    }
}

/// Operand descriptor (SPEC §4.1): 24 bytes — base u64 + 4 × stride u32.
/// Effective address: base +w Σ idx[j]·w stride[j] (mod 2^64, SPEC §4.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Operand {
    pub base: u64,
    pub stride: [u32; 4],
}

impl Operand {
    pub fn at(base: u64) -> Self {
        Self { base, stride: [0; 4] }
    }

    fn encode_into(&self, b: &mut [u8]) {
        b[0..8].copy_from_slice(&self.base.to_le_bytes());
        for (j, s) in self.stride.iter().enumerate() {
            b[8 + 4 * j..12 + 4 * j].copy_from_slice(&s.to_le_bytes());
        }
    }

    fn decode_from(b: &[u8]) -> Self {
        let mut stride = [0u32; 4];
        for (j, s) in stride.iter_mut().enumerate() {
            *s = u32::from_le_bytes(b[8 + 4 * j..12 + 4 * j].try_into().unwrap());
        }
        Self {
            base: u64::from_le_bytes(b[0..8].try_into().unwrap()),
            stride,
        }
    }
}

/// One instruction. `opcode` is kept raw (u8) so adversarial/padding values
/// are representable; decode happens at execution (SPEC §4.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Instr {
    pub opcode: u8,
    pub k: u8,
    pub s: u8,
    pub imm: u32,
    pub target: u32,
    pub a: Operand,
    pub b: Operand,
    pub w: Operand,
}

impl Instr {
    /// All-zero instruction == program-tree padding (opcode 0x00 ⇒ trap T2).
    pub fn zero() -> Self {
        Self::default()
    }

    pub fn op(opcode: Opcode) -> Self {
        Self {
            opcode: opcode as u8,
            ..Self::default()
        }
    }

    /// Canonical 96-byte encoding; reserved bytes ([3], [12..16], [88..96])
    /// are zero (SPEC §4.1 — compiler MUST zero, verifier ignores).
    pub fn encode(&self) -> [u8; INSTR_ENC_LEN] {
        let mut b = [0u8; INSTR_ENC_LEN];
        b[0] = self.opcode;
        b[1] = self.k;
        b[2] = self.s;
        b[4..8].copy_from_slice(&self.imm.to_le_bytes());
        b[8..12].copy_from_slice(&self.target.to_le_bytes());
        self.a.encode_into(&mut b[16..40]);
        self.b.encode_into(&mut b[40..64]);
        self.w.encode_into(&mut b[64..88]);
        b
    }

    /// Total decode (never fails); reserved bytes are ignored.
    pub fn decode(b: &[u8; INSTR_ENC_LEN]) -> Self {
        Self {
            opcode: b[0],
            k: b[1],
            s: b[2],
            imm: u32::from_le_bytes(b[4..8].try_into().unwrap()),
            target: u32::from_le_bytes(b[8..12].try_into().unwrap()),
            a: Operand::decode_from(&b[16..40]),
            b: Operand::decode_from(&b[40..64]),
            w: Operand::decode_from(&b[64..88]),
        }
    }
}
