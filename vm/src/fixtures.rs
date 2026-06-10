//! Deterministic fixtures shared by conformance tests and the golden
//! generator binary. Nothing here is consensus code, but the golden machine
//! it builds pins the cross-platform trace digest (conformance C-6), so its
//! construction must itself be deterministic — hence the fixed-seed
//! xorshift, never `rand`/`HashMap`.

use crate::exec::Machine;
use crate::isa::{Instr, Opcode, Operand};
use crate::PAGE_SIZE;

/// xorshift64 (Marsaglia) — tiny deterministic byte source for fixtures.
pub struct XorShift64(u64);

impl XorShift64 {
    pub fn new(seed: u64) -> Self {
        assert!(seed != 0, "xorshift64 state must be nonzero");
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    pub fn fill(&mut self, buf: &mut [u8]) {
        for chunk in buf.chunks_mut(8) {
            let b = self.next_u64().to_le_bytes();
            chunk.copy_from_slice(&b[..chunk.len()]);
        }
    }
}

/// Fixed seed for the golden machine (arbitrary odd constant; changing it
/// invalidates the pinned C-6 digest by design).
pub const GOLDEN_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

// Golden-machine memory layout (d = 8 ⇒ 256 KiB):
//   0..192    three 64-byte i8 lines (DOT8 A operand, strided by idx0)
//   192..256  one 64-byte i8 line (DOT8 B operand)
//   256       i32 const 3   (MUL32 multiplier)
//   260       i32 const 7   (DIV32 divisor)
//   384..396  three i32 result cells ("logits", ST32 target)
//   448..460  three interleaved i16 pairs (MAC16 operands)
//   512       u32 token cell (argmax result, JEQ subject)
//   516, 520  LUT16 / CLAMP8 scratch outputs
//   65536..196608  128 KiB pseudo-LUT16 table (seeded bytes)
const GOLDEN_D: u8 = 8;
const GOLDEN_P: u8 = 5; // 22 instructions fit in 2^5 slots

pub fn golden_program() -> Vec<Instr> {
    let op = |o: Opcode| Instr::op(o);
    let strided = |base: u64, j: usize, stride: u32| {
        let mut o = Operand::at(base);
        o.stride[j] = stride;
        o
    };
    vec![
        // -- body, looped 3× over idx0 ------------------------------------
        /* 0 */
        Instr { imm: (-7i32) as u32, ..op(Opcode::Ldc) },
        /* 1 */
        Instr { imm: 64, a: strided(0, 0, 64), b: Operand::at(192), ..op(Opcode::Dot8) },
        /* 2 */ Instr { a: Operand::at(256), ..op(Opcode::Mul32) },
        /* 3 */ Instr { s: 2, ..op(Opcode::ShiftRndn) },
        /* 4 */
        Instr { a: strided(448, 0, 4), b: strided(450, 0, 4), ..op(Opcode::Mac16) },
        /* 5 */ Instr { a: Operand::at(260), ..op(Opcode::Div32) },
        /* 6 */ Instr { k: 0, w: strided(384, 0, 4), ..op(Opcode::St32) },
        /* 7 */ Instr { k: 0, target: 0, imm: 3, ..op(Opcode::Loop) },
        // -- argmax over the three result cells via idx1 -------------------
        /* 8 */ Instr { imm: 0x8000_0000, ..op(Opcode::Ldc) }, // acc = i32::MIN
        /* 9 */ Instr { k: 1, a: strided(384, 1, 4), ..op(Opcode::ArgmaxStep) },
        /* 10 */ Instr { k: 1, target: 9, imm: 3, ..op(Opcode::Loop) },
        /* 11 */ Instr { k: 1, w: Operand::at(512), ..op(Opcode::St32) }, // token = aux
        /* 12 */ Instr { k: 2, a: Operand::at(512), ..op(Opcode::Ldidx) },
        /* 13 */ Instr { a: Operand::at(65536), ..op(Opcode::Lut16) },
        /* 14 */ Instr { w: Operand::at(516), ..op(Opcode::Clamp16) },
        // -- tail: LD32/ADD32/rounding/CLAMP8 + a data-dependent branch ----
        /* 15 */ Instr { a: Operand::at(384), ..op(Opcode::Ld32) },
        /* 16 */ Instr { a: Operand::at(388), ..op(Opcode::Add32) },
        /* 17 */ Instr { s: 1, ..op(Opcode::ShiftRndn) },
        /* 18 */ Instr { w: Operand::at(520), ..op(Opcode::Clamp8) },
        /* 19 */ Instr { a: Operand::at(512), imm: 1, target: 21, ..op(Opcode::Jeq) },
        /* 20 */ Instr { a: Operand::at(1), ..op(Opcode::Ld8) },
        /* 21 */ op(Opcode::Halt),
    ]
}

pub fn golden_machine() -> Machine {
    let mut m = Machine::new(GOLDEN_D, GOLDEN_P, golden_program());
    let mut rng = XorShift64::new(GOLDEN_SEED);

    // Compose the full initial image, then install whole pages — cheaper
    // than per-cell writes and identical in result.
    let mem_bytes = (1usize << GOLDEN_D) * PAGE_SIZE;
    let mut image = vec![0u8; mem_bytes];
    rng.fill(&mut image[0..256]); // DOT lines A0..A2 and B
    image[256..260].copy_from_slice(&3i32.to_le_bytes());
    image[260..264].copy_from_slice(&7i32.to_le_bytes());
    rng.fill(&mut image[448..460]); // MAC16 pairs
    rng.fill(&mut image[65536..65536 + 131072]); // pseudo-LUT table

    for (i, page) in image.chunks_exact(PAGE_SIZE).enumerate() {
        if page.iter().any(|&b| b != 0) {
            m.mem.set_page(i as u64, page.try_into().unwrap());
        }
    }
    m
}
