/// The one-step verifier (SPEC §8.4 V1–V9) — line-by-line port of
/// vm/src/onestep.rs::verify_step. Pure integer arithmetic; signed ops via
/// dispute::signed (u64 two's-complement). Returns 0 = resolver wins,
/// 1 = challenger wins; ABORTS on malformed proofs (no decision, no clock
/// reset — SPEC §8.2).
///
/// Check ORDER is normative and must match the Rust twin exactly; the
/// generated vector suite holds the two implementations together.
module dispute::interp;

use dispute::bytes_le as ble;
use dispute::merkle;
use dispute::signed as sg;

const E_PRE_ROOT: u64 = 1;
const E_BAD_INSTR_PROOF: u64 = 2;
const E_MISSING_OPENING: u64 = 3;
const E_BAD_OPENING: u64 = 4;

const PAGE: u64 = 1024;
const U32M: u64 = 0xFFFF_FFFF;
const RESOLVER_WINS: u8 = 0;
const CHALLENGER_WINS: u8 = 1;

// Opcodes (SPEC §5.2, append-only).
const OP_MAC8: u8 = 0x01;
const OP_MAC16: u8 = 0x02;
const OP_LD8: u8 = 0x03;
const OP_LD32: u8 = 0x04;
const OP_LDC: u8 = 0x05;
const OP_ADD32: u8 = 0x06;
const OP_MUL32: u8 = 0x07;
const OP_DIV32: u8 = 0x08;
const OP_SHIFT_RNDN: u8 = 0x09;
const OP_CLAMP8: u8 = 0x0A;
const OP_CLAMP16: u8 = 0x0B;
const OP_LUT16: u8 = 0x0C;
const OP_ST32: u8 = 0x0D;
const OP_LDIDX: u8 = 0x0E;
const OP_ARGMAX: u8 = 0x0F;
const OP_JMP: u8 = 0x10;
const OP_JEQ: u8 = 0x11;
const OP_LOOP: u8 = 0x12;
const OP_HALT: u8 = 0x13;
const OP_DOT8: u8 = 0x14;
const OP_DOT16: u8 = 0x15;
const OP_ARGMAX_OFF: u8 = 0x16;
const OP_LD16: u8 = 0x17;
const OP_DOT8X16: u8 = 0x18;
const OP_DOTBM: u8 = 0x19;

// Instruction field offsets (SPEC §4.1).
const F_IMM: u64 = 4;
const F_TARGET: u64 = 8;
const F_OPA: u64 = 16;
const F_OPB: u64 = 40;
const F_OPW: u64 = 64;

public struct Regs has copy, drop {
    pc: u64, // u32 value
    halted: u8,
    step: u64,
    acc: u64, // i64 bit pattern
    aux: u64, // i64 bit pattern
    i0: u64, // u32 values
    i1: u64,
    i2: u64,
    i3: u64,
}

fun decode_regs(b: &vector<u8>): Regs {
    Regs {
        pc: ble::u32_at(b, 0),
        halted: b[4],
        step: ble::u64_at(b, 5),
        acc: ble::u64_at(b, 13),
        aux: ble::u64_at(b, 21),
        i0: ble::u32_at(b, 29),
        i1: ble::u32_at(b, 33),
        i2: ble::u32_at(b, 37),
        i3: ble::u32_at(b, 41),
    }
}

fun encode_regs(r: &Regs): vector<u8> {
    let mut b = vector::empty<u8>();
    ble::push_u32(&mut b, r.pc);
    b.push_back(r.halted);
    ble::push_u64(&mut b, r.step);
    ble::push_u64(&mut b, r.acc);
    ble::push_u64(&mut b, r.aux);
    ble::push_u32(&mut b, r.i0);
    ble::push_u32(&mut b, r.i1);
    ble::push_u32(&mut b, r.i2);
    ble::push_u32(&mut b, r.i3);
    b
}

fun get_idx(r: &Regs, k: u8): u64 {
    if (k == 0) { r.i0 } else if (k == 1) { r.i1 } else if (k == 2) { r.i2 } else { r.i3 }
}

fun set_idx(r: &mut Regs, k: u8, v: u64) {
    if (k == 0) { r.i0 = v } else if (k == 1) { r.i1 = v } else if (k == 2) { r.i2 = v }
    else { r.i3 = v }
}

/// Effective address (SPEC §4.2): base +w Σ idx[j]·w stride[j].
fun ea(r: &Regs, instr: &vector<u8>, off: u64): u64 {
    let mut a = ble::u64_at(instr, off);
    a = sg::wadd(a, sg::wmul(r.i0, ble::u32_at(instr, off + 8)));
    a = sg::wadd(a, sg::wmul(r.i1, ble::u32_at(instr, off + 12)));
    a = sg::wadd(a, sg::wmul(r.i2, ble::u32_at(instr, off + 16)));
    a = sg::wadd(a, sg::wmul(r.i3, ble::u32_at(instr, off + 20)));
    a
}

fun ok_access(a: u64, size: u64, mem_bytes: u64): bool {
    a % size == 0 && a <= mem_bytes - size
}

fun ok_line(a: u64, mem_bytes: u64): bool {
    a % 64 == 0 && a <= mem_bytes - 64
}

/// T7 wide variant (SPEC §5.2 DOT8X16/DOTBM): 128-byte i16 line. 128 | 1024
/// so an aligned wide line never straddles a page.
fun ok_line128(a: u64, mem_bytes: u64): bool {
    a % 128 == 0 && a <= mem_bytes - 128
}

/// Verify a page opening against mem_root (SPEC §8.4 V6); abort if bad.
fun check_open(
    page: &vector<u8>,
    sibs: &vector<vector<u8>>,
    page_index: u64,
    d: u8,
    mem_root: &vector<u8>,
) {
    assert!(page.length() != 0, E_MISSING_OPENING);
    assert!(page.length() == PAGE && sibs.length() == (d as u64), E_BAD_OPENING);
    assert!(merkle::fold(merkle::page_leaf(page), page_index, sibs) == *mem_root, E_BAD_OPENING);
}

fun trap_regs(pre: &Regs): Regs {
    let mut t = *pre; // pc, acc, aux, idx frozen (SPEC §4.4)
    t.halted = 2;
    t.step = sg::wadd(pre.step, 1);
    t
}

fun verdict(mem_root_post: &vector<u8>, post: &Regs, claimed: &vector<u8>): u8 {
    if (merkle::state_root(mem_root_post, &encode_regs(post)) == *claimed) {
        RESOLVER_WINS
    } else {
        CHALLENGER_WINS
    }
}

public fun verify_step(
    pre_root: &vector<u8>,
    claimed_post: &vector<u8>,
    d: u8,
    p: u8,
    program_root: &vector<u8>,
    regs_bytes: vector<u8>,
    mem_root: vector<u8>,
    instr: vector<u8>,
    instr_sibs: vector<vector<u8>>,
    page_a: vector<u8>,
    sibs_a: vector<vector<u8>>,
    page_b: vector<u8>,
    sibs_b: vector<vector<u8>>,
    page_w: vector<u8>,
    sibs_w: vector<vector<u8>>,
): u8 {
    let mem_bytes = (1u64 << d) * PAGE;

    // V1: revealed (mem_root, regs) must BE the agreed pre-state.
    assert!(regs_bytes.length() == 45, E_PRE_ROOT);
    assert!(merkle::state_root(&mem_root, &regs_bytes) == *pre_root, E_PRE_ROOT);
    let pre = decode_regs(&regs_bytes);

    // V2: terminality — halted/trapped states have no successor.
    if (pre.halted != 0) { return CHALLENGER_WINS };

    // V3: T1 — pc outside the program tree; trap without any opening.
    if (pre.pc >= (1u64 << p)) {
        return verdict(&mem_root, &trap_regs(&pre), claimed_post)
    };

    // V4: instruction inclusion at index pc.
    assert!(instr.length() == 96 && instr_sibs.length() == (p as u64), E_BAD_INSTR_PROOF);
    assert!(
        merkle::fold(merkle::prog_leaf(&instr), pre.pc, &instr_sibs) == *program_root,
        E_BAD_INSTR_PROOF,
    );
    let op = instr[0];
    if (op == 0 || op > OP_DOTBM) {
        // T2: unknown opcode (includes zero padding).
        return verdict(&mem_root, &trap_regs(&pre), claimed_post)
    };

    // V5–V7: execute. post starts at pc+1/step+1; arms adjust.
    let mut post = pre;
    post.pc = (pre.pc + 1) & U32M;
    post.step = sg::wadd(pre.step, 1);
    let mut has_write = false;
    let mut write_ea = 0u64;
    let mut write_bytes = vector::empty<u8>();
    let k = instr[1];
    let s = instr[2];
    let imm = ble::u32_at(&instr, F_IMM);
    let target = ble::u32_at(&instr, F_TARGET);

    if (op == OP_MAC8 || op == OP_MAC16) {
        let size = if (op == OP_MAC8) { 1 } else { 2 };
        let ea_a = ea(&pre, &instr, F_OPA);
        let ea_b = ea(&pre, &instr, F_OPB);
        if (!ok_access(ea_a, size, mem_bytes) || !ok_access(ea_b, size, mem_bytes)) {
            return verdict(&mem_root, &trap_regs(&pre), claimed_post)
        };
        check_open(&page_a, &sibs_a, ea_a / PAGE, d, &mem_root);
        check_open(&page_b, &sibs_b, ea_b / PAGE, d, &mem_root);
        let (av, bv) = if (op == OP_MAC8) {
            (sg::sext8(page_a[ea_a % PAGE]), sg::sext8(page_b[ea_b % PAGE]))
        } else {
            (sg::sext16(ble::u16_at(&page_a, ea_a % PAGE)),
             sg::sext16(ble::u16_at(&page_b, ea_b % PAGE)))
        };
        post.acc = sg::wadd(pre.acc, sg::wmul(av, bv));
    } else if (op == OP_DOT8 || op == OP_DOT16) {
        let cap = if (op == OP_DOT8) { 64 } else { 32 };
        if (imm == 0 || imm > cap) {
            return verdict(&mem_root, &trap_regs(&pre), claimed_post) // T7 lanes
        };
        let ea_a = ea(&pre, &instr, F_OPA);
        let ea_b = ea(&pre, &instr, F_OPB);
        if (!ok_line(ea_a, mem_bytes) || !ok_line(ea_b, mem_bytes)) {
            return verdict(&mem_root, &trap_regs(&pre), claimed_post) // T7 line
        };
        check_open(&page_a, &sibs_a, ea_a / PAGE, d, &mem_root);
        check_open(&page_b, &sibs_b, ea_b / PAGE, d, &mem_root);
        let (oa, ob) = (ea_a % PAGE, ea_b % PAGE);
        let mut acc = pre.acc;
        let mut j = 0u64;
        while (j < imm) {
            let (av, bv) = if (op == OP_DOT8) {
                (sg::sext8(page_a[oa + j]), sg::sext8(page_b[ob + j]))
            } else {
                (sg::sext16(ble::u16_at(&page_a, oa + 2 * j)),
                 sg::sext16(ble::u16_at(&page_b, ob + 2 * j)))
            };
            acc = sg::wadd(acc, sg::wmul(av, bv));
            j = j + 1;
        };
        post.acc = acc;
    } else if (op == OP_LD8 || op == OP_LD16) {
        let size = if (op == OP_LD8) { 1 } else { 2 };
        let ea_a = ea(&pre, &instr, F_OPA);
        if (!ok_access(ea_a, size, mem_bytes)) {
            return verdict(&mem_root, &trap_regs(&pre), claimed_post)
        };
        check_open(&page_a, &sibs_a, ea_a / PAGE, d, &mem_root);
        post.acc = if (op == OP_LD8) {
            sg::sext8(page_a[ea_a % PAGE])
        } else {
            sg::sext16(ble::u16_at(&page_a, ea_a % PAGE))
        };
    } else if (op == OP_DOT8X16 || op == OP_DOTBM) {
        if (imm == 0 || imm > 64) {
            return verdict(&mem_root, &trap_regs(&pre), claimed_post) // T7 lanes
        };
        let ea_a = ea(&pre, &instr, F_OPA);
        let ea_b = ea(&pre, &instr, F_OPB);
        if (!ok_line(ea_a, mem_bytes) || !ok_line128(ea_b, mem_bytes)) {
            return verdict(&mem_root, &trap_regs(&pre), claimed_post) // T7 line
        };
        // DOTBM: the W slot is a READ — the per-block multiplier cell
        // (the one ISA asymmetry, SPEC §5.2).
        let ea_w = ea(&pre, &instr, F_OPW);
        if (op == OP_DOTBM && !ok_access(ea_w, 4, mem_bytes)) {
            return verdict(&mem_root, &trap_regs(&pre), claimed_post)
        };
        check_open(&page_a, &sibs_a, ea_a / PAGE, d, &mem_root);
        check_open(&page_b, &sibs_b, ea_b / PAGE, d, &mem_root);
        let (oa, ob) = (ea_a % PAGE, ea_b % PAGE);
        let mut p = 0u64; // i64 two's-complement carrier: fresh partial
        let mut j = 0u64;
        while (j < imm) {
            let av = sg::sext8(page_a[oa + j]);
            let bv = sg::sext16(ble::u16_at(&page_b, ob + 2 * j));
            p = sg::wadd(p, sg::wmul(av, bv));
            j = j + 1;
        };
        if (op == OP_DOTBM) {
            check_open(&page_w, &sibs_w, ea_w / PAGE, d, &mem_root);
            let m = sg::sext32(ble::u32_at(&page_w, ea_w % PAGE));
            post.acc = sg::wadd(pre.acc, sg::wmul(p, m));
        } else {
            post.acc = sg::wadd(pre.acc, p);
        };
    } else if (op == OP_LD32 || op == OP_ADD32 || op == OP_MUL32 || op == OP_DIV32) {
        let ea_a = ea(&pre, &instr, F_OPA);
        if (!ok_access(ea_a, 4, mem_bytes)) {
            return verdict(&mem_root, &trap_regs(&pre), claimed_post)
        };
        check_open(&page_a, &sibs_a, ea_a / PAGE, d, &mem_root);
        let v = sg::sext32(ble::u32_at(&page_a, ea_a % PAGE));
        if (op == OP_LD32) {
            post.acc = v;
        } else if (op == OP_ADD32) {
            post.acc = sg::wadd(pre.acc, v);
        } else if (op == OP_MUL32) {
            post.acc = sg::wmul(pre.acc, v);
        } else {
            // T5: divisor ≤ 0 — value-dependent trap after the read.
            if (!sg::sgt(v, 0)) {
                return verdict(&mem_root, &trap_regs(&pre), claimed_post)
            };
            post.acc = sg::sdiv(pre.acc, v);
        };
    } else if (op == OP_LDC) {
        post.acc = sg::sext32(imm);
    } else if (op == OP_SHIFT_RNDN) {
        if (s > 63) {
            return verdict(&mem_root, &trap_regs(&pre), claimed_post) // T4
        };
        post.acc = sg::rnd(pre.acc, s);
    } else if (op == OP_CLAMP8 || op == OP_CLAMP16) {
        let size = if (op == OP_CLAMP8) { 1 } else { 2 };
        let ea_w = ea(&pre, &instr, F_OPW);
        if (!ok_access(ea_w, size, mem_bytes)) {
            return verdict(&mem_root, &trap_regs(&pre), claimed_post)
        };
        has_write = true;
        write_ea = ea_w;
        if (op == OP_CLAMP8) {
            write_bytes.push_back(sg::sat8(pre.acc));
        } else {
            let v = sg::sat16(pre.acc);
            write_bytes.push_back(((v & 0xFF) as u8));
            write_bytes.push_back((((v >> 8) & 0xFF) as u8));
        };
    } else if (op == OP_LUT16) {
        // index = sat16(acc) + 32768 == sat16(acc) XOR 0x8000 on 16 bits.
        let index = sg::sat16(pre.acc) ^ 0x8000;
        let ea_t = sg::wadd(ble::u64_at(&instr, F_OPA), 2 * index); // base only
        if (!ok_access(ea_t, 2, mem_bytes)) {
            return verdict(&mem_root, &trap_regs(&pre), claimed_post)
        };
        check_open(&page_a, &sibs_a, ea_t / PAGE, d, &mem_root);
        post.acc = sg::sext16(ble::u16_at(&page_a, ea_t % PAGE));
    } else if (op == OP_ST32) {
        let src = if (k == 0) {
            pre.acc & U32M
        } else if (k == 1) {
            pre.aux & U32M
        } else if (k <= 5) {
            get_idx(&pre, k - 2)
        } else {
            return verdict(&mem_root, &trap_regs(&pre), claimed_post) // T6
        };
        let ea_w = ea(&pre, &instr, F_OPW);
        if (!ok_access(ea_w, 4, mem_bytes)) {
            return verdict(&mem_root, &trap_regs(&pre), claimed_post)
        };
        has_write = true;
        write_ea = ea_w;
        ble::push_u32(&mut write_bytes, src);
    } else if (op == OP_LDIDX) {
        if (k > 3) {
            return verdict(&mem_root, &trap_regs(&pre), claimed_post) // T6
        };
        let ea_a = ea(&pre, &instr, F_OPA);
        if (!ok_access(ea_a, 4, mem_bytes)) {
            return verdict(&mem_root, &trap_regs(&pre), claimed_post)
        };
        check_open(&page_a, &sibs_a, ea_a / PAGE, d, &mem_root);
        set_idx(&mut post, k, ble::u32_at(&page_a, ea_a % PAGE));
    } else if (op == OP_ARGMAX || op == OP_ARGMAX_OFF) {
        if (k > 3) {
            return verdict(&mem_root, &trap_regs(&pre), claimed_post) // T6
        };
        let ea_a = ea(&pre, &instr, F_OPA);
        if (!ok_access(ea_a, 4, mem_bytes)) {
            return verdict(&mem_root, &trap_regs(&pre), claimed_post)
        };
        check_open(&page_a, &sibs_a, ea_a / PAGE, d, &mem_root);
        let v = sg::sext32(ble::u32_at(&page_a, ea_a % PAGE));
        if (sg::sgt(v, pre.acc)) {
            post.acc = v;
            // ARGMAX_OFF: aux ← imm +w idx[k] (global row index from a
            // chunk-local scan — SPEC §5.2 streaming head).
            let base = if (op == OP_ARGMAX_OFF) { imm } else { 0 };
            post.aux = sg::wadd(base, get_idx(&pre, k));
        };
    } else if (op == OP_JMP) {
        post.pc = target;
    } else if (op == OP_JEQ) {
        let ea_a = ea(&pre, &instr, F_OPA);
        if (!ok_access(ea_a, 4, mem_bytes)) {
            return verdict(&mem_root, &trap_regs(&pre), claimed_post)
        };
        check_open(&page_a, &sibs_a, ea_a / PAGE, d, &mem_root);
        if (ble::u32_at(&page_a, ea_a % PAGE) == imm) {
            post.pc = target;
        };
    } else if (op == OP_LOOP) {
        if (k > 3) {
            return verdict(&mem_root, &trap_regs(&pre), claimed_post) // T6
        };
        let nxt = (get_idx(&pre, k) + 1) & U32M;
        if (nxt < imm) {
            set_idx(&mut post, k, nxt);
            post.pc = target;
        } else {
            set_idx(&mut post, k, 0); // auto-reset for clean nesting
        };
    } else {
        // The only opcode left in the validated range [0x01, 0x15] is HALT
        // — pc unchanged, halted ← 1 (SPEC §5.2).
        assert!(op == OP_HALT, 0); // structurally unreachable
        post.pc = pre.pc;
        post.halted = 1;
    };

    // V7 write application + V8 post-root.
    let mem_root_post = if (has_write) {
        let widx = write_ea / PAGE;
        check_open(&page_w, &sibs_w, widx, d, &mem_root);
        let patched = patch(&page_w, write_ea % PAGE, &write_bytes);
        merkle::fold(merkle::page_leaf(&patched), widx, &sibs_w)
    } else {
        mem_root
    };

    // V9.
    verdict(&mem_root_post, &post, claimed_post)
}

fun patch(page: &vector<u8>, off: u64, bytes: &vector<u8>): vector<u8> {
    let mut out = vector::empty<u8>();
    let n = page.length();
    let m = bytes.length();
    let mut i = 0u64;
    while (i < n) {
        if (i >= off && i < off + m) {
            out.push_back(bytes[i - off]);
        } else {
            out.push_back(page[i]);
        };
        i = i + 1;
    };
    out
}
