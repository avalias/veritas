/// Little-endian byte readers (SPEC §2.1: LE everywhere).
module dispute::bytes_le;

public fun u8_at(v: &vector<u8>, off: u64): u64 {
    (v[off] as u64)
}

public fun u16_at(v: &vector<u8>, off: u64): u64 {
    (v[off] as u64) | ((v[off + 1] as u64) << 8)
}

public fun u32_at(v: &vector<u8>, off: u64): u64 {
    (v[off] as u64)
        | ((v[off + 1] as u64) << 8)
        | ((v[off + 2] as u64) << 16)
        | ((v[off + 3] as u64) << 24)
}

public fun u64_at(v: &vector<u8>, off: u64): u64 {
    u32_at(v, off) | (u32_at(v, off + 4) << 32)
}

public fun push_u32(v: &mut vector<u8>, x: u64) {
    v.push_back(((x & 0xFF) as u8));
    v.push_back((((x >> 8) & 0xFF) as u8));
    v.push_back((((x >> 16) & 0xFF) as u8));
    v.push_back((((x >> 24) & 0xFF) as u8));
}

public fun push_u64(v: &mut vector<u8>, x: u64) {
    push_u32(v, x & 0xFFFF_FFFF);
    push_u32(v, x >> 32);
}
