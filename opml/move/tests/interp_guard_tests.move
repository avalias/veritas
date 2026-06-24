/// verify_step totality: a Fact's (d, p) are caller-supplied u8 (up to 255).
/// Oversized dimensions must abort cleanly with E_BAD_DEPTH BEFORE any `1<<d`/
/// `1<<p` shift or the mem_bytes multiply — never a raw arithmetic abort
/// (twin of the Rust ProofError::BadDepth guard; SPEC §8.2 totality).
#[test_only]
module opml::interp_guard_tests;

use opml::interp;

fun z(n: u64): vector<u8> {
    let mut v = vector<u8>[];
    let mut i = 0;
    while (i < n) { v.push_back(0u8); i = i + 1; };
    v
}

fun zz(): vector<vector<u8>> { vector[] }

#[test]
#[expected_failure(abort_code = 5, location = opml::interp)]
fun verify_step_rejects_oversized_depth() {
    // d = 64 > MAX_DEPTH(48) → E_BAD_DEPTH before any `1<<d` shift.
    let _ = interp::verify_step(
        &z(32), &z(32), 64, 8, &z(32),
        z(45), z(32), z(96), zz(),
        z(0), zz(), z(0), zz(), z(0), zz(),
    );
}

#[test]
#[expected_failure(abort_code = 5, location = opml::interp)]
fun verify_step_rejects_oversized_program() {
    // p = 200 > MAX_DEPTH → E_BAD_DEPTH before the `1<<p` check.
    let _ = interp::verify_step(
        &z(32), &z(32), 8, 200, &z(32),
        z(45), z(32), z(96), zz(),
        z(0), zz(), z(0), zz(), z(0), zz(),
    );
}
