//! Deterministic tensor VM — reference implementation of SPEC.md.
//!
//! This crate is consensus-critical: every byte it hashes and every integer
//! it produces is normative (SPEC.md is the source of truth; this code is an
//! implementation of it). Rules enforced here and by workspace lints:
//!
//! - No floats anywhere (Invariant 1; `clippy::float_arithmetic` is denied).
//! - No `HashMap`/`HashSet` iteration in execution paths (Invariant 2).
//! - The per-step interpreter is single-threaded and allocation-stable.
//! - All wrapping arithmetic is *explicit* (`wrapping_*`); tests build with
//!   `overflow-checks = true` so unspecified overflow panics loudly.

pub mod exec;
pub mod fixtures;
pub mod hash;
pub mod isa;
pub mod merkle;
pub mod onestep;
pub mod softfloat;
pub mod state;
pub mod trace;

/// Memory page size in bytes (SPEC §3.1).
/// 1024 balances opening calldata (page + siblings ≈ 1.7 KiB at depth 20)
/// against tree depth, and 1024 % 8 == 0 means a naturally aligned access
/// never straddles a page.
pub const PAGE_SIZE: usize = 1024;

/// DOT operand line size in bytes (SPEC §5.2): one cache line. 64 divides
/// every common LLM dimension and divides PAGE_SIZE, so a 64-aligned line
/// never straddles a page.
pub const DOT_LINE: usize = 64;

/// Maximum memory tree depth (SPEC §3.1): 2^24 pages = 16 GiB, capping
/// sibling paths at 24 × 32 B = 768 B of calldata per opening.
pub const MAX_MEM_DEPTH: u8 = 24;

/// Maximum program tree depth (SPEC §3.5): pc is u32.
pub const MAX_PROG_DEPTH: u8 = 32;
