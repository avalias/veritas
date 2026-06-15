//! Prints the golden constants pinned in vm/tests/conformance.rs.
//!
//! Regenerate ONLY when the spec version changes deliberately — a moved
//! golden is a consensus break, not a refactor.
//!
//! Usage: cargo run -p vm --bin gen_goldens

use vm::exec::Machine;
use vm::fixtures::golden_machine;
use vm::merkle::zero_page_subtrees;
use vm::trace::trace_digest;

fn hex(h: &[u8; 32]) -> String {
    h.iter().map(|b| format!("{b:02x}")).collect()
}

fn main() {
    println!("== zero-page subtree hashes Z_0..Z_4 (SPEC §3.4) ==");
    for (l, z) in zero_page_subtrees(4).iter().enumerate() {
        println!("Z_{l} = {}", hex(z));
    }

    let zero = Machine::new(2, 2, vec![]);
    println!("\n== zero machine (d=2, p=2, empty program) ==");
    println!("state_root = {}", hex(&zero.state_root()));

    let mut m = golden_machine();
    println!("\n== golden run (fixtures::golden_machine) ==");
    println!("genesis_root = {}", hex(&m.state_root()));
    let (digest, result) = trace_digest(&mut m, 10_000).expect("golden run must terminate");
    println!("trace_digest = {}", hex(&digest));
    println!("steps        = {}", result.steps);
    println!("outcome      = {:?}", result.outcome);
    println!("final_root   = {}", hex(&result.final_root));
}
