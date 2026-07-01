/// The dispute protocol objects and entry points (SPEC §8.1–§8.5) —
/// on-chain twin of game/src/chain.rs (MockChain), with sui::clock time
/// and real SUI bonds.
module opml::dispute;

use opml::interp;
use opml::merkle;
use sui::balance::Balance;
use sui::clock::Clock;
use sui::coin::{Self, Coin};
use sui::sui::SUI;

const STATUS_OPEN: u8 = 0;
const STATUS_CHALLENGED: u8 = 1;
const STATUS_OUTPUT_CHALLENGED: u8 = 2;
const STATUS_FINALIZED: u8 = 3;
const STATUS_REJECTED: u8 = 4;

const E_WRONG_STATUS: u64 = 10;
const E_WINDOW_STILL_OPEN: u64 = 11;
const E_WINDOW_CLOSED: u64 = 12;
const E_NOT_YOUR_TURN: u64 = 13;
const E_NO_PENDING_MID: u64 = 14;
const E_INTERVAL_NOT_ATOMIC: u64 = 15;
const E_INTERVAL_ATOMIC: u64 = 16;
const E_DEADLINE_NOT_PASSED: u64 = 17;
const E_BAD_BOND: u64 = 18;
const E_BAD_FINAL_STATE: u64 = 19;
const E_NOT_PARTY: u64 = 20;

const PAGE: u64 = 1024;

public struct Fact has key {
    id: UID,
    // -- judge identity (SPEC §8.1) --
    d: u8,
    p: u8,
    program_root: vector<u8>,
    genesis_root: vector<u8>,
    out_base: u64,
    // -- claim --
    n: u64,
    root_n: vector<u8>,
    output: vector<u8>,
    // -- economics --
    resolver: address,
    challenger: address,
    pot: Balance<SUI>,
    bond_value: u64,
    created_ms: u64,
    window_ms: u64,
    timeout_ms: u64,
    status: u8,
    // -- bisection (valid while status == CHALLENGED) --
    lo: u64,
    hi: u64,
    root_lo: vector<u8>,
    root_hi: vector<u8>,
    has_pending: bool,
    pending_mid: vector<u8>,
    challenger_turn: bool,
    deadline_ms: u64,
}

/// Assert a judgment optimistically; the bond is escrowed until the
/// challenge window closes or a dispute settles.
entry fun assert_fact(
    d: u8,
    p: u8,
    program_root: vector<u8>,
    genesis_root: vector<u8>,
    out_base: u64,
    n: u64,
    root_n: vector<u8>,
    output: vector<u8>,
    bond: Coin<SUI>,
    window_ms: u64,
    timeout_ms: u64,
    clock: &Clock,
    ctx: &mut TxContext,
) {
    let bond_value = bond.value();
    assert!(bond_value > 0 && n > 0, E_BAD_BOND);
    transfer::share_object(Fact {
        id: object::new(ctx),
        d,
        p,
        program_root,
        genesis_root,
        out_base,
        n,
        root_n,
        output,
        resolver: ctx.sender(),
        challenger: @0x0,
        pot: bond.into_balance(),
        bond_value,
        created_ms: clock.timestamp_ms(),
        window_ms,
        timeout_ms,
        status: STATUS_OPEN,
        lo: 0,
        hi: 0,
        root_lo: vector[],
        root_hi: vector[],
        has_pending: false,
        pending_mid: vector[],
        challenger_turn: false,
        deadline_ms: 0,
    });
}

/// Unchallenged after the window ⇒ the claim stands; bond returns.
entry fun finalize(f: &mut Fact, clock: &Clock, ctx: &mut TxContext) {
    assert!(f.status == STATUS_OPEN, E_WRONG_STATUS);
    assert!(clock.timestamp_ms() >= f.created_ms + f.window_ms, E_WINDOW_STILL_OPEN);
    settle(f, /*resolver_wins=*/ true, ctx);
}

/// Open the bisection game (SPEC §8.3): interval [0, N], genesis agreed by
/// construction, claim disputed. Challenger matches the bond.
entry fun challenge(f: &mut Fact, bond: Coin<SUI>, clock: &Clock, ctx: &TxContext) {
    assert!(f.status == STATUS_OPEN, E_WRONG_STATUS);
    let now = clock.timestamp_ms();
    assert!(now < f.created_ms + f.window_ms, E_WINDOW_CLOSED);
    assert!(bond.value() == f.bond_value, E_BAD_BOND);
    f.pot.join(bond.into_balance());
    f.challenger = ctx.sender();
    f.status = STATUS_CHALLENGED;
    f.lo = 0;
    f.hi = f.n;
    f.root_lo = f.genesis_root;
    f.root_hi = f.root_n;
    f.challenger_turn = false; // resolver posts the first midpoint
    f.deadline_ms = now + f.timeout_ms;
}

/// Resolver posts the midpoint root of the current interval.
entry fun post_mid(f: &mut Fact, root: vector<u8>, clock: &Clock, ctx: &TxContext) {
    assert!(f.status == STATUS_CHALLENGED, E_WRONG_STATUS);
    assert!(ctx.sender() == f.resolver, E_NOT_PARTY);
    assert!(!f.challenger_turn, E_NOT_YOUR_TURN);
    assert!(f.hi - f.lo > 1, E_INTERVAL_ATOMIC);
    f.pending_mid = root;
    f.has_pending = true;
    f.challenger_turn = true;
    f.deadline_ms = clock.timestamp_ms() + f.timeout_ms;
}

/// Challenger agrees (lo ← mid) or disagrees (hi ← mid). The invariant
/// "agreed at lo, disputed at hi" is preserved by construction.
entry fun respond(f: &mut Fact, agree: bool, clock: &Clock, ctx: &TxContext) {
    assert!(f.status == STATUS_CHALLENGED, E_WRONG_STATUS);
    assert!(ctx.sender() == f.challenger, E_NOT_PARTY);
    assert!(f.challenger_turn, E_NOT_YOUR_TURN);
    assert!(f.has_pending, E_NO_PENDING_MID);
    let mid = f.lo + (f.hi - f.lo) / 2;
    if (agree) {
        f.lo = mid;
        f.root_lo = f.pending_mid;
    } else {
        f.hi = mid;
        f.root_hi = f.pending_mid;
    };
    f.has_pending = false;
    f.challenger_turn = false;
    f.deadline_ms = clock.timestamp_ms() + f.timeout_ms;
}

/// Final one-step verification (SPEC §8.4). Either party may submit; the
/// comparison decides. Malformed proofs abort inside interp (no decision).
entry fun verify_step(
    f: &mut Fact,
    regs: vector<u8>,
    mem_root: vector<u8>,
    instr: vector<u8>,
    instr_sibs: vector<vector<u8>>,
    page_a: vector<u8>,
    sibs_a: vector<vector<u8>>,
    page_b: vector<u8>,
    sibs_b: vector<vector<u8>>,
    page_w: vector<u8>,
    sibs_w: vector<vector<u8>>,
    ctx: &mut TxContext,
) {
    assert!(f.status == STATUS_CHALLENGED, E_WRONG_STATUS);
    assert!(f.hi - f.lo == 1, E_INTERVAL_NOT_ATOMIC);
    let v = interp::verify_step(
        &f.root_lo,
        &f.root_hi,
        f.d,
        f.p,
        &f.program_root,
        regs,
        mem_root,
        instr,
        instr_sibs,
        page_a,
        sibs_a,
        page_b,
        sibs_b,
        page_w,
        sibs_w,
    );
    settle(f, v == 0, ctx);
}

/// Whoever owes the next move and missed the deadline loses (SPEC §8.2).
/// At the atomic interval the proof obligation is the resolver's; in the
/// output challenge the reveal obligation is the resolver's.
entry fun claim_timeout(f: &mut Fact, clock: &Clock, ctx: &mut TxContext) {
    let now = clock.timestamp_ms();
    if (f.status == STATUS_OUTPUT_CHALLENGED) {
        assert!(now > f.deadline_ms, E_DEADLINE_NOT_PASSED);
        settle(f, false, ctx);
        return
    };
    assert!(f.status == STATUS_CHALLENGED, E_WRONG_STATUS);
    assert!(now > f.deadline_ms, E_DEADLINE_NOT_PASSED);
    let resolver_stalled = (f.hi - f.lo == 1) || !f.challenger_turn;
    settle(f, !resolver_stalled, ctx);
}

/// Output-binding challenge (SPEC §8.5): dispute the claim's output field
/// without disputing execution.
entry fun challenge_output(f: &mut Fact, bond: Coin<SUI>, clock: &Clock, ctx: &TxContext) {
    assert!(f.status == STATUS_OPEN, E_WRONG_STATUS);
    let now = clock.timestamp_ms();
    assert!(now < f.created_ms + f.window_ms, E_WINDOW_CLOSED);
    assert!(bond.value() == f.bond_value, E_BAD_BOND);
    f.pot.join(bond.into_balance());
    f.challenger = ctx.sender();
    f.status = STATUS_OUTPUT_CHALLENGED;
    f.deadline_ms = now + f.timeout_ms;
}

/// Resolver reveals the final state: root_n preimage + output-page opening.
/// Checks (SPEC §8.5): halted == 1, step == N, output bytes match the
/// claim. A failed check slashes the resolver by their own revelation.
entry fun reveal_final_state(
    f: &mut Fact,
    regs: vector<u8>,
    mem_root: vector<u8>,
    out_page: vector<u8>,
    out_sibs: vector<vector<u8>>,
    ctx: &mut TxContext,
) {
    assert!(f.status == STATUS_OUTPUT_CHALLENGED, E_WRONG_STATUS);
    // The revelation must be the genuine preimage of the claimed root —
    // anything else is garbage, not a decision.
    assert!(regs.length() == 45, E_BAD_FINAL_STATE);
    assert!(merkle::state_root(&mem_root, &regs) == f.root_n, E_BAD_FINAL_STATE);
    let halted = regs[4];
    let step = opml::bytes_le::u64_at(&regs, 5);
    let ok = halted == 1 && step == f.n && output_matches(f, &mem_root, &out_page, &out_sibs);
    settle(f, ok, ctx);
}

fun output_matches(
    f: &Fact,
    mem_root: &vector<u8>,
    out_page: &vector<u8>,
    out_sibs: &vector<vector<u8>>,
): bool {
    let want = &f.output;
    let m = want.length();
    if (m < 4 || m > PAGE) { return false };
    if (out_page.length() != PAGE || out_sibs.length() != (f.d as u64)) { return false };
    let page_index = f.out_base / PAGE;
    if (merkle::fold(merkle::page_leaf(out_page), page_index, out_sibs) != *mem_root) {
        return false
    };
    let off = f.out_base % PAGE;
    // The output region must fit inside the single opened page — a region that
    // straddles a page boundary can't be proven by one opening. Return false
    // (totality: never an out-of-bounds abort on a malformed/non-aligned base).
    if (off + m > PAGE) { return false };
    let mut i = 0u64;
    while (i < m) {
        if (out_page[off + i] != want[i]) { return false };
        i = i + 1;
    };
    true
}

/// Pay the whole pot to the winner; the loser is thereby slashed.
fun settle(f: &mut Fact, resolver_wins: bool, ctx: &mut TxContext) {
    let winner = if (resolver_wins) { f.resolver } else { f.challenger };
    f.status = if (resolver_wins) { STATUS_FINALIZED } else { STATUS_REJECTED };
    let amount = f.pot.value();
    if (amount > 0) {
        transfer::public_transfer(coin::from_balance(f.pot.split(amount), ctx), winner);
    };
}

// -- read-only helpers for clients/tests + the market layer (§ market.move) --

public fun status(f: &Fact): u8 { f.status }

public fun interval(f: &Fact): (u64, u64) { (f.lo, f.hi) }

public fun pot_value(f: &Fact): u64 { f.pot.value() }

/// The verdict bytes the resolver optimistically committed. NOTE: the bisection
/// game protects `root_n`, NOT this field — `output` enters the game only via the
/// optional, mutually-exclusive challenge_output path (SPEC §8.5). A consumer
/// that reads this for value MUST first bind it to root_n with `output_is_bound`.
public fun output(f: &Fact): vector<u8> { f.output }

/// TRUE iff (regs, mem_root, out_page, out_sibs) opens this Fact's committed
/// final root_n and proves `output` really is the terminal output region:
/// state_root(mem_root, regs) == root_n, halted == 1, step == n, and the
/// `output` bytes fold into mem_root at out_base. These are exactly the
/// reveal_final_state checks (SPEC §8.5), exposed as a pure predicate so the
/// market can bind output to root_n AT THE POINT OF USE. This collapses an
/// output-lie into a root_n-lie the bisection game already adjudicates —
/// unifying output's protection with root_n's and retiring the separate,
/// bypassable challenge_output surface as a required trust root. (It does not
/// make root_n itself non-arbitrary on the unchallenged finalize path — that is
/// the inherent optimistic-trust residual, mitigated by the committed window /
/// timeout and the slashable bond.)
public fun output_is_bound(
    f: &Fact,
    regs: vector<u8>,
    mem_root: vector<u8>,
    out_page: vector<u8>,
    out_sibs: vector<vector<u8>>,
): bool {
    if (regs.length() != 45) { return false };
    if (merkle::state_root(&mem_root, &regs) != f.root_n) { return false };
    let halted = regs[4];
    let step = opml::bytes_le::u64_at(&regs, 5);
    halted == 1 && step == f.n && output_matches(f, &mem_root, &out_page, &out_sibs)
}

/// Judge identity: a market binds itself to (program_root, genesis_root)
/// so a Fact can only resolve it if it ran THIS judge on THIS input.
public fun program_root(f: &Fact): vector<u8> { f.program_root }

public fun genesis_root(f: &Fact): vector<u8> { f.genesis_root }

/// Base address of the output region (SPEC §7.3). `output` is self-describing
/// (`[n][tokens]`); this is exposed for callers that re-derive the region.
public fun out_base(f: &Fact): u64 { f.out_base }

/// The challenge window (ms) the resolver committed at assertion. A consumer
/// that trusts `is_finalized` as proof of the true extraction MUST also check
/// this was long enough for a real challenge to occur — a Fact asserted with
/// window_ms=0 finalizes in the same transaction and proves nothing (SPEC §8.2).
public fun window_ms(f: &Fact): u64 { f.window_ms }

/// The per-move timeout (ms) of the bisection game. Bounds the OTHER path to
/// FINALIZED (settle via claim_timeout): a tiny self-chosen timeout lets a
/// self-challenging resolver drive its own Fact to FINALIZED with no real
/// independent challenger. Consumers gate on a committed minimum.
public fun timeout_ms(f: &Fact): u64 { f.timeout_ms }

public fun depth(f: &Fact): u8 { f.d }

public fun n_steps(f: &Fact): u64 { f.n }

/// True once the claim has stood (unchallenged finalize, or challenger lost).
public fun is_finalized(f: &Fact): bool { f.status == STATUS_FINALIZED }

/// True once a challenger has proven the claim fraudulent.
public fun is_rejected(f: &Fact): bool { f.status == STATUS_REJECTED }

/// TEST-ONLY: build and share a FINALIZED Fact directly, bypassing the
/// assert→challenge→bisection flow, so a market's drop_misextracted binding
/// can be tested. Real Facts only reach FINALIZED via finalize()/settle().
#[test_only]
public fun share_finalized_fact_for_testing(
    d: u8,
    p: u8,
    program_root: vector<u8>,
    genesis_root: vector<u8>,
    out_base: u64,
    output: vector<u8>,
    root_n: vector<u8>, // committed final-state root (for output_is_bound tests)
    window_ms: u64,  // committed challenge window — a consumer may require a minimum
    timeout_ms: u64, // committed per-move timeout — likewise
    ctx: &mut TxContext,
) {
    transfer::share_object(Fact {
        id: object::new(ctx),
        d,
        p,
        program_root,
        genesis_root,
        out_base,
        n: 1,
        root_n,
        output,
        resolver: ctx.sender(),
        challenger: @0x0,
        pot: sui::balance::zero<SUI>(),
        bond_value: 0,
        created_ms: 0,
        window_ms,
        timeout_ms,
        status: STATUS_FINALIZED,
        lo: 0,
        hi: 0,
        root_lo: vector[],
        root_hi: vector[],
        has_pending: false,
        pending_mid: vector[],
        challenger_turn: false,
        deadline_ms: 0,
    });
}
