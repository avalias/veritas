/// market.move — "Polymarket of the future": a prediction market whose
/// outcome is a PURE FUNCTION of provenance-verified evidence, decided by
/// a committed rule. The fraud-provable LLM judge is the per-item
/// EXTRACTION layer (EVIDENCE.md §5).
///
/// The whole product in one object. Lifecycle (all discretion is fixed at
/// CREATION; nothing is decided by a human after money is at stake — see
/// EVIDENCE.md §1):
///
///   TRADING   create_market → buy_yes / buy_no   (a solvent complete-set
///             CPMM: every share pair is backed 1:1 by SUI collateral)
///   EVIDENCE  submit_evidence   (admissible ONLY with a Web Credential —
///             a native ed25519 signature by a pinned issuer over the
///             claim; nobody can submit "whatever they want")
///   RESOLVE   resolve   (apply the COMMITTED decision rule to the count
///             of confirmations, counted at the trust-GROUP level)
///   REDEEM    redeem   (winning shares pay 1 SUI each; a void refunds the
///             trader's full stake; the LP recovers the residual)
///
/// NEXT MILESTONE — the on-chain extraction backstop. Each item's YES/NO
/// claim is the judge's deterministic reading of signed content, and a
/// wrong reading is convicted by the bisection game (proven independently:
/// dispute/tests/fqwen_conviction.move). Auto-dropping a mis-extracted
/// item on-chain soundly requires a FINALIZED counter-extraction `Fact`
/// (positive proof the true extraction is D≠claim) bound to the item via
/// on-chain GENESIS CONSTRUCTION from the signed content (SPEC §7.2,
/// postponed). A `Fact` that is merely REJECTED proves only that its
/// asserter lied about the final root — its `output` is attacker-chosen
/// and says nothing about the true extraction — so it must NOT gate the
/// count. v1 therefore counts every admitted item and leaves the on-chain
/// auto-drop to that milestone; the extraction stays publicly checkable
/// (the judge is public + deterministic over public signed content).
module dispute::market;

use sui::balance::Balance;
use sui::coin::{Self, Coin};
use sui::sui::SUI;
use sui::clock::Clock;
use sui::event;
use sui::hash;

// -- outcomes --------------------------------------------------------------
const OUTCOME_OPEN: u8 = 0; // not yet resolved
const OUTCOME_YES: u8 = 1;
const OUTCOME_NO: u8 = 2;
const OUTCOME_UNRESOLVED: u8 = 3; // void → traders refunded their stake

// -- burden of proof (committed per market; EVIDENCE.md §2) ----------------
/// "Did X happen by T?" — YES needs ≥k group-confirmations; silence ⇒ NO.
const BURDEN_OCCURRENCE: u8 = 0;
/// "Is X true?" — each side needs ≥k; if neither reaches k ⇒ UNRESOLVED.
const BURDEN_STATE: u8 = 1;

// -- phases ----------------------------------------------------------------
const PHASE_TRADING: u8 = 0;
const PHASE_EVIDENCE: u8 = 1;
const PHASE_RESOLVED: u8 = 2;

// -- claim tags an evidence item can assert --------------------------------
const CLAIM_YES: u8 = 1;
const CLAIM_NO: u8 = 2;

/// Cap on admitted evidence items: bounds the resolve() loop's gas and
/// stops a captured issuer from flooding the market (EVIDENCE.md §4 row 7).
/// Per-trust-group quotas are the production refinement.
const MAX_EVIDENCE: u64 = 256;

// -- errors ----------------------------------------------------------------
const E_BAD_PARAMS: u64 = 1;
const E_WRONG_PHASE: u64 = 2;
const E_TOO_EARLY: u64 = 3;
const E_BAD_ISSUER: u64 = 5;
const E_BAD_SIGNATURE: u64 = 6;
const E_DUPLICATE_EVIDENCE: u64 = 7;
const E_BAD_CLAIM: u64 = 8;
const E_NOT_RESOLVED: u64 = 9;
const E_NOTHING_TO_REDEEM: u64 = 10;
const E_ZERO: u64 = 11;
const E_OUT_OF_WINDOW: u64 = 12;
const E_DUP_KEY: u64 = 16;
const E_K_UNSATISFIABLE: u64 = 17;
const E_BAD_TIME: u64 = 18;
const E_TOO_MANY_EVIDENCE: u64 = 19;
const E_NOT_LP: u64 = 20;
const E_BAD_SCHEME: u64 = 21;

/// An admitted piece of evidence: a signed claim by a pinned issuer. Stored
/// so resolution is a transparent, recomputable function of on-chain bytes.
public struct EvidenceItem has store {
    issuer_idx: u64, // index into issuer_keys
    group: u64, // trust group of the issuer (independence accounting)
    claim: u8, // CLAIM_YES | CLAIM_NO (the judge's extraction)
    content_hash: vector<u8>, // 32-byte id of the signed content
    signed_ms: u64, // issuer-asserted timestamp (must fall in the window)
    submitter: address,
}

/// A user's position. `paid` is the gross SUI they put in, so a VOID can
/// refund their exact stake even though the CPMM gives one-sided holdings.
public struct Position has store {
    yes: u64,
    no: u64,
    paid: u64,
}

/// Minted to the market creator: the right to recover the seed liquidity
/// (on void) or the pool inventory + accrued fees (on a decisive outcome)
/// once the market is resolved, never touching collateral owed to winners.
public struct LPCap has key, store {
    id: UID,
    market: address,
}

public struct Market has key {
    id: UID,
    // -- the question + the committed judge spec (fixed at creation) -------
    question: vector<u8>,
    judge_program_root: vector<u8>, // which model+prompt extracts claims
    judge_depth: u8,
    // -- source policy (EVIDENCE.md §2): who may be heard, and how counted -
    issuer_keys: vector<vector<u8>>, // pinned pubkeys (distinct)
    issuer_schemes: vector<u8>, // issuer_idx → credential scheme (ed25519 | ES256/C2PA)
    issuer_groups: vector<u64>, // issuer_idx → trust group id
    k: u64, // confirmations required, counted at the GROUP level
    burden: u8, // BURDEN_OCCURRENCE | BURDEN_STATE
    // -- timing (immutable signed snapshots only; EVIDENCE.md §1) ----------
    resolve_after_ms: u64, // evidence window opens (trading closes)
    evidence_window_ms: u64, // length of the evidence window
    created_ms: u64,
    // -- AMM: a solvent complete-set CPMM ---------------------------------
    collateral: Balance<SUI>,
    reserve_yes: u64, // pool inventory (constant product with reserve_no)
    reserve_no: u64,
    fee_bps: u64,
    positions: sui::table::Table<address, Position>,
    // liability tracking so the LP can withdraw only the TRUE residual,
    // never collateral owed to un-redeemed winners.
    user_yes: u64, // outstanding user-held YES shares
    user_no: u64, // outstanding user-held NO shares
    user_paid: u64, // outstanding user stake (for void refunds)
    // -- evidence + resolution --------------------------------------------
    seen: sui::table::Table<vector<u8>, bool>, // content_hash dedup set
    evidence: vector<EvidenceItem>,
    yes_groups: vector<u64>, // distinct trust groups that asserted YES
    no_groups: vector<u64>,
    phase: u8,
    outcome: u8,
}

// -- events: a clean, demoable on-chain narrative --------------------------
public struct MarketCreated has copy, drop { market: address, k: u64, burden: u8 }
public struct Traded has copy, drop { market: address, who: address, yes: bool, paid: u64, shares: u64 }
public struct EvidenceAdmitted has copy, drop {
    market: address, issuer_idx: u64, group: u64, claim: u8, content_hash: vector<u8>,
}
public struct Resolved has copy, drop { market: address, outcome: u8, yes_groups: u64, no_groups: u64 }
public struct Redeemed has copy, drop { market: address, who: address, payout: u64 }
public struct ResidualWithdrawn has copy, drop { market: address, amount: u64 }

// =========================================================================
// CREATION — every resolution parameter is fixed here, before any trade.
// =========================================================================
#[allow(lint(self_transfer))] // the creator IS the intended LPCap recipient
public fun create_market(
    question: vector<u8>,
    judge_program_root: vector<u8>,
    judge_depth: u8,
    issuer_keys: vector<vector<u8>>,
    issuer_schemes: vector<u8>,
    issuer_groups: vector<u64>,
    k: u64,
    burden: u8,
    resolve_after_ms: u64,
    evidence_window_ms: u64,
    fee_bps: u64,
    seed: Coin<SUI>, // initial liquidity → seeds both AMM reserves
    clock: &Clock,
    ctx: &mut TxContext,
): address {
    let n = issuer_keys.length();
    assert!(n > 0 && issuer_groups.length() == n && issuer_schemes.length() == n, E_BAD_PARAMS);
    assert!(burden == BURDEN_OCCURRENCE || burden == BURDEN_STATE, E_BAD_PARAMS);
    assert!(evidence_window_ms > 0 && fee_bps < 10000, E_BAD_PARAMS);
    let now = clock.timestamp_ms();
    // there must be a real trading phase before evidence opens
    assert!(resolve_after_ms > now, E_BAD_TIME);

    // each issuer key must be a natively-verifiable Web Credential key, and
    // keys must be DISTINCT — otherwise one key placed in two groups would
    // forge "independence".
    let mut distinct_groups = vector<u64>[];
    let mut i = 0;
    while (i < n) {
        assert!(dispute::credential::is_native(issuer_schemes[i]), E_BAD_SCHEME);
        assert!(valid_key_len(issuer_schemes[i], issuer_keys[i].length()), E_BAD_PARAMS);
        let mut j = i + 1;
        while (j < n) { assert!(issuer_keys[i] != issuer_keys[j], E_DUP_KEY); j = j + 1; };
        push_unique(&mut distinct_groups, issuer_groups[i]);
        i = i + 1;
    };
    // k is counted at the GROUP level in resolve(), so it must be satisfiable
    // by the number of DISTINCT groups, not merely the issuer count.
    assert!(k > 0 && k <= distinct_groups.length(), E_K_UNSATISFIABLE);

    let l = seed.value();
    assert!(l > 0, E_ZERO);

    let id = object::new(ctx);
    let market_addr = id.to_address();
    // Seed L SUI ⇒ L complete sets in the pool: reserve_yes = reserve_no = L,
    // collateral = L. Solvent by construction (total YES == total NO == L).
    let m = Market {
        id,
        question,
        judge_program_root,
        judge_depth,
        issuer_keys,
        issuer_schemes,
        issuer_groups,
        k,
        burden,
        resolve_after_ms,
        evidence_window_ms,
        created_ms: now,
        collateral: seed.into_balance(),
        reserve_yes: l,
        reserve_no: l,
        fee_bps,
        positions: sui::table::new(ctx),
        user_yes: 0,
        user_no: 0,
        user_paid: 0,
        seen: sui::table::new(ctx),
        evidence: vector[],
        yes_groups: vector[],
        no_groups: vector[],
        phase: PHASE_TRADING,
        outcome: OUTCOME_OPEN,
    };
    // the creator/LP gets the right to recover their capital + fees later
    transfer::public_transfer(LPCap { id: object::new(ctx), market: market_addr }, ctx.sender());
    event::emit(MarketCreated { market: market_addr, k, burden });
    transfer::share_object(m);
    market_addr
}

/// CLI/PTB-friendly creation: shares the market, mints the LPCap to the
/// caller, and drops the returned address (recover it from the event).
public fun create_market_entry(
    question: vector<u8>,
    judge_program_root: vector<u8>,
    judge_depth: u8,
    issuer_keys: vector<vector<u8>>,
    issuer_schemes: vector<u8>,
    issuer_groups: vector<u64>,
    k: u64,
    burden: u8,
    resolve_after_ms: u64,
    evidence_window_ms: u64,
    fee_bps: u64,
    seed: Coin<SUI>,
    clock: &Clock,
    ctx: &mut TxContext,
) {
    let _ = create_market(
        question, judge_program_root, judge_depth, issuer_keys, issuer_schemes, issuer_groups,
        k, burden, resolve_after_ms, evidence_window_ms, fee_bps, seed, clock, ctx,
    );
}

// =========================================================================
// TRADING — a solvent complete-set CPMM.
//
// buy_yes(dS): mint dS complete sets to the buyer (collateral += dS), then
// swap the buyer's dS NO into the pool for YES at constant product:
// yes_out = reserve_yes * dS / (reserve_no + dS). total_YES and total_NO
// each rise by exactly dS, so total_YES == total_NO == collateral (minus
// fees, which accrue to the LP) → always solvent.
// =========================================================================
public fun buy_yes(m: &mut Market, payment: Coin<SUI>, clock: &Clock, ctx: &mut TxContext) {
    let who = ctx.sender();
    let (shares_out, paid) = trade(m, payment, true, clock, ctx);
    event::emit(Traded { market: m.id.to_address(), who, yes: true, paid, shares: shares_out });
}

public fun buy_no(m: &mut Market, payment: Coin<SUI>, clock: &Clock, ctx: &mut TxContext) {
    let who = ctx.sender();
    let (shares_out, paid) = trade(m, payment, false, clock, ctx);
    event::emit(Traded { market: m.id.to_address(), who, yes: false, paid, shares: shares_out });
}

/// Returns (shares_out, complete_sets_minted). Internal so buy_yes/buy_no
/// share code.
fun trade(m: &mut Market, payment: Coin<SUI>, yes: bool, clock: &Clock, ctx: &TxContext): (u64, u64) {
    assert!(phase_now(m, clock) == PHASE_TRADING, E_WRONG_PHASE);
    let gross = payment.value();
    assert!(gross > 0, E_ZERO);
    let fee = mul_div(gross, m.fee_bps, 10000); // u128 inside → no overflow
    let ds = gross - fee; // complete sets minted (fee stays in collateral for the LP)
    assert!(ds > 0, E_ZERO);
    m.collateral.join(payment.into_balance());

    let shares_out;
    if (yes) {
        let yes_out = mul_div(m.reserve_yes, ds, m.reserve_no + ds);
        m.reserve_no = m.reserve_no + ds;
        m.reserve_yes = m.reserve_yes - yes_out;
        shares_out = ds + yes_out;
    } else {
        let no_out = mul_div(m.reserve_no, ds, m.reserve_yes + ds);
        m.reserve_yes = m.reserve_yes + ds;
        m.reserve_no = m.reserve_no - no_out;
        shares_out = ds + no_out;
    };

    let who = ctx.sender();
    ensure_position(m, who);
    let p = &mut m.positions[who];
    p.paid = p.paid + gross;
    if (yes) { p.yes = p.yes + shares_out } else { p.no = p.no + shares_out };
    m.user_paid = m.user_paid + gross;
    if (yes) { m.user_yes = m.user_yes + shares_out } else { m.user_no = m.user_no + shares_out };
    (shares_out, ds)
}

// =========================================================================
// EVIDENCE — provenance-gated admission. The ONLY way bytes enter.
//
// Admissible iff it carries a Web Credential: a native ed25519 signature by
// a pinned issuer over blake2b256(market || claim || content_hash || ms_le).
// =========================================================================
public fun submit_evidence(
    m: &mut Market,
    issuer_idx: u64,
    claim: u8,
    content_hash: vector<u8>,
    signed_ms: u64,
    signature: vector<u8>,
    clock: &Clock,
    ctx: &TxContext,
) {
    assert!(phase_now(m, clock) == PHASE_EVIDENCE, E_WRONG_PHASE);
    assert!(claim == CLAIM_YES || claim == CLAIM_NO, E_BAD_CLAIM);
    assert!(issuer_idx < m.issuer_keys.length(), E_BAD_ISSUER);
    assert!(content_hash.length() == 32, E_BAD_PARAMS);
    assert!(m.evidence.length() < MAX_EVIDENCE, E_TOO_MANY_EVIDENCE);
    // the signed timestamp must fall inside this market's evidence window
    assert!(signed_ms >= m.resolve_after_ms, E_OUT_OF_WINDOW);
    assert!(signed_ms < m.resolve_after_ms + m.evidence_window_ms, E_OUT_OF_WINDOW);
    // dedup: the same signed content counts once (majority-by-duplication killed)
    assert!(!m.seen.contains(content_hash), E_DUPLICATE_EVIDENCE);

    let msg = canonical_message(m.id.to_address(), claim, &content_hash, signed_ms);
    let ok = dispute::credential::verify(
        m.issuer_schemes[issuer_idx], &m.issuer_keys[issuer_idx], &msg, &signature,
    );
    assert!(ok, E_BAD_SIGNATURE);

    admit_core(m, issuer_idx, claim, content_hash, signed_ms, ctx.sender());
}

/// Record an admitted item (after all validation incl. the signature).
fun admit_core(m: &mut Market, issuer_idx: u64, claim: u8, content_hash: vector<u8>, signed_ms: u64, who: address) {
    let group = m.issuer_groups[issuer_idx];
    m.seen.add(content_hash, true);
    m.evidence.push_back(EvidenceItem { issuer_idx, group, claim, content_hash, signed_ms, submitter: who });
    event::emit(EvidenceAdmitted { market: m.id.to_address(), issuer_idx, group, claim, content_hash });
}

// =========================================================================
// RESOLVE — apply the COMMITTED decision rule. Pure function of the
// admitted evidence, counted at the trust-GROUP level.
// =========================================================================
public fun resolve(m: &mut Market, clock: &Clock) {
    let now = clock.timestamp_ms();
    assert!(m.phase != PHASE_RESOLVED, E_WRONG_PHASE);
    assert!(now >= m.resolve_after_ms + m.evidence_window_ms, E_TOO_EARLY);

    let mut yes_groups = vector<u64>[];
    let mut no_groups = vector<u64>[];
    let mut i = 0;
    let n = m.evidence.length();
    while (i < n) {
        let it = &m.evidence[i];
        if (it.claim == CLAIM_YES) { push_unique(&mut yes_groups, it.group); }
        else { push_unique(&mut no_groups, it.group); };
        i = i + 1;
    };
    let ny = yes_groups.length();
    let nn = no_groups.length();

    let outcome = if (m.burden == BURDEN_OCCURRENCE) {
        if (ny >= m.k) { OUTCOME_YES } else { OUTCOME_NO }
    } else {
        if (ny >= m.k && ny > nn) { OUTCOME_YES }
        else if (nn >= m.k && nn > ny) { OUTCOME_NO }
        else { OUTCOME_UNRESOLVED }
    };

    m.yes_groups = yes_groups;
    m.no_groups = no_groups;
    m.outcome = outcome;
    m.phase = PHASE_RESOLVED;
    event::emit(Resolved { market: m.id.to_address(), outcome, yes_groups: ny, no_groups: nn });
}

// =========================================================================
// REDEEM — winning shares pay 1 SUI each. On VOID, the trader's FULL stake
// is refunded (not just matched complete sets) so a one-sided CPMM holder
// is never locked out — the exact harm the design exists to remove.
// =========================================================================
public fun redeem(m: &mut Market, ctx: &mut TxContext): Coin<SUI> {
    assert!(m.phase == PHASE_RESOLVED, E_NOT_RESOLVED);
    let who = ctx.sender();
    assert!(m.positions.contains(who), E_NOTHING_TO_REDEEM);
    let p = &mut m.positions[who];
    let payout;
    if (m.outcome == OUTCOME_YES) {
        payout = p.yes;
        m.user_yes = m.user_yes - p.yes;
    } else if (m.outcome == OUTCOME_NO) {
        payout = p.no;
        m.user_no = m.user_no - p.no;
    } else {
        // VOID: refund the trader's full stake.
        payout = p.paid;
        m.user_paid = m.user_paid - p.paid;
    };
    p.yes = 0;
    p.no = 0;
    p.paid = 0;
    assert!(payout > 0, E_NOTHING_TO_REDEEM);
    event::emit(Redeemed { market: m.id.to_address(), who, payout });
    coin::from_balance(m.collateral.split(payout), ctx)
}

/// CLI/PTB-friendly redeem: sends the payout coin to the caller.
#[allow(lint(self_transfer))]
public fun redeem_to_sender(m: &mut Market, ctx: &mut TxContext) {
    let c = redeem(m, ctx);
    transfer::public_transfer(c, ctx.sender());
}

// =========================================================================
// LP SETTLEMENT — the creator recovers the residual collateral (seed on a
// void; pool inventory + accrued fees on a decisive outcome) WITHOUT ever
// touching collateral owed to un-redeemed winners. Solvency preserved:
// collateral >= current winner liability at all times, so the residual is
// exactly collateral − liability.
// =========================================================================
public fun withdraw_residual(m: &mut Market, cap: &LPCap, ctx: &mut TxContext): Coin<SUI> {
    assert!(m.phase == PHASE_RESOLVED, E_NOT_RESOLVED);
    assert!(cap.market == m.id.to_address(), E_NOT_LP);
    let liability = current_liability(m);
    let claimable = m.collateral.value() - liability;
    assert!(claimable > 0, E_NOTHING_TO_REDEEM);
    event::emit(ResidualWithdrawn { market: m.id.to_address(), amount: claimable });
    coin::from_balance(m.collateral.split(claimable), ctx)
}

#[allow(lint(self_transfer))]
public fun withdraw_residual_to_sender(m: &mut Market, cap: &LPCap, ctx: &mut TxContext) {
    let c = withdraw_residual(m, cap, ctx);
    transfer::public_transfer(c, ctx.sender());
}

/// Collateral still owed to un-redeemed winners (or, on void, traders).
fun current_liability(m: &Market): u64 {
    if (m.outcome == OUTCOME_YES) { m.user_yes }
    else if (m.outcome == OUTCOME_NO) { m.user_no }
    else { m.user_paid } // OUTCOME_UNRESOLVED
}

// =========================================================================
// helpers
// =========================================================================
fun phase_now(m: &Market, clock: &Clock): u8 {
    if (m.phase == PHASE_RESOLVED) return PHASE_RESOLVED;
    let now = clock.timestamp_ms();
    if (now < m.resolve_after_ms) PHASE_TRADING
    else if (now < m.resolve_after_ms + m.evidence_window_ms) PHASE_EVIDENCE
    else PHASE_RESOLVED // window closed but resolve() not yet called
}

/// Canonical signed message: blake2b( market || claim || hash || ms_le ).
fun canonical_message(market: address, claim: u8, content_hash: &vector<u8>, signed_ms: u64): vector<u8> {
    let mut pre = market.to_bytes();
    pre.push_back(claim);
    pre.append(*content_hash);
    pre.append(u64_le(signed_ms));
    hash::blake2b256(&pre)
}

fun u64_le(mut x: u64): vector<u8> {
    let mut out = vector<u8>[];
    let mut i = 0;
    while (i < 8) { out.push_back(((x & 0xff) as u8)); x = x >> 8; i = i + 1; };
    out
}

fun ensure_position(m: &mut Market, who: address) {
    if (!m.positions.contains(who)) { m.positions.add(who, Position { yes: 0, no: 0, paid: 0 }); };
}

/// Expected pubkey length for a credential scheme (ed25519 = 32, ES256
/// compressed P-256 = 33).
fun valid_key_len(scheme: u8, len: u64): bool {
    if (scheme == dispute::credential::scheme_ed25519()) { len == 32 }
    else if (scheme == dispute::credential::scheme_es256()) { len == 33 }
    else { false }
}

fun push_unique(v: &mut vector<u64>, x: u64) {
    let mut i = 0;
    let n = v.length();
    while (i < n) { if (v[i] == x) return; i = i + 1; };
    v.push_back(x);
}

/// floor(a * b / c) in u128 to avoid overflow on the CPMM product / fees.
fun mul_div(a: u64, b: u64, c: u64): u64 {
    (((a as u128) * (b as u128)) / (c as u128)) as u64
}

// -- read-only accessors (clients / UI / tests) ---------------------------
public fun phase(m: &Market): u8 { m.phase }
public fun outcome(m: &Market): u8 { m.outcome }
public fun collateral_value(m: &Market): u64 { m.collateral.value() }
public fun reserves(m: &Market): (u64, u64) { (m.reserve_yes, m.reserve_no) }
/// Spot price of YES in basis points: reserve_no / (reserve_yes + reserve_no).
public fun price_yes_bps(m: &Market): u64 { mul_div(m.reserve_no, 10000, m.reserve_yes + m.reserve_no) }
public fun evidence_count(m: &Market): u64 { m.evidence.length() }
public fun group_counts(m: &Market): (u64, u64) { (m.yes_groups.length(), m.no_groups.length()) }
public fun position_of(m: &Market, who: address): (u64, u64, u64) {
    if (!m.positions.contains(who)) return (0, 0, 0);
    let p = &m.positions[who];
    (p.yes, p.no, p.paid)
}

// -- test-only hooks (the real signed path runs on localnet) ---------------

/// Admit an item bypassing ONLY the signature check (all other validation —
/// phase, window, claim, issuer, dedup, cap — is identical).
#[test_only]
public fun test_admit(
    m: &mut Market, issuer_idx: u64, claim: u8, content_hash: vector<u8>, signed_ms: u64,
    clock: &Clock, ctx: &TxContext,
) {
    assert!(phase_now(m, clock) == PHASE_EVIDENCE, E_WRONG_PHASE);
    assert!(claim == CLAIM_YES || claim == CLAIM_NO, E_BAD_CLAIM);
    assert!(issuer_idx < m.issuer_keys.length(), E_BAD_ISSUER);
    assert!(content_hash.length() == 32, E_BAD_PARAMS);
    assert!(m.evidence.length() < MAX_EVIDENCE, E_TOO_MANY_EVIDENCE);
    assert!(signed_ms >= m.resolve_after_ms, E_OUT_OF_WINDOW);
    assert!(signed_ms < m.resolve_after_ms + m.evidence_window_ms, E_OUT_OF_WINDOW);
    assert!(!m.seen.contains(content_hash), E_DUPLICATE_EVIDENCE);
    admit_core(m, issuer_idx, claim, content_hash, signed_ms, ctx.sender());
}

/// Distinct trust groups per side, computed live (pre-resolve).
#[test_only]
public fun group_counts_live(m: &Market): (u64, u64) {
    let mut yg = vector<u64>[];
    let mut ng = vector<u64>[];
    let mut i = 0;
    let n = m.evidence.length();
    while (i < n) {
        let it = &m.evidence[i];
        if (it.claim == CLAIM_YES) { push_unique(&mut yg, it.group); }
        else { push_unique(&mut ng, it.group); };
        i = i + 1;
    };
    (yg.length(), ng.length())
}

#[test_only]
public fun test_is_seen(m: &Market, content_hash: vector<u8>): bool { m.seen.contains(content_hash) }

#[test_only]
public fun test_canonical_message(market: address, claim: u8, content_hash: vector<u8>, signed_ms: u64): vector<u8> {
    canonical_message(market, claim, &content_hash, signed_ms)
}
