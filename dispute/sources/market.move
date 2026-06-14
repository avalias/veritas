/// market.move — "Polymarket of the future": a prediction market whose
/// outcome is a PURE FUNCTION of provenance-verified evidence, decided by
/// a committed rule, with the fraud-provable LLM judge as the backstop.
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
///   REDEEM    redeem   (winning shares pay 1 SUI each from collateral)
///
/// The AI judge is the per-item EXTRACTION layer (EVIDENCE.md §5): each
/// evidence item's YES/NO claim is the judge's deterministic reading of
/// signed content, and a wrong reading is convicted by the bisection game
/// (`dispute::Fact`). `challenge_item` wires that backstop in: an item
/// whose extraction Fact was REJECTED is dropped from the count.
module dispute::market;

use dispute::dispute::Fact;
use sui::balance::Balance;
use sui::coin::{Self, Coin};
use sui::sui::SUI;
use sui::clock::Clock;
use sui::ed25519;
use sui::event;
use sui::hash;
use std::u64;

// -- outcomes --------------------------------------------------------------
const OUTCOME_OPEN: u8 = 0; // not yet resolved
const OUTCOME_YES: u8 = 1;
const OUTCOME_NO: u8 = 2;
const OUTCOME_UNRESOLVED: u8 = 3; // void → stakes refundable as complete sets

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
const E_FACT_MISMATCH: u64 = 13;
const E_NOT_REJECTED: u64 = 14;
const E_ALREADY_FLAGGED: u64 = 15;

/// An admitted piece of evidence: a signed claim by a pinned issuer. Stored
/// so resolution is a transparent, recomputable function of on-chain bytes.
public struct EvidenceItem has store {
    issuer_idx: u64, // index into SourcePolicy.issuer_keys
    group: u64, // trust group of the issuer (independence accounting)
    claim: u8, // CLAIM_YES | CLAIM_NO (the judge's extraction)
    content_hash: vector<u8>, // 32-byte id of the signed content
    signed_ms: u64, // issuer-asserted timestamp (must fall in the window)
    submitter: address,
    flagged: bool, // dropped from the count iff a REJECTED Fact proved the extraction wrong
}

/// A user's position. Solvency invariant: total YES == total NO ==
/// collateral, because every share pair was minted from exactly 1 SUI.
public struct Position has store {
    yes: u64,
    no: u64,
}

public struct Market has key {
    id: UID,
    // -- the question + the committed judge spec (fixed at creation) -------
    question: vector<u8>,
    judge_program_root: vector<u8>, // which model+prompt extracts claims
    judge_depth: u8,
    // -- source policy (EVIDENCE.md §2): who may be heard, and how counted -
    issuer_keys: vector<vector<u8>>, // pinned ed25519 pubkeys (32 bytes each)
    issuer_groups: vector<u64>, // issuer_idx → trust group id (AP+syndicators share one)
    k: u64, // confirmations required, counted at the GROUP level
    burden: u8, // BURDEN_OCCURRENCE | BURDEN_STATE
    // -- timing (immutable signed snapshots only; EVIDENCE.md §1) ----------
    resolve_after_ms: u64, // evidence window opens (trading closes)
    evidence_window_ms: u64, // length of the evidence window
    created_ms: u64,
    // -- AMM: a solvent complete-set CPMM ---------------------------------
    collateral: Balance<SUI>, // == total YES == total NO outstanding
    reserve_yes: u64, // pool inventory (constant product with reserve_no)
    reserve_no: u64,
    fee_bps: u64,
    positions: sui::table::Table<address, Position>,
    // -- evidence + resolution --------------------------------------------
    seen: sui::table::Table<vector<u8>, bool>, // content_hash dedup set
    evidence: vector<EvidenceItem>,
    yes_groups: vector<u64>, // distinct trust groups that asserted YES (deduped)
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
public struct ItemFlagged has copy, drop { market: address, content_hash: vector<u8> }
public struct Resolved has copy, drop { market: address, outcome: u8, yes_groups: u64, no_groups: u64 }
public struct Redeemed has copy, drop { market: address, who: address, payout: u64 }

// =========================================================================
// CREATION — every resolution parameter is fixed here, before any trade.
// =========================================================================
public fun create_market(
    question: vector<u8>,
    judge_program_root: vector<u8>,
    judge_depth: u8,
    issuer_keys: vector<vector<u8>>,
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
    let n_issuers = issuer_keys.length();
    assert!(n_issuers > 0 && issuer_groups.length() == n_issuers, E_BAD_PARAMS);
    assert!(k > 0 && k <= n_issuers, E_BAD_PARAMS);
    assert!(burden == BURDEN_OCCURRENCE || burden == BURDEN_STATE, E_BAD_PARAMS);
    assert!(evidence_window_ms > 0 && fee_bps < 10000, E_BAD_PARAMS);
    let l = seed.value();
    assert!(l > 0, E_ZERO);
    // every issuer key is a 32-byte ed25519 public key
    let mut i = 0;
    while (i < n_issuers) { assert!(issuer_keys[i].length() == 32, E_BAD_PARAMS); i = i + 1; };

    let id = object::new(ctx);
    let market_addr = id.to_address();
    // Seed L SUI ⇒ L complete sets into the pool: reserve_yes = reserve_no = L,
    // collateral = L. Solvent by construction (total YES == total NO == L).
    let m = Market {
        id,
        question,
        judge_program_root,
        judge_depth,
        issuer_keys,
        issuer_groups,
        k,
        burden,
        resolve_after_ms,
        evidence_window_ms,
        created_ms: clock.timestamp_ms(),
        collateral: seed.into_balance(),
        reserve_yes: l,
        reserve_no: l,
        fee_bps,
        positions: sui::table::new(ctx),
        seen: sui::table::new(ctx),
        evidence: vector[],
        yes_groups: vector[],
        no_groups: vector[],
        phase: PHASE_TRADING,
        outcome: OUTCOME_OPEN,
    };
    event::emit(MarketCreated { market: market_addr, k, burden });
    transfer::share_object(m);
    market_addr
}

// =========================================================================
// TRADING — a solvent complete-set CPMM.
//
// buy_yes(dS): mint dS complete sets to the buyer (collateral += dS), then
// swap the buyer's dS NO into the pool for YES at constant product. Closed
// form: yes_out = reserve_yes * dS / (reserve_no + dS). The buyer ends with
// dS + yes_out YES and 0 NO. total_YES and total_NO each rise by exactly dS,
// so total_YES == total_NO == collateral is preserved → always solvent.
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

/// Returns (shares_out, net_paid). Internal so buy_yes/buy_no share code.
fun trade(m: &mut Market, payment: Coin<SUI>, yes: bool, clock: &Clock, ctx: &TxContext): (u64, u64) {
    assert!(phase_now(m, clock) == PHASE_TRADING, E_WRONG_PHASE);
    let gross = payment.value();
    assert!(gross > 0, E_ZERO);
    let fee = gross * m.fee_bps / 10000;
    let ds = gross - fee; // complete sets minted (fee stays in collateral as PnL for LPs)
    assert!(ds > 0, E_ZERO);
    m.collateral.join(payment.into_balance());

    // constant-product swap of the minted opposite-side shares into the pool
    let shares_out;
    if (yes) {
        // yes_out = reserve_yes * ds / (reserve_no + ds); pool: +ds NO, -yes_out YES
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
    if (yes) { p.yes = p.yes + shares_out } else { p.no = p.no + shares_out };
    (shares_out, ds)
}

// =========================================================================
// EVIDENCE — provenance-gated admission. The ONLY way bytes enter.
//
// A submission is admissible iff it carries a Web Credential: a native
// ed25519 signature by a pinned issuer over the canonical message
//   blake2b( market_addr || claim || content_hash || signed_ms_le ).
// Nobody can submit "whatever they want": the admissible universe is
// exactly what the pinned issuers actually signed.
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
    // the signed timestamp must fall inside this market's evidence window
    assert!(signed_ms >= m.resolve_after_ms, E_OUT_OF_WINDOW);
    assert!(signed_ms < m.resolve_after_ms + m.evidence_window_ms, E_OUT_OF_WINDOW);
    // dedup: the same signed content counts once (majority-by-duplication killed)
    assert!(!m.seen.contains(content_hash), E_DUPLICATE_EVIDENCE);

    // verify the issuer's signature over the canonical message
    let msg = canonical_message(m.id.to_address(), claim, &content_hash, signed_ms);
    let ok = ed25519::ed25519_verify(&signature, &m.issuer_keys[issuer_idx], &msg);
    assert!(ok, E_BAD_SIGNATURE);

    admit_core(m, issuer_idx, claim, content_hash, signed_ms, ctx.sender());
}

/// Record an admitted item (after all validation incl. the signature). The
/// trust-group is looked up here so independence accounting is centralized.
fun admit_core(m: &mut Market, issuer_idx: u64, claim: u8, content_hash: vector<u8>, signed_ms: u64, who: address) {
    let group = m.issuer_groups[issuer_idx];
    m.seen.add(content_hash, true);
    m.evidence.push_back(EvidenceItem {
        issuer_idx, group, claim, content_hash, signed_ms, submitter: who, flagged: false,
    });
    event::emit(EvidenceAdmitted { market: m.id.to_address(), issuer_idx, group, claim, content_hash });
}

/// The fraud-proof backstop (EVIDENCE.md §4 row 2 / §5). If a judge
/// EXTRACTION was wrong — the signed content does not actually assert what
/// the item claims — anyone disputes it via the bisection game; once that
/// `Fact` is REJECTED, the item is permanently dropped from the count.
///
/// The Fact must be the extraction run for THIS item (its `output` is the
/// item's content_hash || claim) under THIS market's judge.
public fun challenge_item(
    m: &mut Market,
    content_hash: vector<u8>,
    extraction_fact: &Fact,
    clock: &Clock,
) {
    assert!(phase_now(m, clock) != PHASE_RESOLVED, E_WRONG_PHASE);
    assert!(dispute::dispute::is_rejected(extraction_fact), E_NOT_REJECTED);
    // bind: the Fact must be the extraction of this item under this judge
    assert!(dispute::dispute::program_root(extraction_fact) == m.judge_program_root, E_FACT_MISMATCH);
    let mut i = 0;
    let n = m.evidence.length();
    while (i < n) {
        let it = &mut m.evidence[i];
        if (it.content_hash == content_hash) {
            assert!(!it.flagged, E_ALREADY_FLAGGED);
            // the Fact's committed output must name this exact item+claim
            let mut want = content_hash;
            want.push_back(it.claim);
            assert!(dispute::dispute::output(extraction_fact) == want, E_FACT_MISMATCH);
            it.flagged = true;
            event::emit(ItemFlagged { market: m.id.to_address(), content_hash });
            return
        };
        i = i + 1;
    };
    abort E_FACT_MISMATCH
}

// =========================================================================
// RESOLVE — apply the COMMITTED decision rule. Pure function of the
// admitted (unflagged) evidence, counted at the trust-GROUP level.
// =========================================================================
public fun resolve(m: &mut Market, clock: &Clock) {
    let now = clock.timestamp_ms();
    assert!(m.phase != PHASE_RESOLVED, E_WRONG_PHASE);
    assert!(now >= m.resolve_after_ms + m.evidence_window_ms, E_TOO_EARLY);

    // count DISTINCT trust groups asserting each side, ignoring flagged items
    let mut yes_groups = vector<u64>[];
    let mut no_groups = vector<u64>[];
    let mut i = 0;
    let n = m.evidence.length();
    while (i < n) {
        let it = &m.evidence[i];
        if (!it.flagged) {
            if (it.claim == CLAIM_YES) { push_unique(&mut yes_groups, it.group); }
            else { push_unique(&mut no_groups, it.group); };
        };
        i = i + 1;
    };
    let ny = yes_groups.length();
    let nn = no_groups.length();

    // the committed rule (EVIDENCE.md §2)
    let outcome = if (m.burden == BURDEN_OCCURRENCE) {
        // YES iff ≥k groups confirm the occurrence; silence ⇒ NO
        if (ny >= m.k) { OUTCOME_YES } else { OUTCOME_NO }
    } else {
        // STATE: each side needs ≥k; ties / neither ⇒ UNRESOLVED (void)
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
// REDEEM — winning shares pay 1 SUI each from collateral. On UNRESOLVED,
// a complete set (min(yes,no)) refunds 1 SUI (stake returned).
// =========================================================================
public fun redeem(m: &mut Market, ctx: &mut TxContext): Coin<SUI> {
    assert!(m.phase == PHASE_RESOLVED, E_NOT_RESOLVED);
    let who = ctx.sender();
    assert!(m.positions.contains(who), E_NOTHING_TO_REDEEM);
    let p = &mut m.positions[who];
    let payout = if (m.outcome == OUTCOME_YES) {
        let v = p.yes; p.yes = 0; p.no = 0; v
    } else if (m.outcome == OUTCOME_NO) {
        let v = p.no; p.yes = 0; p.no = 0; v
    } else {
        // UNRESOLVED: refund matched complete sets at 1 SUI each
        let v = u64::min(p.yes, p.no); p.yes = p.yes - v; p.no = p.no - v; v
    };
    assert!(payout > 0, E_NOTHING_TO_REDEEM);
    event::emit(Redeemed { market: m.id.to_address(), who, payout });
    coin::from_balance(m.collateral.split(payout), ctx)
}

// =========================================================================
// helpers
// =========================================================================

/// The phase implied by the clock (trading → evidence → resolved). Once
/// `resolve` runs, the stored phase is PHASE_RESOLVED and wins.
fun phase_now(m: &Market, clock: &Clock): u8 {
    if (m.phase == PHASE_RESOLVED) return PHASE_RESOLVED;
    let now = clock.timestamp_ms();
    if (now < m.resolve_after_ms) PHASE_TRADING
    else if (now < m.resolve_after_ms + m.evidence_window_ms) PHASE_EVIDENCE
    else PHASE_RESOLVED // window closed but resolve() not yet called
}

/// Canonical signed message: blake2b( market || claim || hash || ms_le ).
/// A Web Credential is a signature by a pinned issuer over exactly this.
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
    if (!m.positions.contains(who)) { m.positions.add(who, Position { yes: 0, no: 0 }); };
}

fun push_unique(v: &mut vector<u64>, x: u64) {
    let mut i = 0;
    let n = v.length();
    while (i < n) { if (v[i] == x) return; i = i + 1; };
    v.push_back(x);
}

/// floor(a * b / c) in u128 to avoid overflow on the CPMM product.
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
public fun position_of(m: &Market, who: address): (u64, u64) {
    if (!m.positions.contains(who)) return (0, 0);
    let p = &m.positions[who];
    (p.yes, p.no)
}

// -- test-only hooks (the real signed path is exercised on localnet) -------

/// Admit an item bypassing ONLY the signature check (all other validation —
/// phase, window, claim, issuer, dedup — is identical to submit_evidence).
#[test_only]
public fun test_admit(
    m: &mut Market, issuer_idx: u64, claim: u8, content_hash: vector<u8>, signed_ms: u64,
    clock: &Clock, ctx: &TxContext,
) {
    assert!(phase_now(m, clock) == PHASE_EVIDENCE, E_WRONG_PHASE);
    assert!(claim == CLAIM_YES || claim == CLAIM_NO, E_BAD_CLAIM);
    assert!(issuer_idx < m.issuer_keys.length(), E_BAD_ISSUER);
    assert!(content_hash.length() == 32, E_BAD_PARAMS);
    assert!(signed_ms >= m.resolve_after_ms, E_OUT_OF_WINDOW);
    assert!(signed_ms < m.resolve_after_ms + m.evidence_window_ms, E_OUT_OF_WINDOW);
    assert!(!m.seen.contains(content_hash), E_DUPLICATE_EVIDENCE);
    admit_core(m, issuer_idx, claim, content_hash, signed_ms, ctx.sender());
}

/// Distinct unflagged trust groups per side, computed live (pre-resolve).
#[test_only]
public fun group_counts_live(m: &Market): (u64, u64) {
    let mut yg = vector<u64>[];
    let mut ng = vector<u64>[];
    let mut i = 0;
    let n = m.evidence.length();
    while (i < n) {
        let it = &m.evidence[i];
        if (!it.flagged) {
            if (it.claim == CLAIM_YES) { push_unique(&mut yg, it.group); }
            else { push_unique(&mut ng, it.group); };
        };
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
