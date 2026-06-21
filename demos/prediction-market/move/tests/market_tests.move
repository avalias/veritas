/// Unit tests for the market product (veritas::market). The full signed
/// end-to-end (real Web Credentials over a real market address) runs on
/// localnet via dispute/demo/market_e2e.py; here we prove AMM solvency,
/// the lifecycle, the decision rule, dedup/independence accounting, the
/// void refund + LP settlement, the creation-time guards, and that the
/// on-chain ed25519 admission matches an off-chain signer bit-for-bit.
#[test_only]
module veritas::market_tests;

use veritas::market;
use sui::clock;
use sui::coin;
use sui::sui::SUI;
use sui::test_scenario as ts;

const ADMIN: address = @0xA;
const ALICE: address = @0xA11CE;
const BOB: address = @0xB0B;
const CAROL: address = @0xCA401;

// 4 issuers, two trust groups {0,0,1,1}, k=2, occurrence; trading until
// t=1000, evidence window [1000, 2000).
fun mk_keys(): vector<vector<u8>> {
    let mut v = vector[];
    let mut i = 0u8;
    while (i < 4) {
        let mut key = vector[];
        let mut j = 0; while (j < 32) { key.push_back(i); j = j + 1; };
        v.push_back(key);
        i = i + 1;
    };
    v
}

fun fresh_market(s: &mut ts::Scenario, clock: &clock::Clock, k: u64, burden: u8): address {
    ts::next_tx(s, ADMIN);
    let seed = coin::mint_for_testing<SUI>(1_000_000, ts::ctx(s));
    market::create_market(
        b"Did event E happen by the deadline?",
        x"aabbccdd",
        12,
        x"", 1, 2, // judge_static_genesis_root, yes_token, no_token (placeholder; unused here)
        mk_keys(),
        vector[0, 0, 0, 0], // all ed25519 (scheme 0)
        vector[0, 0, 1, 1], // issuers 0,1 share group 0; issuers 2,3 share group 1
        k,
        burden,
        1000, // resolve_after_ms (must be > clock-now at creation)
        1000, // evidence_window_ms
        100, // fee_bps = 1%
        seed,
        clock,
        ts::ctx(s),
    )
}

fun hash32(tag: u8): vector<u8> {
    let mut h = vector[]; let mut i = 0; while (i < 32) { h.push_back(tag); i = i + 1; }; h
}

#[test]
fun amm_solvency_and_pricing() {
    let mut s = ts::begin(ADMIN);
    let mut clk = clock::create_for_testing(ts::ctx(&mut s));
    clk.set_for_testing(0);
    let mkt = fresh_market(&mut s, &clk, 2, 0);

    ts::next_tx(&mut s, ALICE);
    {
        let mut m = ts::take_shared_by_id<market::Market>(&s, object::id_from_address(mkt));
        let py0 = market::price_yes_bps(&m);
        let pay = coin::mint_for_testing<SUI>(200_000, ts::ctx(&mut s));
        market::buy_yes(&mut m, pay, &clk, ts::ctx(&mut s));
        let (ay, an, paid) = market::position_of(&m, ALICE);
        assert!(ay > 0 && an == 0 && paid == 200_000, 0);
        assert!(market::price_yes_bps(&m) > py0, 1); // buying YES raises YES price
        ts::return_shared(m);
    };
    ts::next_tx(&mut s, BOB);
    {
        let mut m = ts::take_shared_by_id<market::Market>(&s, object::id_from_address(mkt));
        let pay = coin::mint_for_testing<SUI>(150_000, ts::ctx(&mut s));
        market::buy_no(&mut m, pay, &clk, ts::ctx(&mut s));
        let (by, bn, _) = market::position_of(&m, BOB);
        assert!(bn > 0 && by == 0, 2);

        // SOLVENCY (complete-set invariant): total_YES == total_NO, and
        // collateral covers either side with fees as LP surplus.
        let coll = market::collateral_value(&m);
        let (ay, _, _) = market::position_of(&m, ALICE);
        let (_, bn2, _) = market::position_of(&m, BOB);
        let (ry, rn) = market::reserves(&m);
        let total_yes = ry + ay;
        let total_no = rn + bn2;
        assert!(total_yes == total_no, 3);
        assert!(coll >= total_yes, 4);
        assert!(coll - total_yes == 3500, 5); // exactly the 1% fees (2000 + 1500)
        ts::return_shared(m);
    };

    clk.destroy_for_testing();
    ts::end(s);
}

#[test]
fun occurrence_yes_needs_k_groups_then_redeem_and_lp_withdraws() {
    let mut s = ts::begin(ADMIN);
    let mut clk = clock::create_for_testing(ts::ctx(&mut s));
    clk.set_for_testing(0);
    let mkt = fresh_market(&mut s, &clk, 2, 0); // occurrence, k=2

    ts::next_tx(&mut s, ALICE);
    {
        let mut m = ts::take_shared_by_id<market::Market>(&s, object::id_from_address(mkt));
        let pay = coin::mint_for_testing<SUI>(300_000, ts::ctx(&mut s));
        market::buy_yes(&mut m, pay, &clk, ts::ctx(&mut s));
        ts::return_shared(m);
    };
    ts::next_tx(&mut s, BOB);
    {
        let mut m = ts::take_shared_by_id<market::Market>(&s, object::id_from_address(mkt));
        let pay = coin::mint_for_testing<SUI>(120_000, ts::ctx(&mut s));
        market::buy_no(&mut m, pay, &clk, ts::ctx(&mut s)); // Bob bets NO and will lose
        ts::return_shared(m);
    };

    // evidence window: two DISTINCT groups confirm YES
    clk.set_for_testing(1200);
    ts::next_tx(&mut s, BOB);
    {
        let mut m = ts::take_shared_by_id<market::Market>(&s, object::id_from_address(mkt));
        market::test_admit(&mut m, 0, 1, hash32(1), 1100, &clk, ts::ctx(&mut s));
        market::test_admit(&mut m, 2, 1, hash32(2), 1150, &clk, ts::ctx(&mut s));
        let (yg, ng) = market::group_counts_live(&m);
        assert!(yg == 2 && ng == 0, 0);
        ts::return_shared(m);
    };

    clk.set_for_testing(2000);
    ts::next_tx(&mut s, ADMIN);
    {
        let mut m = ts::take_shared_by_id<market::Market>(&s, object::id_from_address(mkt));
        market::resolve(&mut m, &clk);
        assert!(market::outcome(&m) == 1, 1); // YES
        ts::return_shared(m);
    };

    // Alice redeems winning YES 1:1 at a profit (paid 300k, holds >300k YES)
    ts::next_tx(&mut s, ALICE);
    {
        let mut m = ts::take_shared_by_id<market::Market>(&s, object::id_from_address(mkt));
        let (ay, _, _) = market::position_of(&m, ALICE);
        let payout = market::redeem(&mut m, ts::ctx(&mut s));
        assert!(coin::value(&payout) == ay && ay > 300_000, 2);
        coin::burn_for_testing(payout);
        ts::return_shared(m);
    };

    // The LP (ADMIN) recovers the residual: seed + pool inventory + fees +
    // Bob's losing stake, never touching collateral owed to winners.
    ts::next_tx(&mut s, ADMIN);
    {
        let mut m = ts::take_shared_by_id<market::Market>(&s, object::id_from_address(mkt));
        let cap = ts::take_from_sender<market::LPCap>(&s);
        let resid = market::withdraw_residual(&mut m, &cap, ts::ctx(&mut s));
        assert!(coin::value(&resid) > 0, 3);
        // after both winners and LP are paid, nothing is stranded
        assert!(market::collateral_value(&m) == 0, 4);
        coin::burn_for_testing(resid);
        ts::return_to_sender(&s, cap);
        ts::return_shared(m);
    };

    clk.destroy_for_testing();
    ts::end(s);
}

#[test]
fun directional_buyer_refunded_in_full_on_void() {
    // THE regression test: a one-sided CPMM holder (YES only) must recover
    // their FULL stake when the market voids — the bug that re-created the
    // UMA "no refunds" harm.
    let mut s = ts::begin(ADMIN);
    let mut clk = clock::create_for_testing(ts::ctx(&mut s));
    clk.set_for_testing(0);
    let mkt = fresh_market(&mut s, &clk, 1, 1); // STATE burden, k=1

    ts::next_tx(&mut s, ALICE);
    {
        let mut m = ts::take_shared_by_id<market::Market>(&s, object::id_from_address(mkt));
        let pay = coin::mint_for_testing<SUI>(500_000, ts::ctx(&mut s));
        market::buy_yes(&mut m, pay, &clk, ts::ctx(&mut s)); // YES only → no==0
        let (ay, an, paid) = market::position_of(&m, ALICE);
        assert!(ay > 0 && an == 0 && paid == 500_000, 0);
        ts::return_shared(m);
    };

    // conflicting evidence → UNRESOLVED
    clk.set_for_testing(1200);
    ts::next_tx(&mut s, BOB);
    {
        let mut m = ts::take_shared_by_id<market::Market>(&s, object::id_from_address(mkt));
        market::test_admit(&mut m, 0, 1, hash32(1), 1100, &clk, ts::ctx(&mut s)); // group 0 YES
        market::test_admit(&mut m, 2, 2, hash32(2), 1150, &clk, ts::ctx(&mut s)); // group 1 NO
        ts::return_shared(m);
    };
    clk.set_for_testing(2000);
    ts::next_tx(&mut s, ADMIN);
    {
        let mut m = ts::take_shared_by_id<market::Market>(&s, object::id_from_address(mkt));
        market::resolve(&mut m, &clk);
        assert!(market::outcome(&m) == 3, 1); // UNRESOLVED
        ts::return_shared(m);
    };
    // Alice gets her full 500k back even though she holds zero NO shares.
    ts::next_tx(&mut s, ALICE);
    {
        let mut m = ts::take_shared_by_id<market::Market>(&s, object::id_from_address(mkt));
        let refund = market::redeem(&mut m, ts::ctx(&mut s));
        assert!(coin::value(&refund) == 500_000, 2);
        coin::burn_for_testing(refund);
        ts::return_shared(m);
    };
    clk.destroy_for_testing();
    ts::end(s);
}

#[test]
fun occurrence_silence_is_no() {
    let mut s = ts::begin(ADMIN);
    let mut clk = clock::create_for_testing(ts::ctx(&mut s));
    clk.set_for_testing(0);
    let mkt = fresh_market(&mut s, &clk, 2, 0);

    clk.set_for_testing(1200);
    ts::next_tx(&mut s, BOB);
    {
        let mut m = ts::take_shared_by_id<market::Market>(&s, object::id_from_address(mkt));
        market::test_admit(&mut m, 0, 1, hash32(1), 1100, &clk, ts::ctx(&mut s)); // one group only
        ts::return_shared(m);
    };
    clk.set_for_testing(2000);
    ts::next_tx(&mut s, ADMIN);
    {
        let mut m = ts::take_shared_by_id<market::Market>(&s, object::id_from_address(mkt));
        market::resolve(&mut m, &clk);
        assert!(market::outcome(&m) == 2, 0); // NO (occurrence: silence ⇒ NO)
        ts::return_shared(m);
    };
    clk.destroy_for_testing();
    ts::end(s);
}

#[test]
fun syndication_does_not_fake_diversity() {
    let mut s = ts::begin(ADMIN);
    let mut clk = clock::create_for_testing(ts::ctx(&mut s));
    clk.set_for_testing(0);
    let mkt = fresh_market(&mut s, &clk, 2, 0);

    clk.set_for_testing(1200);
    ts::next_tx(&mut s, BOB);
    {
        let mut m = ts::take_shared_by_id<market::Market>(&s, object::id_from_address(mkt));
        market::test_admit(&mut m, 0, 1, hash32(1), 1100, &clk, ts::ctx(&mut s)); // group 0
        market::test_admit(&mut m, 1, 1, hash32(2), 1110, &clk, ts::ctx(&mut s)); // group 0 again
        let (yg, _) = market::group_counts_live(&m);
        assert!(yg == 1, 0); // syndication collapses to one group
        ts::return_shared(m);
    };
    clk.set_for_testing(2000);
    ts::next_tx(&mut s, ADMIN);
    {
        let mut m = ts::take_shared_by_id<market::Market>(&s, object::id_from_address(mkt));
        market::resolve(&mut m, &clk);
        assert!(market::outcome(&m) == 2, 1); // NO — never reached k independent groups
        ts::return_shared(m);
    };
    clk.destroy_for_testing();
    ts::end(s);
}

#[test]
fun dedup_marks_seen() {
    let mut s = ts::begin(ADMIN);
    let mut clk = clock::create_for_testing(ts::ctx(&mut s));
    clk.set_for_testing(0);
    let mkt = fresh_market(&mut s, &clk, 2, 0);
    clk.set_for_testing(1200);
    ts::next_tx(&mut s, BOB);
    {
        let mut m = ts::take_shared_by_id<market::Market>(&s, object::id_from_address(mkt));
        market::test_admit(&mut m, 0, 1, hash32(7), 1100, &clk, ts::ctx(&mut s));
        assert!(market::test_is_seen(&m, hash32(7)), 0);
        assert!(!market::test_is_seen(&m, hash32(8)), 1);
        ts::return_shared(m);
    };
    clk.destroy_for_testing();
    ts::end(s);
}

#[test]
#[expected_failure(abort_code = 17, location = veritas::market)]
fun create_rejects_unsatisfiable_k() {
    // k=2 but every issuer is in ONE group → the rule can never be met.
    let mut s = ts::begin(ADMIN);
    let mut clk = clock::create_for_testing(ts::ctx(&mut s));
    clk.set_for_testing(0);
    ts::next_tx(&mut s, ADMIN);
    let seed = coin::mint_for_testing<SUI>(1_000, ts::ctx(&mut s));
    market::create_market(
        b"q", x"aa", 12, x"", 1, 2, mk_keys(), vector[0, 0, 0, 0], vector[0, 0, 0, 0], 2, 0, 1000, 1000, 0, seed, &clk, ts::ctx(&mut s),
    );
    abort 99
}

#[test]
#[expected_failure(abort_code = 16, location = veritas::market)]
fun create_rejects_duplicate_keys() {
    let mut s = ts::begin(ADMIN);
    let mut clk = clock::create_for_testing(ts::ctx(&mut s));
    clk.set_for_testing(0);
    ts::next_tx(&mut s, ADMIN);
    let mut keys = mk_keys();
    *&mut keys[1] = keys[0]; // duplicate key in a different group
    let seed = coin::mint_for_testing<SUI>(1_000, ts::ctx(&mut s));
    market::create_market(
        b"q", x"aa", 12, x"", 1, 2, keys, vector[0, 0, 0, 0], vector[0, 1, 2, 3], 2, 0, 1000, 1000, 0, seed, &clk, ts::ctx(&mut s),
    );
    abort 99
}

#[test]
fun ed25519_admission_matches_offchain_signer() {
    // One generated Web Credential: issuer keypair signs the canonical
    // message for a FIXED market address; the on-chain verify must accept.
    // Vector from dispute/tests/gen_market_vector.py (ed25519 + blake2b).
    let pubkey = x"03a107bff3ce10be1d70dd18e74bc09967e4d6309ba50d5f1ddc8664125531b8";
    let signature = x"86ef24ad7738dc35f54c7437d0c60d4eb7b71378160ba8385380b0adab42a33722bcf19840cc249896aae332c144c1428764bfb3b4ff8ce6183313256f4f8f0f";
    let content_hash = x"1111111111111111111111111111111111111111111111111111111111111111";
    let market_addr = @0xCAFE;
    let msg = market::test_canonical_message(market_addr, 1, content_hash, 1500);
    assert!(sui::ed25519::ed25519_verify(&signature, &pubkey, &msg), 0);
    // a tampered claim must NOT verify
    let bad = market::test_canonical_message(market_addr, 2, content_hash, 1500);
    assert!(!sui::ed25519::ed25519_verify(&signature, &pubkey, &bad), 1);
}

#[test]
fun lp_cannot_drain_collateral_owed_to_unredeemed_winners() {
    // Solvency under INTERLEAVED settlement: the LP withdraws the residual
    // while a winning buyer has NOT yet redeemed. withdraw_residual must pay
    // only collateral − outstanding winner liability (never the un-redeemed
    // winner's funds), the late winner must still redeem in full, and total
    // payouts must conserve total stake-in. (Existing tests only cover the
    // all-winners-redeemed-first ordering.)
    let mut s = ts::begin(ADMIN);
    let mut clk = clock::create_for_testing(ts::ctx(&mut s));
    clk.set_for_testing(0);
    let mkt = fresh_market(&mut s, &clk, 2, 0); // occurrence, k=2

    // Alice + Bob bet YES (winners); Carol bets NO (loser).
    ts::next_tx(&mut s, ALICE);
    {
        let mut m = ts::take_shared_by_id<market::Market>(&s, object::id_from_address(mkt));
        market::buy_yes(&mut m, coin::mint_for_testing<SUI>(300_000, ts::ctx(&mut s)), &clk, ts::ctx(&mut s));
        ts::return_shared(m);
    };
    ts::next_tx(&mut s, BOB);
    {
        let mut m = ts::take_shared_by_id<market::Market>(&s, object::id_from_address(mkt));
        market::buy_yes(&mut m, coin::mint_for_testing<SUI>(250_000, ts::ctx(&mut s)), &clk, ts::ctx(&mut s));
        ts::return_shared(m);
    };
    ts::next_tx(&mut s, CAROL);
    {
        let mut m = ts::take_shared_by_id<market::Market>(&s, object::id_from_address(mkt));
        market::buy_no(&mut m, coin::mint_for_testing<SUI>(180_000, ts::ctx(&mut s)), &clk, ts::ctx(&mut s));
        ts::return_shared(m);
    };

    // two DISTINCT groups confirm YES → resolve YES
    clk.set_for_testing(1200);
    ts::next_tx(&mut s, ADMIN);
    {
        let mut m = ts::take_shared_by_id<market::Market>(&s, object::id_from_address(mkt));
        market::test_admit(&mut m, 0, 1, hash32(1), 1100, &clk, ts::ctx(&mut s));
        market::test_admit(&mut m, 2, 1, hash32(2), 1150, &clk, ts::ctx(&mut s));
        ts::return_shared(m);
    };
    clk.set_for_testing(2000);
    ts::next_tx(&mut s, ADMIN);
    {
        let mut m = ts::take_shared_by_id<market::Market>(&s, object::id_from_address(mkt));
        market::resolve(&mut m, &clk);
        assert!(market::outcome(&m) == 1, 0); // YES
        ts::return_shared(m);
    };

    let mut alice_payout = 0;
    let mut bob_yes = 0;
    let mut lp_residual = 0;
    let mut bob_payout = 0;

    // Alice redeems first; Bob has NOT redeemed yet.
    ts::next_tx(&mut s, ALICE);
    {
        let mut m = ts::take_shared_by_id<market::Market>(&s, object::id_from_address(mkt));
        let c = market::redeem(&mut m, ts::ctx(&mut s));
        alice_payout = coin::value(&c);
        coin::burn_for_testing(c);
        ts::return_shared(m);
    };

    // LP withdraws the residual WHILE Bob still holds un-redeemed YES.
    ts::next_tx(&mut s, ADMIN);
    {
        let mut m = ts::take_shared_by_id<market::Market>(&s, object::id_from_address(mkt));
        let (by, _, _) = market::position_of(&m, BOB);
        bob_yes = by;
        let cap = ts::take_from_sender<market::LPCap>(&s);
        let r = market::withdraw_residual(&mut m, &cap, ts::ctx(&mut s));
        lp_residual = coin::value(&r);
        // CRITICAL: collateral left must still cover Bob's un-redeemed winning
        // shares 1:1 — the LP cannot touch funds owed to a winner.
        assert!(market::collateral_value(&m) == bob_yes, 1);
        coin::burn_for_testing(r);
        ts::return_to_sender(&s, cap);
        ts::return_shared(m);
    };

    // Bob now redeems in FULL despite the LP having already withdrawn.
    ts::next_tx(&mut s, BOB);
    {
        let mut m = ts::take_shared_by_id<market::Market>(&s, object::id_from_address(mkt));
        let c = market::redeem(&mut m, ts::ctx(&mut s));
        bob_payout = coin::value(&c);
        assert!(bob_payout == bob_yes, 2); // full, untouched by the LP withdraw
        coin::burn_for_testing(c);
        assert!(market::collateral_value(&m) == 0, 3); // nothing stranded
        ts::return_shared(m);
    };

    // CONSERVATION: everything paid out equals everything staked in
    // (seed 1_000_000 + Alice 300k + Bob 250k + Carol 180k).
    assert!(alice_payout + bob_payout + lp_residual == 1_730_000, 4);

    clk.destroy_for_testing();
    ts::end(s);
}

// Deterministic LCG (glibc constants, modulus 2^31 keeps every product < 2^62
// so u64 never overflows). Test-only randomness for the property test below.
fun lcg(seed: &mut u64): u64 {
    *seed = (*seed * 1103515245 + 12345) % 2147483648;
    *seed
}

#[test]
fun cpmm_solvency_holds_under_random_trades() {
    // Property: after EVERY trade the complete-set invariant holds —
    // reserve_yes + user_yes == reserve_no + user_no — and collateral always
    // covers the outstanding shares (the surplus is accrued LP fees). A single
    // trader is used so per-user totals equal market totals without a new
    // accessor; the CPMM arithmetic (mul_div, reserve depletion, flooring) is
    // identical regardless of who trades.
    let mut s = ts::begin(ADMIN);
    let mut clk = clock::create_for_testing(ts::ctx(&mut s));
    clk.set_for_testing(0); // stay in TRADING (resolve_after_ms = 1000)
    let mkt = fresh_market(&mut s, &clk, 2, 0);

    let mut seed = 0xC0FFEE;
    let mut i = 0;
    while (i < 120) {
        ts::next_tx(&mut s, ALICE);
        let mut m = ts::take_shared_by_id<market::Market>(&s, object::id_from_address(mkt));
        let amt = 1 + lcg(&mut seed) % 50_000;
        if (lcg(&mut seed) % 2 == 0) {
            market::buy_yes(&mut m, coin::mint_for_testing<SUI>(amt, ts::ctx(&mut s)), &clk, ts::ctx(&mut s));
        } else {
            market::buy_no(&mut m, coin::mint_for_testing<SUI>(amt, ts::ctx(&mut s)), &clk, ts::ctx(&mut s));
        };
        let (ry, rn) = market::reserves(&m);
        let (ay, an, _) = market::position_of(&m, ALICE);
        assert!(ry + ay == rn + an, 0);                          // complete-set invariant
        assert!(market::collateral_value(&m) >= ry + ay, 1);     // solvent
        ts::return_shared(m);
        i = i + 1;
    };
    clk.destroy_for_testing();
    ts::end(s);
}

#[test]
#[expected_failure(abort_code = 19, location = veritas::market)]
fun evidence_cap_enforced() {
    // The MAX_EVIDENCE (256) cap bounds resolve()'s gas and stops a captured
    // issuer from flooding. The 257th admit must abort E_TOO_MANY_EVIDENCE (19),
    // which is checked BEFORE dedup so a reused hash still trips the cap.
    let mut s = ts::begin(ADMIN);
    let mut clk = clock::create_for_testing(ts::ctx(&mut s));
    clk.set_for_testing(0);
    let mkt = fresh_market(&mut s, &clk, 2, 0);
    clk.set_for_testing(1200);
    ts::next_tx(&mut s, BOB);
    let mut m = ts::take_shared_by_id<market::Market>(&s, object::id_from_address(mkt));
    let mut i = 0u64;
    while (i < 256) {
        market::test_admit(&mut m, 0, 1, hash32((i as u8)), 1100, &clk, ts::ctx(&mut s));
        i = i + 1;
    };
    market::test_admit(&mut m, 0, 1, hash32(0), 1100, &clk, ts::ctx(&mut s)); // 257th → abort 19
    abort 99
}

#[test]
fun resolve_is_permissionless() {
    // resolve() takes no capability: anyone can finalize once the window
    // closes (it is a pure function of admitted evidence). A stranger settles
    // it. This documents+pins the permissionless property so a reviewer doesn't
    // read it as a missing auth check.
    let mut s = ts::begin(ADMIN);
    let mut clk = clock::create_for_testing(ts::ctx(&mut s));
    clk.set_for_testing(0);
    let mkt = fresh_market(&mut s, &clk, 2, 0); // creator/LP = ADMIN
    clk.set_for_testing(2000); // evidence window closed
    ts::next_tx(&mut s, BOB); // a stranger, neither creator nor LP
    {
        let mut m = ts::take_shared_by_id<market::Market>(&s, object::id_from_address(mkt));
        market::resolve(&mut m, &clk);
        assert!(market::outcome(&m) == 2, 0); // NO (occurrence, no evidence ⇒ NO)
        ts::return_shared(m);
    };
    clk.destroy_for_testing();
    ts::end(s);
}

// -- decode_verdict (Fact.output `[n][token×n]` → claim) -------------------
fun u32le(x: u64): vector<u8> {
    let mut v = vector<u8>[];
    v.push_back((x & 0xff) as u8);
    v.push_back(((x >> 8) & 0xff) as u8);
    v.push_back(((x >> 16) & 0xff) as u8);
    v.push_back(((x >> 24) & 0xff) as u8);
    v
}

fun build_output(tokens: vector<u64>): vector<u8> {
    let n = tokens.length();
    let mut o = u32le(n);
    let mut i = 0;
    while (i < n) { o.append(u32le(tokens[i])); i = i + 1; };
    o
}

#[test]
fun decode_verdict_cases() {
    let yes: u32 = 9001;
    let no: u32 = 9002;
    // single YES / NO verdict token
    assert!(market::test_decode_verdict(build_output(vector[9001]), yes, no) == 1, 0);
    assert!(market::test_decode_verdict(build_output(vector[9002]), yes, no) == 2, 1);
    // YES after an unrelated token still wins
    assert!(market::test_decode_verdict(build_output(vector[5, 9001]), yes, no) == 1, 2);
    // NO appears before YES ⇒ NO wins (earliest standalone)
    assert!(market::test_decode_verdict(build_output(vector[9002, 9001]), yes, no) == 2, 3);
    // neither token present ⇒ ABSTAIN (3)
    assert!(market::test_decode_verdict(build_output(vector[5, 6, 7]), yes, no) == 3, 4);
    // empty token list ⇒ ABSTAIN
    assert!(market::test_decode_verdict(build_output(vector[]), yes, no) == 3, 5);
    // too short to hold even the length prefix ⇒ ABSTAIN
    assert!(market::test_decode_verdict(vector[0u8, 1u8], yes, no) == 3, 6);
}

// -- drop_misextracted: the engine<->market binding (SPEC §7.2) ------------
fun page(tag: u8): vector<u8> {
    let mut p = vector<u8>[];
    let mut i = 0;
    while (i < 1024) { p.push_back(tag); i = i + 1; };
    p
}

#[test]
fun drop_misextracted_flips_resolution() {
    // Synthetic depth-2 judge genesis: pages 0,1 static (weights); pages 2,3 are
    // THIS item's input region. (Same construction as opml::genesis_tests.)
    let p0 = page(11);
    let p1 = page(22);
    let g2 = page(33);
    let g3 = page(44);
    let l0 = opml::merkle::page_leaf(&p0);
    let l1 = opml::merkle::page_leaf(&p1);
    let z0 = opml::genesis::zero_page_leaf();
    let left = opml::merkle::node(&l0, &l1);
    let static_root = opml::merkle::node(&left, &opml::merkle::node(&z0, &z0));
    let l2 = opml::merkle::page_leaf(&g2);
    let l3 = opml::merkle::page_leaf(&g3);
    let genesis_f = opml::merkle::node(&left, &opml::merkle::node(&l2, &l3));
    let pages = vector[g2, g3];
    let indices = vector[2u64, 3u64];
    let siblings = vector[vector[copy z0, copy left], vector[copy l2, copy left]];

    let pr = x"deadbeef"; // judge program root
    let yes_tok: u32 = 100;
    let no_tok: u32 = 200;

    let mut s = ts::begin(ADMIN);
    let mut clk = clock::create_for_testing(ts::ctx(&mut s));
    clk.set_for_testing(0);

    // a market committing this judge identity; occurrence, k=1 (one YES group ⇒ YES)
    ts::next_tx(&mut s, ADMIN);
    let seed = coin::mint_for_testing<SUI>(1_000_000, ts::ctx(&mut s));
    let mkt = market::create_market(
        b"Did it happen?", pr, 2,
        static_root, yes_tok, no_tok,
        mk_keys(), vector[0, 0, 0, 0], vector[0, 0, 1, 1],
        1, 0, 1000, 1000, 0, seed, &clk, ts::ctx(&mut s),
    );

    // admit a YES item — the judge will be PROVEN to actually read NO.
    clk.set_for_testing(1200);
    ts::next_tx(&mut s, ADMIN);
    {
        let mut m = ts::take_shared_by_id<market::Market>(&s, object::id_from_address(mkt));
        market::test_admit(&mut m, 0, 1, hash32(1), 1100, &clk, ts::ctx(&mut s));
        let (yg, _) = market::group_counts_live(&m);
        assert!(yg == 1, 0); // counts as YES before the drop
        ts::return_shared(m);
    };

    // a FINALIZED counter-extraction Fact: same program, genesis BUILT from the
    // item's input, output = the NO token ⇒ true reading NO ≠ asserted YES.
    ts::next_tx(&mut s, ADMIN);
    opml::dispute::share_finalized_fact_for_testing(
        2, 0, pr, genesis_f, 0, build_output(vector[200]), ts::ctx(&mut s),
    );

    // drop the mis-extracted item
    ts::next_tx(&mut s, ADMIN);
    {
        let mut m = ts::take_shared_by_id<market::Market>(&s, object::id_from_address(mkt));
        let fact = ts::take_shared<opml::dispute::Fact>(&s);
        market::drop_misextracted(&mut m, 0, &fact, pages, indices, siblings);
        let (yg2, _) = market::group_counts_live(&m);
        assert!(yg2 == 0, 1); // dropped ⇒ no longer counts
        ts::return_shared(fact);
        ts::return_shared(m);
    };

    // resolve: occurrence, zero YES groups ⇒ NO — the wrong YES was slashed out.
    clk.set_for_testing(2000);
    ts::next_tx(&mut s, ADMIN);
    {
        let mut m = ts::take_shared_by_id<market::Market>(&s, object::id_from_address(mkt));
        market::resolve(&mut m, &clk);
        assert!(market::outcome(&m) == 2, 2); // NO (flipped from YES)
        ts::return_shared(m);
    };
    clk.destroy_for_testing();
    ts::end(s);
}
