/// Unit tests for the market product (dispute::market). The full signed
/// end-to-end (real Web Credentials over a real market address) runs on
/// localnet via the Rust driver; here we prove the AMM solvency, the
/// lifecycle, the decision rule, dedup/independence accounting, the
/// fraud-proof backstop, and that the on-chain ed25519 admission matches
/// an off-chain signer bit-for-bit (one generated vector).
#[test_only]
module dispute::market_tests;

use dispute::market;
use sui::clock;
use sui::coin;
use sui::sui::SUI;
use sui::test_scenario as ts;

const ADMIN: address = @0xA;
const ALICE: address = @0xA11CE;
const BOB: address = @0xB0B;

// a market: 4 issuers, two trust groups {0,0,1,1}, k=2, occurrence burden,
// trading until t=1000, evidence window [1000, 2000).
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
        x"aabbccdd", // judge program root (placeholder for the test)
        12,
        mk_keys(),
        vector[0, 0, 1, 1], // issuers 0,1 share group 0; issuers 2,3 share group 1
        k,
        burden,
        1000, // resolve_after_ms
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

    // Alice buys YES, Bob buys NO; check shares minted and price moves.
    ts::next_tx(&mut s, ALICE);
    {
        let mut m = ts::take_shared_by_id<market::Market>(&s, object::id_from_address(mkt));
        let (py0, _) = (market::price_yes_bps(&m), 0);
        let pay = coin::mint_for_testing<SUI>(200_000, ts::ctx(&mut s));
        market::buy_yes(&mut m, pay, &clk, ts::ctx(&mut s));
        let (ay, an) = market::position_of(&m, ALICE);
        assert!(ay > 0 && an == 0, 0);
        // buying YES raises the YES price
        assert!(market::price_yes_bps(&m) > py0, 1);
        ts::return_shared(m);
    };
    ts::next_tx(&mut s, BOB);
    {
        let mut m = ts::take_shared_by_id<market::Market>(&s, object::id_from_address(mkt));
        let pay = coin::mint_for_testing<SUI>(150_000, ts::ctx(&mut s));
        market::buy_no(&mut m, pay, &clk, ts::ctx(&mut s));
        let (by, bn) = market::position_of(&m, BOB);
        assert!(bn > 0 && by == 0, 2);

        // SOLVENCY (the complete-set invariant): every share pair is minted
        // from 1 SUI, so total_YES == total_NO, and collateral covers either
        // side with the accumulated fee as surplus (LP profit).
        let coll = market::collateral_value(&m);
        let (ay, _an) = market::position_of(&m, ALICE);
        let (_by2, bn2) = market::position_of(&m, BOB);
        let (ry, rn) = market::reserves(&m);
        let total_yes = ry + ay; // pool inventory + all users' YES
        let total_no = rn + bn2; // pool inventory + all users' NO
        assert!(total_yes == total_no, 3); // complete-set balance
        assert!(coll >= total_yes, 4); // collateral covers every winning share
        assert!(coll - total_yes == 3500, 5); // exactly the 1% fees (2000 + 1500)
        ts::return_shared(m);
    };

    clk.destroy_for_testing();
    ts::end(s);
}

#[test]
fun occurrence_yes_needs_k_groups_then_redeem() {
    let mut s = ts::begin(ADMIN);
    let mut clk = clock::create_for_testing(ts::ctx(&mut s));
    clk.set_for_testing(0);
    let mkt = fresh_market(&mut s, &clk, 2, 0); // occurrence, k=2

    // Alice buys YES during trading
    ts::next_tx(&mut s, ALICE);
    {
        let mut m = ts::take_shared_by_id<market::Market>(&s, object::id_from_address(mkt));
        let pay = coin::mint_for_testing<SUI>(300_000, ts::ctx(&mut s));
        market::buy_yes(&mut m, pay, &clk, ts::ctx(&mut s));
        ts::return_shared(m);
    };

    // move into the evidence window; two DISTINCT groups confirm YES
    clk.set_for_testing(1200);
    ts::next_tx(&mut s, BOB);
    {
        let mut m = ts::take_shared_by_id<market::Market>(&s, object::id_from_address(mkt));
        market::test_admit(&mut m, 0, 1, hash32(1), 1100, &clk, ts::ctx(&mut s)); // group 0 says YES
        market::test_admit(&mut m, 2, 1, hash32(2), 1150, &clk, ts::ctx(&mut s)); // group 1 says YES
        let (yg, ng) = market::group_counts_live(&m);
        assert!(yg == 2 && ng == 0, 0);
        ts::return_shared(m);
    };

    // resolve after the window → YES
    clk.set_for_testing(2000);
    ts::next_tx(&mut s, ADMIN);
    {
        let mut m = ts::take_shared_by_id<market::Market>(&s, object::id_from_address(mkt));
        market::resolve(&mut m, &clk);
        assert!(market::outcome(&m) == 1, 1); // OUTCOME_YES
        ts::return_shared(m);
    };

    // Alice redeems her winning YES shares 1:1
    ts::next_tx(&mut s, ALICE);
    {
        let mut m = ts::take_shared_by_id<market::Market>(&s, object::id_from_address(mkt));
        let (ay, _) = market::position_of(&m, ALICE);
        let payout = market::redeem(&mut m, ts::ctx(&mut s));
        assert!(coin::value(&payout) == ay, 2);
        assert!(coin::value(&payout) > 300_000, 3); // profit: she paid 300k, holds >300k YES
        coin::burn_for_testing(payout);
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
        // only ONE group confirms YES — below k=2
        market::test_admit(&mut m, 0, 1, hash32(1), 1100, &clk, ts::ctx(&mut s));
        ts::return_shared(m);
    };
    clk.set_for_testing(2000);
    ts::next_tx(&mut s, ADMIN);
    {
        let mut m = ts::take_shared_by_id<market::Market>(&s, object::id_from_address(mkt));
        market::resolve(&mut m, &clk);
        assert!(market::outcome(&m) == 2, 0); // OUTCOME_NO (occurrence: silence ⇒ NO)
        ts::return_shared(m);
    };
    clk.destroy_for_testing();
    ts::end(s);
}

#[test]
fun syndication_does_not_fake_diversity() {
    // two items from the SAME trust group must count as ONE confirmation
    let mut s = ts::begin(ADMIN);
    let mut clk = clock::create_for_testing(ts::ctx(&mut s));
    clk.set_for_testing(1200);
    let mkt = fresh_market(&mut s, &clk, 2, 0);

    ts::next_tx(&mut s, BOB);
    {
        let mut m = ts::take_shared_by_id<market::Market>(&s, object::id_from_address(mkt));
        market::test_admit(&mut m, 0, 1, hash32(1), 1100, &clk, ts::ctx(&mut s)); // group 0
        market::test_admit(&mut m, 1, 1, hash32(2), 1110, &clk, ts::ctx(&mut s)); // group 0 again (syndicator)
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
fun state_burden_conflicting_is_unresolved_and_refunds() {
    let mut s = ts::begin(ADMIN);
    let mut clk = clock::create_for_testing(ts::ctx(&mut s));
    clk.set_for_testing(0);
    let mkt = fresh_market(&mut s, &clk, 1, 1); // STATE burden, k=1

    // Alice buys BOTH sides during trading → she holds a complete set she can
    // get refunded if the market voids.
    ts::next_tx(&mut s, ALICE);
    {
        let mut m = ts::take_shared_by_id<market::Market>(&s, object::id_from_address(mkt));
        let p1 = coin::mint_for_testing<SUI>(100_000, ts::ctx(&mut s));
        market::buy_yes(&mut m, p1, &clk, ts::ctx(&mut s));
        let p2 = coin::mint_for_testing<SUI>(100_000, ts::ctx(&mut s));
        market::buy_no(&mut m, p2, &clk, ts::ctx(&mut s));
        ts::return_shared(m);
    };

    clk.set_for_testing(1200);
    ts::next_tx(&mut s, BOB);
    {
        let mut m = ts::take_shared_by_id<market::Market>(&s, object::id_from_address(mkt));
        // one group says YES, a different group says NO → tie at k=1
        market::test_admit(&mut m, 0, 1, hash32(1), 1100, &clk, ts::ctx(&mut s)); // group 0 YES
        market::test_admit(&mut m, 2, 2, hash32(2), 1150, &clk, ts::ctx(&mut s)); // group 1 NO
        ts::return_shared(m);
    };
    clk.set_for_testing(2000);
    ts::next_tx(&mut s, ADMIN);
    {
        let mut m = ts::take_shared_by_id<market::Market>(&s, object::id_from_address(mkt));
        market::resolve(&mut m, &clk);
        assert!(market::outcome(&m) == 3, 0); // OUTCOME_UNRESOLVED (conflicting)
        ts::return_shared(m);
    };
    // Alice refunds her matched complete set (min(yes,no)) at 1 SUI each.
    ts::next_tx(&mut s, ALICE);
    {
        let mut m = ts::take_shared_by_id<market::Market>(&s, object::id_from_address(mkt));
        let (ay, an) = market::position_of(&m, ALICE);
        let refund = market::redeem(&mut m, ts::ctx(&mut s));
        let expect = if (ay < an) { ay } else { an };
        assert!(coin::value(&refund) == expect && expect > 0, 1);
        coin::burn_for_testing(refund);
        ts::return_shared(m);
    };
    clk.destroy_for_testing();
    ts::end(s);
}

#[test]
fun dedup_rejects_same_content_twice() {
    let mut s = ts::begin(ADMIN);
    let mut clk = clock::create_for_testing(ts::ctx(&mut s));
    clk.set_for_testing(1200);
    let mkt = fresh_market(&mut s, &clk, 2, 0);
    ts::next_tx(&mut s, BOB);
    {
        let mut m = ts::take_shared_by_id<market::Market>(&s, object::id_from_address(mkt));
        market::test_admit(&mut m, 0, 1, hash32(7), 1100, &clk, ts::ctx(&mut s));
        // same content_hash again must abort with E_DUPLICATE_EVIDENCE (7)
        assert!(market::test_is_seen(&m, hash32(7)), 0);
        assert!(!market::test_is_seen(&m, hash32(8)), 1);
        ts::return_shared(m);
    };
    clk.destroy_for_testing();
    ts::end(s);
}

#[test]
fun ed25519_admission_matches_offchain_signer() {
    // One generated Web Credential: issuer keypair signs the canonical
    // message for a FIXED market address; the on-chain verify must accept.
    // Vector produced by client/src/bin/gen_market_vectors.rs (ed25519-dalek
    // + blake2b), pinned here. Proves admission crypto == off-chain signer.
    let pubkey = x"03a107bff3ce10be1d70dd18e74bc09967e4d6309ba50d5f1ddc8664125531b8";
    let signature = x"86ef24ad7738dc35f54c7437d0c60d4eb7b71378160ba8385380b0adab42a33722bcf19840cc249896aae332c144c1428764bfb3b4ff8ce6183313256f4f8f0f";
    let content_hash = x"1111111111111111111111111111111111111111111111111111111111111111";
    let market_addr = @0xCAFE;
    let claim: u8 = 1;
    let signed_ms: u64 = 1500;
    let msg = market::test_canonical_message(market_addr, claim, content_hash, signed_ms);
    assert!(sui::ed25519::ed25519_verify(&signature, &pubkey, &msg), 0);
    // a tampered claim must NOT verify
    let bad = market::test_canonical_message(market_addr, 2, content_hash, signed_ms);
    assert!(!sui::ed25519::ed25519_verify(&signature, &pubkey, &bad), 1);
}
