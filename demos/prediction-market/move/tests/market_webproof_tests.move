/// The market admits a REAL zkTLS web proof (veritas::reclaim) as evidence
/// — the path for unsigned web sources (BBC, Reuters, any URL). Uses the
/// same real attestor signature as reclaim_tests.
#[test_only]
module veritas::market_webproof_tests;

use veritas::market;
use sui::clock;
use sui::coin;
use sui::sui::SUI;
use sui::test_scenario as ts;

const ADMIN: address = @0xA;

const PROVIDER: vector<u8> = x"68747470";
const PARAMETERS: vector<u8> = x"7b2275726c223a2268747470733a2f2f7777772e6262632e636f6d2f6e657773222c226d6574686f64223a22474554222c22726573706f6e73654d617463686573223a5b7b2274797065223a22636f6e7461696e73222c2276616c7565223a2253746172736869702072656163686573206f72626974227d5d7d";
const CONTEXT: vector<u8> = x"7b22657874726163746564506172616d6574657273223a7b22686561646c696e65223a2253746172736869702072656163686573206f72626974227d2c2270726f766964657248617368223a223078626263227d";
const OWNER: vector<u8> = x"307831376335313835313637343031656430306366356635623266633937643962626664623764303235";
const SIG: vector<u8> = x"1885e46afbaca803bc46409eb951f9ecce0651ad8fcadc66a08d43af657455a969ff320c6e07af007bc8b2788d407df009381ae4dcf6513ff5f9d067888783551b";
const ATTESTOR: vector<u8> = x"17c5185167401ed00cf5f5b2fc97d9bbfdb7d025"; // 20-byte pinned attestor

#[test]
fun admits_real_zktls_web_proof() {
    let mut s = ts::begin(ADMIN);
    let mut clk = clock::create_for_testing(ts::ctx(&mut s));
    clk.set_for_testing(0);
    // a market that pins the Reclaim attestor (scheme 3) as its one source,
    // with an evidence window around the proof's witnessed timestamp.
    ts::next_tx(&mut s, ADMIN);
    let seed = coin::mint_for_testing<SUI>(1_000_000, ts::ctx(&mut s));
    let mkt = market::create_market(
        b"Did the BBC report the event before the deadline?",
        x"aabbccdd", 12,
        x"", 1, 2,             // judge_static_genesis_root, yes_token, no_token (placeholder)
        vector[ATTESTOR],      // issuer_keys: the 20-byte attestor address
        vector[3],             // issuer_schemes: SCHEME_RECLAIM
        vector[0],             // issuer_groups
        1, 0,                  // k=1, occurrence
        1734199999000, 4000, 0,    // resolve_after_ms, evidence_window_ms, fee
        seed, &clk, ts::ctx(&mut s),
    );

    // into the evidence window (the proof was witnessed at timestamp_s)
    clk.set_for_testing(1734200000 * 1000);
    ts::next_tx(&mut s, ADMIN);
    {
        let mut m = ts::take_shared_by_id<market::Market>(&s, object::id_from_address(mkt));
        market::submit_web_proof(&mut m, 0, 1 /*YES*/, PROVIDER, PARAMETERS, CONTEXT, OWNER, 1734200000, 1, SIG, &clk, ts::ctx(&mut s));
        assert!(market::evidence_count(&m) == 1, 0);
        let (yg, _) = market::group_counts_live(&m);
        assert!(yg == 1, 1); // the attestor group confirmed YES
        ts::return_shared(m);
    };

    // resolve YES (k=1 occurrence, one attestor group confirmed)
    clk.set_for_testing(1734200000 * 1000 + 4000);
    ts::next_tx(&mut s, ADMIN);
    {
        let mut m = ts::take_shared_by_id<market::Market>(&s, object::id_from_address(mkt));
        market::resolve(&mut m, &clk);
        assert!(market::outcome(&m) == 1, 2); // YES
        ts::return_shared(m);
    };
    clk.destroy_for_testing();
    ts::end(s);
}

/// Helper: a market pinning the Reclaim attestor with a window over the proof.
fun reclaim_market(s: &mut ts::Scenario, clk: &clock::Clock, q: vector<u8>): address {
    ts::next_tx(s, ADMIN);
    market::create_market(
        q, x"aabbccdd", 12,
        x"", 1, 2,                              // judge_static_genesis_root, yes_token, no_token (placeholder)
        vector[ATTESTOR], vector[3], vector[0], // attestor, SCHEME_RECLAIM, group 0
        1, 0,                                   // k=1, occurrence
        1734199999000, 4000, 0,                 // window covers timestamp_s
        coin::mint_for_testing<SUI>(1_000_000, ts::ctx(s)), clk, ts::ctx(s),
    )
}

/// By design, the SAME attested zkTLS fact is admissible into a SECOND market
/// that asks about it — a Reclaim proof witnesses a market-agnostic fact, so
/// reuse across markets is intended (contrast a signed Web Credential, whose
/// signature binds the market address and cannot be replayed elsewhere).
#[test]
fun web_proof_reusable_across_markets() {
    let mut s = ts::begin(ADMIN);
    let mut clk = clock::create_for_testing(ts::ctx(&mut s));
    clk.set_for_testing(0);
    let mkt_a = reclaim_market(&mut s, &clk, b"Market A: did the BBC report it?");
    let mkt_b = reclaim_market(&mut s, &clk, b"Market B: a different question, same event");

    clk.set_for_testing(1734200000 * 1000);
    ts::next_tx(&mut s, ADMIN);
    {
        let mut m = ts::take_shared_by_id<market::Market>(&s, object::id_from_address(mkt_a));
        market::submit_web_proof(&mut m, 0, 1, PROVIDER, PARAMETERS, CONTEXT, OWNER, 1734200000, 1, SIG, &clk, ts::ctx(&mut s));
        assert!(market::evidence_count(&m) == 1, 0);
        ts::return_shared(m);
    };
    ts::next_tx(&mut s, ADMIN);
    {
        // the SAME proof is also admissible into market B
        let mut m = ts::take_shared_by_id<market::Market>(&s, object::id_from_address(mkt_b));
        market::submit_web_proof(&mut m, 0, 1, PROVIDER, PARAMETERS, CONTEXT, OWNER, 1734200000, 1, SIG, &clk, ts::ctx(&mut s));
        assert!(market::evidence_count(&m) == 1, 1);
        ts::return_shared(m);
    };
    clk.destroy_for_testing();
    ts::end(s);
}

/// Within ONE market the same proof counts at most once: the second submission
/// of the identical proof aborts E_DUPLICATE_EVIDENCE (7).
#[test]
#[expected_failure(abort_code = 7, location = veritas::market)]
fun web_proof_deduped_within_market() {
    let mut s = ts::begin(ADMIN);
    let mut clk = clock::create_for_testing(ts::ctx(&mut s));
    clk.set_for_testing(0);
    let mkt = reclaim_market(&mut s, &clk, b"Did the BBC report the event?");
    clk.set_for_testing(1734200000 * 1000);
    ts::next_tx(&mut s, ADMIN);
    let mut m = ts::take_shared_by_id<market::Market>(&s, object::id_from_address(mkt));
    market::submit_web_proof(&mut m, 0, 1, PROVIDER, PARAMETERS, CONTEXT, OWNER, 1734200000, 1, SIG, &clk, ts::ctx(&mut s));
    // identical proof again → E_DUPLICATE_EVIDENCE
    market::submit_web_proof(&mut m, 0, 1, PROVIDER, PARAMETERS, CONTEXT, OWNER, 1734200000, 1, SIG, &clk, ts::ctx(&mut s));
    abort 99
}
