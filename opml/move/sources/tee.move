/// tee.move — the optional SECOND security layer for the judge (Sui Nautilus).
///
/// This is the on-chain verification side of Sui **Nautilus**, Mysten's
/// framework for verifiable offchain computation: run the committed judge
/// inside an AWS Nitro enclave, and verify its attestation on-chain by its
/// PCR measurements. (Nautilus IS Nitro-based — it is the Sui-blessed way
/// to do TEE compute, and this module is exactly its attestation check.)
///
/// Defense in depth. The fraud proof (opml::dispute) is the HARD
/// guarantee: it proves the committed judge ran CORRECTLY, byte-for-byte,
/// and slashes a liar. This module adds a complementary, independent
/// guarantee: that the judge ran inside a genuine AWS Nitro ENCLAVE whose
/// image measurement (PCR0) matches a build committed up front — verified
/// on-chain via Sui's NATIVE `nitro_attestation` (COSE + the AWS Nitro
/// cert chain to its root, all checked in the protocol).
///
/// Why both:
///   - the TEE attests the judge ran in tamper-resistant hardware of a
///     known build (a hardware-vendor trust root), giving fast,
///     OPTIMISTIC soft-finality and a second wall an attacker must also
///     breach;
///   - the fraud proof needs NO hardware trust and is the final word if a
///     dispute is raised.
/// An attacker now has to both forge an AWS Nitro attestation AND win a
/// one-micro-op bisection. The trust roots are orthogonal and additive
/// (WEBPROOFS.md §3.1): TEE = the hardware vendor; fraud proof = the chain.
module opml::tee;

use sui::nitro_attestation::NitroAttestationDocument;

const E_BAD_ENCLAVE: u64 = 1; // PCR0 (enclave image) != the committed build
const E_NO_PCR0: u64 = 2;

/// A verified statement that the committed judge build ran in a genuine
/// Nitro enclave. `user_data` binds the run (e.g. program_root || output).
public struct JudgeAttestation has drop {
    pcr0: vector<u8>, // enclave image measurement = the judge build
    timestamp_ms: u64,
    user_data: Option<vector<u8>>,
}

/// Bind a loaded Nitro attestation to a market by requiring its enclave
/// image (PCR0) to equal `expected_pcr0` — the build a market committed.
///
/// `doc` must come from `sui::nitro_attestation::load_nitro_attestation`
/// (an `entry` native that verifies the COSE signature + the AWS Nitro
/// cert chain to its root, and checks validity against the clock). That
/// native call is a separate PTB command; this function is the binding
/// the application controls. Aborts unless PCR0 matches the committed
/// build, so you can only attest the exact judge image you committed.
public fun verify_judge_enclave(
    doc: &NitroAttestationDocument,
    expected_pcr0: vector<u8>,
): JudgeAttestation {
    let pcrs = doc.pcrs();
    let mut pcr0 = vector<u8>[];
    let mut found = false;
    let mut i = 0;
    let n = pcrs.length();
    while (i < n) {
        if (pcrs[i].index() == 0) { pcr0 = *pcrs[i].value(); found = true; };
        i = i + 1;
    };
    assert!(found, E_NO_PCR0);
    assert!(pcr0 == expected_pcr0, E_BAD_ENCLAVE);
    JudgeAttestation { pcr0, timestamp_ms: *doc.timestamp(), user_data: *doc.user_data() }
}

public fun pcr0(a: &JudgeAttestation): vector<u8> { a.pcr0 }
public fun timestamp_ms(a: &JudgeAttestation): u64 { a.timestamp_ms }
public fun user_data(a: &JudgeAttestation): &Option<vector<u8>> { &a.user_data }

// INTEGRATION & TESTING NOTE.
//
// The PTB flow is two commands:
//   1) sui::nitro_attestation::load_nitro_attestation(att_bytes, clock)
//      → NitroAttestationDocument   (native: COSE signature + the full AWS
//        Nitro cert chain to its root, validity vs the clock — all checked
//        in-protocol). This native is an `entry`, so it is invoked as its
//        OWN PTB command, not from inside another module.
//   2) opml::tee::verify_judge_enclave(&doc, expected_pcr0)
//      → JudgeAttestation, the binding to the committed judge build.
//
// The native verification is exercised by Sui's own framework tests; this
// module's logic is the PCR0 binding above. A passing unit test here needs
// a real attestation document, and on-chain parsing of a given sample is
// protocol-version-sensitive (the framework sample requires an upgraded
// parser gated by a protocol flag), so the end-to-end check belongs on a
// network with the matching protocol + a live enclave rather than in
// `sui move test`. Producing a fresh attestation requires actual Nitro
// hardware running the committed judge image.
