/// reclaim.move — on-chain verification of a Reclaim zkTLS web proof.
///
/// This is how we get REAL data from sources that DON'T sign their content
/// (BBC, Reuters, any website): a Reclaim attestor witnesses the TLS
/// session to the site and signs the extracted claim. On-chain we verify
/// that attestor signature, exactly as Reclaim's own verifier does, and
/// require the signer to be a PINNED attestor — so the market admits
/// "site X served value V at time T" with no trust beyond the attestor set.
///
/// The verification is the standard Reclaim flow, all Sui-native:
///   identifier = keccak256(provider \n parameters \n context)
///   signed     = identifier \n owner \n timestamp \n epoch          (EIP-191)
///   recover the secp256k1 signer of `signed`; require it == a pinned attestor
///
/// (Proof GENERATION happens in the Reclaim app/SDK — a browser/mobile
/// client doing the witnessed fetch; that is a client step, not a chain
/// step. This module is the chain half, and it is real and native.)
///
/// Trust note (WEBPROOFS.md): the attestor is a witness, so a Reclaim
/// proof carries the attestor's trust assumption. In a market it is a
/// capped, diversified tier (EVIDENCE.md §3): the attestor set is ONE
/// trust group, and zkTLS items corroborate rather than carry alone.
module veritas::reclaim;

use sui::ecdsa_k1;
use sui::hash;

const E_BAD_ATTESTOR: u64 = 1; // recovered signer is not the pinned attestor
const E_BAD_SIG_LEN: u64 = 2;

/// A verified zkTLS claim: the attestor witnessed `provider`/`parameters`
/// returning `context` (the extracted values) at `timestamp_s`.
public struct WebProof has drop {
    provider: vector<u8>,
    parameters: vector<u8>,
    context: vector<u8>,
    attestor: vector<u8>, // 20-byte recovered address (== the pinned one)
    timestamp_s: u64,
    epoch: u64,
}

/// Verify a Reclaim proof. Aborts unless a PINNED attestor signed the
/// canonical claim. Returns the verified web proof.
public fun verify(
    provider: vector<u8>,
    parameters: vector<u8>,
    context: vector<u8>,
    owner: vector<u8>, // ascii lowercase 0x-address of the proof owner
    timestamp_s: u64,
    epoch: u64,
    signature: vector<u8>, // 65-byte r||s||v (eth v in {27,28} or {0,1})
    expected_attestor: vector<u8>, // 20-byte pinned attestor address
): WebProof {
    assert!(signature.length() == 65, E_BAD_SIG_LEN);

    // identifier = "0x" + hex(keccak256(provider \n parameters \n context))
    let mut claim = provider;
    claim.push_back(0x0a);
    claim.append(parameters);
    claim.push_back(0x0a);
    claim.append(context);
    let id_hex = hex0x(&hash::keccak256(&claim));

    // signed = identifier \n owner \n ascii(timestamp) \n ascii(epoch)
    let mut signed = id_hex;
    signed.push_back(0x0a);
    signed.append(owner);
    signed.push_back(0x0a);
    signed.append(ascii_u64(timestamp_s));
    signed.push_back(0x0a);
    signed.append(ascii_u64(epoch));

    // EIP-191 personal_sign: 0x19 || "Ethereum Signed Message:\n" || len || signed
    let mut msg = eip191_header();
    msg.append(ascii_u64(signed.length()));
    msg.append(signed);

    // recover the secp256k1 signer (Sui hashes msg with keccak256 = flag 0)
    let mut sig = signature;
    let v = sig[64];
    if (v >= 27) { *(&mut sig[64]) = v - 27; }; // eth {27,28} -> sui {0,1}
    let comp = ecdsa_k1::secp256k1_ecrecover(&sig, &msg, 0);
    let uncomp = ecdsa_k1::decompress_pubkey(&comp); // 65 bytes: 0x04 || X || Y

    // eth address = last 20 bytes of keccak256(X || Y)
    let mut xy = vector<u8>[];
    let mut i = 1;
    while (i < 65) { xy.push_back(uncomp[i]); i = i + 1; };
    let ah = hash::keccak256(&xy);
    let mut addr = vector<u8>[];
    let mut j = 12;
    while (j < 32) { addr.push_back(ah[j]); j = j + 1; };

    assert!(addr == expected_attestor, E_BAD_ATTESTOR);
    WebProof { provider, parameters, context, attestor: addr, timestamp_s, epoch }
}

fun eip191_header(): vector<u8> {
    let mut v = vector<u8>[0x19];
    v.append(b"Ethereum Signed Message:");
    v.push_back(0x0a); // '\n'
    v
}

/// lowercase "0x"-prefixed hex of bytes.
fun hex0x(b: &vector<u8>): vector<u8> {
    let digits = b"0123456789abcdef";
    let mut out = vector<u8>[0x30, 0x78]; // "0x"
    let mut i = 0;
    let n = b.length();
    while (i < n) {
        let byte = b[i];
        out.push_back(digits[(byte >> 4) as u64]);
        out.push_back(digits[(byte & 0x0f) as u64]);
        i = i + 1;
    };
    out
}

/// decimal ascii of a u64.
fun ascii_u64(mut x: u64): vector<u8> {
    if (x == 0) return vector<u8>[0x30];
    let mut rev = vector<u8>[];
    while (x > 0) { rev.push_back((48 + (x % 10)) as u8); x = x / 10; };
    let mut out = vector<u8>[];
    let mut i = rev.length();
    while (i > 0) { i = i - 1; out.push_back(rev[i]); };
    out
}

// -- accessors --
public fun attestor(p: &WebProof): vector<u8> { p.attestor }
public fun provider(p: &WebProof): vector<u8> { p.provider }
public fun parameters(p: &WebProof): vector<u8> { p.parameters }
public fun context(p: &WebProof): vector<u8> { p.context }
public fun timestamp_s(p: &WebProof): u64 { p.timestamp_s }

#[test_only]
public fun build_signed_preview(provider: vector<u8>, parameters: vector<u8>, context: vector<u8>): vector<u8> {
    let mut claim = provider;
    claim.push_back(0x0a); claim.append(parameters); claim.push_back(0x0a); claim.append(context);
    hex0x(&hash::keccak256(&claim))
}
