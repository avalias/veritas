/// credential.move — the universal Web Credential verifier.
///
/// One interface over the signature schemes the real world already emits,
/// so a market can admit evidence from any of them (VISION.md §3):
///
///   ED25519  signed feeds / oracles (Pyth), generic signers — Sui-native.
///   ES256    P-256 / SHA-256 = COSE ES256, the scheme C2PA "Content
///            Credentials" use to sign news photos/video (BBC Verify,
///            IPTC, Adobe). Verified on Sui NATIVELY via ecdsa_r1.
///   RSA2048  PKCS#1 v1.5 / SHA-256 = DKIM, how news orgs sign their email
///            (and RS256 OAuth JWTs). On-chain, the cheap production path
///            is a zkEmail-style Groth16 proof (sui::groth16 is native);
///            naive 2048-bit modexp in Move is correct but gas-prohibitive,
///            so RSA is verified off-chain / via Groth16, not here.
///
/// The point: evidence provenance reduces to ONE call —
/// `credential::verify(scheme, key, message, signature)` — and every
/// scheme chains to a publisher's own key, so there is no trusted oracle,
/// only the publisher's signature you already rely on.
module dispute::credential;

use sui::ecdsa_r1;
use sui::ed25519;

const SCHEME_ED25519: u8 = 0;
const SCHEME_ES256: u8 = 1; // P-256/SHA-256, C2PA / COSE ES256
const SCHEME_RSA2048: u8 = 2; // PKCS#1v1.5/SHA-256, DKIM (off-chain / Groth16)

/// ecdsa_r1 hash selector: 1 = SHA-256 (what ES256/C2PA uses).
const ECDSA_R1_SHA256: u8 = 1;

const E_BAD_SCHEME: u64 = 1;
const E_BAD_KEY_LEN: u64 = 2;

/// True iff `signature` is a valid signature by `public_key` over
/// `message` under `scheme`. Aborts on an unsupported/oversized input so a
/// market can never silently admit on a scheme it doesn't actually check.
public fun verify(scheme: u8, public_key: &vector<u8>, message: &vector<u8>, signature: &vector<u8>): bool {
    if (scheme == SCHEME_ED25519) {
        assert!(public_key.length() == 32, E_BAD_KEY_LEN);
        ed25519::ed25519_verify(signature, public_key, message)
    } else if (scheme == SCHEME_ES256) {
        // compressed P-256 point (33 bytes); secp256r1_verify hashes msg with SHA-256
        assert!(public_key.length() == 33, E_BAD_KEY_LEN);
        ecdsa_r1::secp256r1_verify(signature, public_key, message, ECDSA_R1_SHA256)
    } else {
        // SCHEME_RSA2048 and anything else: not verified on-chain here.
        // DKIM / RS256-JWT are admitted via a Groth16 proof (zkEmail) whose
        // verifying key pins the issuer; see VISION.md §3 and WEBPROOFS.md.
        abort E_BAD_SCHEME
    }
}

/// Is this a scheme `verify` can check natively on-chain right now?
public fun is_native(scheme: u8): bool {
    scheme == SCHEME_ED25519 || scheme == SCHEME_ES256
}

public fun scheme_ed25519(): u8 { SCHEME_ED25519 }
public fun scheme_es256(): u8 { SCHEME_ES256 }
public fun scheme_rsa2048(): u8 { SCHEME_RSA2048 }
