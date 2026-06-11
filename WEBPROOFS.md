# WEBPROOFS.md — the trust floor for getting unsigned web data on-chain

*The hardest sub-problem of [VISION.md](VISION.md): when a data source does
NOT sign its content, how trustlessly can a chain learn what it served?
This document states the cryptographic obstruction exactly, why a single
proxy/notary is irreducibly trusted, and the decomposition that drives the
floor down to the chain's own consensus.*

## 1. The obstruction (a theorem, not an engineering gap)

TLS 1.3 has two halves with opposite cryptographic properties:

- **Handshake — ASYMMETRICALLY signed.** `CertificateVerify` is the
  server's ECDSA/RSA signature (cert key) over a hash of the handshake
  transcript, including the server's ephemeral DH public key. Unforgeable
  without the server's cert private key. ⇒ a **transferable** proof of
  *"a genuine session with the authentic server S at time T."*
- **Application data — SYMMETRICALLY authenticated.** Each record's
  AES-GCM / ChaCha20-Poly1305 tag uses the **shared** session key (both
  parties hold it). The server never asymmetrically signs content. ⇒
  **anyone holding the session key can forge a valid (ciphertext, tag) for
  any plaintext.** The content is *not* transferably provable.

**Consequence.** A web-proof witness (proxy/notary) "binds" the content
only by being **trusted not to collude with the prover** — the prover
holds the session key, so prover + witness together fabricate any
"response." This is irreducible for a *single* witness: no gadget defeats a
symmetric MAC without the key or the server's cooperation
(information-theoretic).

## 2. The escape that needs no witness: source-signed data

When the source signs at the application layer, the symmetric half is
bypassed — the content carries its own asymmetric signature:

DKIM email · C2PA media · signed API responses / webhooks · signed
exchange & oracle feeds (Pyth/Chainlink) · signed government data.

These are **Web Credentials** (VISION §3): zero witness trust, verified
with Sui natives. **Prefer this class; maximize its coverage.** (A bare
OAuth *identity* JWT proves only "I hold an account" — the identity
primitive; *content* needs a signed API response / webhook, or a
witnessed fetch.)

## 3. The decomposition (the actual contribution)

Split every web-proof by forgeability and treat the halves differently:

**(a) The server-signed half → fraud-proven to ZERO added trust.** A
witness posts the full TLS transcript (handshake + records) with a bond.
Our VM already adjudicates signature checking, so it can fraud-verify, with
no witness honesty: cert chain → pinned CA root; `CertificateVerify` over
the transcript; the key schedule; AEAD tag internal consistency. ⇒
*"genuine session with the real server S at time T"* is **trustless**.

**(b) The symmetric content half → drive the witness down to the chain
itself.** Only *content authenticity* rests on the witness. Ladder, by
decreasing added trust:

| rung | added trust | notes |
|---|---|---|
| single proxy/notary (Reclaim/TLSNotary) | the proxy doesn't collude with the prover | what today's zkTLS assumes — the shaky one |
| DECO 3-party handshake | a liveness/timing assumption (commit before key release) | single-party, integrity from timing |
| **k diverse bonded witnesses** | collude with **k** independent ASNs simultaneously | any honest dissenting transcript slashes liars; **buildable now** |
| TEE-proxy (AWS Nitro) | the hardware vendor + auditable enclave code | `sui::nitro_attestation` native; no committee |
| **validator-set as witness** | **= the chain's own safety** (supermajority-honest) | THE FLOOR: adds nothing beyond the L1 you already chose |

The top rung is the breakthrough framing: **the only entity a chain's users
already trust is its validators, with their stake. If the fetch-witness
function is performed by the validator set as part of consensus, the
content attestation inherits the chain's own slashing/diversity/supermajority
security — the oracle adds NO trust beyond the chain hosting it.** An
on-chain oracle cannot be more trustless than its chain; this attains that
bound. You don't add an oracle — you teach the chain to witness.

## 4. The honest floor

- **Source signs** → no witness, trust = the issuer (a Web Credential).
- **Unsigned source** → the handshake is trustless (fraud-proven); the
  content rests on *the chain's own consensus* (top rung) or a *diverse
  bonded quorum* (today). Never a single trusted proxy.
- **Diversity (k-of-n issuers/witnesses)** caps how much any one
  compromise matters; **immutable signed snapshots** remove mutability;
  the **fraud-proven AI** makes everything above the floor trustless.
- Irreducible: a fact that *no credible entity and no witnessed session*
  ever produced cannot be known by any chain. We make the floor explicit,
  minimal, diversified, and bonded — and prove everything above it.

## 5. Build status

- Native-on-Sui pieces exist: `ecdsa_r1`/`ed25519`/`bls12381`/`groth16`/
  `nitro_attestation`, zkLogin (OAuth JWT in zk).
- Buildable now over the proven fraud-proof VM: in-VM TLS-handshake
  verification (signature + key-schedule checks — same op class as the
  softfloat we shipped), the `Credential` interface, k-of-n witness
  diversity, the optimistic web-proof fraud game.
- Research/protocol-change (the ideal floor): validator-set-as-witness.
