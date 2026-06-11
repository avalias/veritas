# WEBPROOFS.md — the trust floor for getting unsigned web data on-chain

*The hardest sub-problem of [VISION.md](VISION.md): when a data source does
NOT sign its content, how trustlessly can a chain learn what it served?
This document states the cryptographic obstruction exactly, why a single
proxy/notary is irreducibly trusted, and the decomposition that splits a
fetch into a zero-trust (server-signed) half and an irreducibly
observer-trusted (symmetric-content) half — NOT a free lunch.*

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

**(b) The symmetric content half → ALWAYS costs an observer you trust.**
This is the corrected, honest statement (an earlier draft wrongly claimed
this half could reach "the chain's own safety" — see §3.1). You cannot
fraud-prove an observation, only a signature; the symmetric content is an
**attested observation**, so SOME observer is trusted. The defenses, by
*increasing* trust:

| option | trust root | honest caveat |
|---|---|---|
| **avoid it — use source-signed content** | the issuer's key | not a witness at all (Web Credential); ALWAYS prefer |
| **k diverse independent witnesses** | ≥1 of k is honest | works ONLY for public, globally-consistent facts (honest observers should agree); bonds DON'T help (fabrication is internally consistent ⇒ unprovable ⇒ unslashable) — the only signal is observers DISAGREEING |
| **DECO 3-party handshake** | a liveness/timing assumption | single-party, integrity from commit-before-key-release |
| **TEE-proxy (AWS Nitro)** | the **hardware vendor** + enclave code | a SEPARATE computer/root, ORTHOGONAL to L1 stake — never collapses into validator trust |
| single proxy/notary (Reclaim/TLSNotary) | that one observer doesn't collude with the prover | today's zkTLS — the weakest |

No option reaches the chain's consensus floor, because external observation
is not deterministically recomputable by validators (each fetch may see
different bytes; a wrong observation is often not even provable, so not
slashable). The strong move is therefore to **avoid the symmetric half**
(source-signed content) and, where unavoidable, **use diversity scoped to
public globally-consistent facts** — not to pretend a witness inherits the
L1.

### 3.1 Why a witness can't reach chain security (the distinction I owe)

Two trust types, not to be conflated:

- **Verifiable consensus** (the chain's real security): validators agree
  on facts EVERY one can independently, deterministically recompute from
  shared data; misbehavior is cryptographically PROVABLE (double-sign →
  slash). "Don't trust, verify."
- **Attested observation** (any oracle): someone CLAIMS to have seen
  external reality; others cannot recompute it (not in the session; the
  server may serve different bytes per observer); a WRONG observation is
  often not even provable, so cannot be slashed. Bonds deter only provable
  faults — and symmetric-content fabrication is the unprovable one.

Unsigned web content is irreducibly attested observation. So it can NEVER
have the chain's everyone-checks security; there is always an observer set
you trust. Validator-witnessing is a NEW, weaker assumption layered on the
chain (honest supermajority performs + agrees on a non-recomputable
external action) — strictly MORE trust than chain consensus, not equal,
and not free. The TEE rung is a DIFFERENT computer rooted in a hardware
vendor's attestation key, orthogonal to (and additive with) L1 stake.

## 4. The honest floor

- **Source signs** → no witness, trust = the issuer (a Web Credential).
- **Unsigned source** → the handshake half is trustless (fraud-proven);
  the content half rests on a trusted observer set — minimized via
  source-signing (avoid it), else k diverse observers SCOPED TO PUBLIC
  globally-consistent facts, else a TEE (hardware-vendor root). It is never
  zero and never the chain's consensus floor.
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
- NOT a free lunch: there is no construction that makes unsigned-content
  observation as trustless as on-chain state. The win is the DECOMPOSITION
  (zero-trust handshake + minimized-observer content) and the strong
  preference for source-signed Web Credentials — not a magic witness.
