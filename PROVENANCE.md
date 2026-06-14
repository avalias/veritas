# PROVENANCE.md — how evidence is authenticated (what's built, what's next)

*Direct answers to: did we use zkTLS? ed25519 or something more impressive?
should the judge also run in a Nitro TEE? This documents the credential
layer as it now exists in code — [credential.move](dispute/sources/credential.move),
[tee.move](dispute/sources/tee.move) — and the path for the rest. Theory:
[WEBPROOFS.md](WEBPROOFS.md). Sourcing policy: [EVIDENCE.md](EVIDENCE.md).*

## The one interface

Every piece of evidence is authenticated by exactly one call:

```
credential::verify(scheme, publisher_key, message, signature) -> bool
```

A market pins, at creation, a set of `(scheme, key, trust_group)` issuers.
Evidence is admitted only if it verifies under one of them. There is no
trusted oracle — only a publisher's own signature, the same trust a byline
already carries, now machine-checked and diversified (k-of-n independent
groups).

## What's built and tested today

| scheme | what it authenticates | on Sui | status |
|---|---|---|---|
| **ED25519** | signed feeds/oracles (Pyth), generic signers | **native** | ✅ verified on-chain; live localnet E2E |
| **ES256 (P-256/SHA-256)** | **C2PA "Content Credentials"** — signed news photos/video (BBC Verify, IPTC, Adobe), and ES256 OAuth JWTs | **native `ecdsa_r1`** | ✅ **real ES256 vector verified on-chain** (`credential_tests`) |

ES256 is the important one: it is the **news industry's actual signing
standard for media provenance**, and Sui verifies it natively (cheap, no
off-chain trust). A real P-256 signature is checked on-chain in the test
suite. The generator also pins a **real DKIM public key pulled live from
DNS** (Reuters `selector1`), so the key-provenance path is real, not toy.

## ed25519 vs zkTLS — the honest answer

They authenticate different things; we use the right tool per source:

- **A source that signs its content** (C2PA media, DKIM email, signed
  API/feed) → use its signature directly. **No zkTLS, no committee, no
  notary.** This is always preferred. ed25519 and ES256 cover this today;
  DKIM/RS256 (RSA-2048) is admitted via a **zkEmail-style Groth16 proof**
  (`sui::groth16` is native) because naive 2048-bit modexp in Move is
  gas-prohibitive — `credential::verify` aborts on raw RSA rather than
  silently passing, forcing the Groth16 path.
- **A source that does NOT sign** (an arbitrary web page, a score on a
  site) → this is the only place **zkTLS** earns its keep. Reclaim has a
  **live Sui mainnet verifier** (`client::verify_proof`), so a Reclaim web
  proof is a drop-in evidence tier. But zkTLS carries an **attestor trust
  assumption** (WEBPROOFS §1): the prover holds the TLS session key, so a
  colluding attestor can fabricate content. So in the source policy it is
  the **last-resort tier, capped below k** (EVIDENCE.md §3): it can
  corroborate, never carry a verdict alone, and all proofs sharing an
  attestor count as one trust group.

So: **not "ed25519 or zkTLS" — both, ranked.** Publisher-signed first
(strongest, no committee), zkTLS only for unsigned sources (weakest,
priced). The buzz is zkTLS; the substance is the signed-publisher tiers.

Wiring a Reclaim proof is the same shape as the table above — one more
`scheme` whose `verify` calls Reclaim's on-chain verifier — and is the
next credential to add. It needs a real proof from Reclaim's attestor
network to test end to end, which is why it is documented here rather than
shipped tonight.

## Should the judge also run in a Nitro TEE? Yes — as a second layer

Built: [tee.move](dispute/sources/tee.move). The fraud proof is the **hard
guarantee** (the judge ran correctly, no hardware trust, a liar is
slashed). The TEE is **defense in depth**, not a replacement:

- A runner runs the committed judge image inside an **AWS Nitro enclave**
  and produces an attestation. Sui's **native** `nitro_attestation`
  verifies the COSE signature + the full AWS Nitro cert chain on-chain;
  `tee::verify_judge_enclave` then binds **PCR0** (the enclave image
  measurement) to the **exact judge build the market committed**.
- This gives fast, **optimistic soft-finality** (you can trust the
  enclave's output immediately, with the fraud window as backstop) and a
  **second independent wall**: an attacker must now both forge an AWS
  Nitro attestation *and* win a one-micro-op bisection.
- The trust roots are **orthogonal and additive** (WEBPROOFS §3.1): TEE =
  the hardware vendor; fraud proof = the chain. Neither weakens the other.

Honest boundary: a real attestation requires actual Nitro hardware running
the committed image; the native verification is Sui's (and Sui's framework
tests cover it), while the sample attestation's on-chain parse is
protocol-version-gated, so the full enclave E2E runs on a matching network,
not in `sui move test`.

## The layered picture

```
strongest, no committee  ┌─ C2PA / ES256        (native, BUILT, tested)
                         ├─ DKIM / RS256-JWT     (Groth16, native verifier)
                         ├─ signed feeds/ed25519 (native, BUILT, tested)
                         ├─ TEE-attested fetch   (nitro, BUILT: tee.move)
weakest, priced, capped  └─ zkTLS / Reclaim      (live Sui verifier, next)
                                  │
                                  ▼
        credential::verify(scheme, key, msg, sig)  ── one gate ──▶ market admission
                                  │
        + judge runs (optionally TEE-attested) and is fraud-provable to one micro-op
```

Every layer is either a publisher's own signature (no new trust) or a
proof the chain checks. The market never trusts a node; it trusts the
named issuers a creator pinned, diversified k-of-n, and proves everything
computed on top.
