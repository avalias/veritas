# VISION.md — the general system: an Authenticated Computation Layer

*What we are actually building, of which "a fraud-provable LLM prediction
market" is one instance. This document states the general primitive, why
it strictly generalizes (and beats) every alternative, the one genuinely
new idea, and the honest boundary. Companions: [DEMO.md](DEMO.md) (the
first instance), [ANALYSIS.md](ANALYSIS.md) (what is built+measured),
[PRIOR_ART.md](PRIOR_ART.md) (the landscape + novelty check),
[SPEC.md](SPEC.md) (normative).*

## 1. The thesis: the impossibility, and the maximal escape

A blockchain has no senses; it cannot perceive an election, a launch, a
rainfall. So **someone must vouch for the raw fact** — no construction and
no math conjures a trustless world-fact onto a chain that did not witness
it. Every oracle that claims otherwise has merely hidden the voucher
(UMA's token-voters, Chainlink's nodes, APRO's zkTLS notary).

Total trust therefore factors: **Trust(raw fact) × Trust(everything done
with it).** The maximal honest goal:

- drive **Trust(computation) → 0** — deterministic execution + fraud
  (now) / validity (later) proofs. **Done and measured.**
- drive **Trust(raw fact) → its irreducible floor**, and make that floor
  **explicit, minimal, diversified, market-chosen, and immutable** — never
  a hidden oracle.

Two design rules get us to the floor, and they are the whole trick:

1. **Read only SIGNED STATEMENTS, never mutable state.** A signed thing —
   an OAuth JWT, a DKIM email, a signed API/feed response, a C2PA
   manifest — is immutable the instant it is signed and carries its own
   timestamp. "The publisher changed their website" cannot alter a
   statement their key signed at time T. This deletes the entire class of
   mutability/retroactive-slash shakiness.
2. **The voucher is the issuer's OWN signature, chained to a pinned PKI
   root — not a new trusted party.** You trust "Sign-in-with-Twitter is
   really Twitter" exactly as much as the whole internet already does, and
   nothing more.

We do not create trust. We route the web's existing cryptographic trust —
at its irreducible, explicit, diversified floor — through verified
computation.

## 2. The one elegant primitive: one program, one trace, one dispute

Competitors glue trusted modules together — a zkTLS verifier *and* an LLM
node *and* threshold signers, each trusting the others. We collapse the
**entire trust-to-answer path into one committed program in one VM,
disputed as one object:**

```
 [native Sui: verify each Web Credential — JWT(zkLogin)/DKIM/feed/C2PA;
  immutable SIGNED statements only, never mutable web state]
                          │  (cheap, one-shot, off the trace)
                          ▼
 k-of-n credentials from independent pinned issuers  →  deterministic AI
        judge (read · weigh · synthesize)  →  verdict
   └──────────── one committed deterministic program, bisectable ─────────┘
        + augmentation game: a new credential that flips the verdict slashes
```

Credential verification is native-chain and one-shot (NOT micro-ops). What
lives in the fraud-proven trace is the part no node can be trusted with:
the AI's reasoning over the credentials, and the deterministic completeness
check. Diversity (k-of-n independent issuers) bounds raw-fact trust;
immutable signed snapshots remove mutability risk. **No trusted module
boundaries inside the answer.**

## 3. The general primitive: the Web Credential, a diversity predicate, a deterministic AI

**The Web Credential.** Strip every real-world trust source to its
skeleton and they are the same object:

    Credential = (claim, signature, key, proof-key-authentic-under-pinned-root, time)

| source | claim | key authenticity chains to |
|---|---|---|
| **OAuth JWT** (Sign-in-with-Google/Twitter/Apple) | account/handle X has/did Y | provider JWKS → TLS cert → Certificate Transparency |
| **DKIM email** (Reuters/AP/NYT alerts) | domain D's server sent these bytes | DNS DKIM key → DNSSEC |
| **signed API/feed** (exchanges, Pyth, banks) | endpoint asserts value V at T | API signing key / cert |
| **C2PA** (media — ONE instance, not the system) | publisher P produced these bytes | C2PA trust list |
| **TLS cert + CT** | domain is keyed K | CA roots → CT logs |
| **eIDAS / gov seal** | official document says Z | EU trust list |

C2PA is one row, not the design. The design is **"a signed statement whose
key authenticity reduces to a small set of on-chain-pinned PKI roots"** —
which covers OAuth (every major platform), email, signed feeds/APIs,
government data, media: an enormous, growing fraction of *valuable* data,
**none of it needing a committee**, because the issuer's own key is the
root. (Genuinely *unsigned* web content — arbitrary HTML — still needs a
witness: TEE-fetch via `sui::nitro_attestation` (native, no committee)
preferred, zkTLS/notary last. Honest: a chain cannot get unsigned facts
more trustlessly than *some* witness — true for everyone.)

**Safety by diversity, not by a single voucher.** A market commits a
PREDICATE over a credential multiset; safety comes from **k-of-n agreement
across INDEPENDENT pinned issuers**:

    YES iff the deterministic AI judge, reading >= k credentials from
    distinct pinned issuers in {AP, Reuters, X-verified, AFP, ...} signed
    within [t0, t1], concludes YES.

The market is safe unless k independent real-world publishers are
simultaneously compromised AND the judge ran wrong (impossible —
fraud-proven). "Is this fact true on-chain" reduces to "do k independent
entities you explicitly chose, who already sign their content, agree — as
read by a fixed, public, provably-executed AI."

**The compute is the deterministic AI, fraud-proven** (proven: integer +
float Qwen, bit-exact, convicted on-chain). Reading/weighing/synthesizing
the credentials is the part that genuinely cannot be trusted to a node —
that is the in-VM win; attestation verification stays NATIVE on Sui.

**Completeness by adversarial augmentation.** "Did the judge see all the
evidence?" is not cryptographic and is not proven directly. Instead,
omitting a decision-relevant credential is a publicly-triggerable slash:
anyone submits another valid credential; if re-running the deterministic
judge with it FLIPS the verdict, the resolver is slashed and the answer
corrected (deterministic ⇒ fraud-provable; griefing priced — a
non-flipping item costs its submitter the bond).

**The math (general, partly native already).** Sui's **zkLogin** is a
deployed circuit verifying an OAuth JWT against pinned provider keys in
zero-knowledge on-chain — the existence proof. The general object is a
**universal web-credential verifier**: one interface over `ecdsa_r1`
(ES256/C2PA, native), `ed25519` (native), and Move-RSA-modexp or a
zkEmail-style `groth16` (DKIM/RSA-JWT — `groth16` verify is native), so
ANY row above becomes one on-chain `Credential`. That generalization —
zkLogin/zkEmail → a universal PKI-credential SNARK — turns the entire
existing internet trust infrastructure into an on-chain-queryable source,
with no notary and no node network.

## 4. What makes it general (the axes)

| axis | range | status |
|---|---|---|
| **program f** | any deterministic program compiled to the ISA (LLM, parser, rule engine, simulation, signature verifier) | integer + float forward passes proven; ISA has control flow |
| **inputs** | anything with an expressible attestation: C2PA/ES256 (Sui-native curve), DKIM-RSA (modexp / zkEmail-Groth16), zkTLS (Reclaim has a live Sui verifier), signed feeds (Pyth on Sui), TEE (sui::nitro_attestation), on-chain facts | provenance classes researched + Sui-native verifiers confirmed |
| **outputs** | binary / scalar / multi-class / structured; bound to the proven final state | output-region challenge exists (SPEC §8.5) |
| **applications** | markets, oracles, insurance, bridges, private identity/KYC, verifiable scrapers, agent gating | one mechanism, many specs |
| **proof system** | optimistic now (~0 honest overhead, FW-6 done) → validity-proof upgrade later (FW-8), same commitments | optimistic complete; zk banked |
| **composition** | a claim's output is another claim's authenticated input → a verifiable computation graph | design (below) |

## 5. Why it beats each alternative — it is their strict generalization

Each competitor is a special case that **trusts the rest of the pipeline**:

| system | proves compute | authenticates inputs | judgment trustless | honest cost |
|---|---|---|---|---|
| UMA / Polymarket | ✗ human vote | ✗ | ✗ | — |
| Chainlink / Pyth | ✗ relays | signs feeds only | ✗ | — |
| EigenAI | determinism, **final hash only** | ✗ | TEE committee | ~2% |
| ORA opML | ✓ fraud proof | ✗ (garbage in) | ✓ | high (generic MIPS VM) |
| APRO × Brevis | ✗ trusted LLM node | ✓ zkTLS feeds | ✗ | — |
| zkML | ✓ validity | ✗ | ✓ | **10⁴–10⁵×** |
| **ours** | **✓ per micro-op** | **✓ in the same trace** | **✓ bit-deterministic** | **~0 (measured)** |

Only our row is ✓✓✓, achieved at *optimistic* cost with the *validity*
upgrade as a drop-in — **zkML-grade trustlessness at opML-grade price,
with inputs authenticated too.** PRIOR_ART §7 confirms no existing project
occupies this combination.

## 6. The honest boundary

We prove the committed model **ran correctly on genuinely-attested
evidence**. We do **not** prove the model's judgment is *wise* — that is a
model-quality question, identical to trusting a named human judge, except
the judge is now fixed, public, auditable, and provably executed on real
inputs. A strictly better trust boundary; claiming more would be the exact
dishonesty this system exists to remove. Two real seams remain, both
named and bounded: tokenization off-chain (DEMO §3.2; closeable by
tokenizer-in-VM), and the attestation schemes' own roots (a publisher can
still sign a falsehood under its own auditable identity — the same trust a
byline already carries, now explicit).

## 7. The minimal generalization of the architecture

1. **Web-Credential I/O.** A `Credential` interface verified NATIVELY on
   Sui per item (zkLogin for OAuth JWT; `ecdsa_r1` for ES256/C2PA;
   `ed25519`; Move-RSA / `groth16` for DKIM/RSA). Only signed, timestamped
   statements admitted — no mutable state, no committee where the issuer
   signs. Unsigned sources fall back to TEE-fetch (`nitro_attestation`,
   native) then zkTLS (priced).
2. **A market `Predicate`** over a credential multiset: `k-of-n` across
   independent pinned issuers within a time window, evaluated by the
   committed deterministic AI judge — diversity is the raw-fact safety
   knob, fraud proofs are the compute safety.
3. **Two games over one `Claim`** (generalizing the market `Fact`):
   compute-correctness (bisection → one micro-op) AND completeness
   (augmentation → deterministic flip → slash).
4. **Composition** — a claim's verdict becomes another claim's credential
   (it is signed by the chain itself): a *verifiable computation graph*.

The universal credential verifier (zkLogin/zkEmail generalized) is the one
piece worth building as new crypto; everything else is assembly over
native Sui + the proven fraud-proof VM.

## 8. One line

**The world already signs its facts; we are the trustless machine that
computes over those signatures and makes the answer an on-chain object no
one has to be trusted for.** Prediction markets are the first thing you
build on it — not the thing it is.
