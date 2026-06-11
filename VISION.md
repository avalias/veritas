# VISION.md — the general system: an Authenticated Computation Layer

*What we are actually building, of which "a fraud-provable LLM prediction
market" is one instance. This document states the general primitive, why
it strictly generalizes (and beats) every alternative, the one genuinely
new idea, and the honest boundary. Companions: [DEMO.md](DEMO.md) (the
first instance), [ANALYSIS.md](ANALYSIS.md) (what is built+measured),
[PRIOR_ART.md](PRIOR_ART.md) (the landscape + novelty check),
[SPEC.md](SPEC.md) (normative).*

## 1. The thesis: dissolve the oracle problem by routing trust, not creating it

The oracle problem — "how does a chain learn a true fact about the world?"
— has only ever been answered by **adding a trusted party**: token-voters
(UMA), a TEE committee (EigenAI), a threshold-signing node (APRO), a
data-provider quorum (Chainlink). Each new system is a new thing to trust.

Our answer inverts it. The world is **already** saturated with
cryptographic attestations: publishers sign content (C2PA/ES256),
newsletters are DKIM-signed, TLS authenticates served bytes, DNS publishes
the keys, exchanges sign feeds, enclaves attest. The missing piece was
never a trusted oracle — it is a way to **compute over that existing web
of signatures, trustlessly, and land the result on-chain.**

So the general system is not a market and not an oracle. It is a
**trustless `f(authenticated_inputs) → output`**: any committed
deterministic program over any cryptographically-attested inputs, where
being wrong about the result is provable on-chain in one micro-op. We do
not create trust. We route existing cryptographic trust through verified
computation.

## 2. The one elegant primitive: one program, one trace, one dispute

Competitors glue trusted modules together — a zkTLS verifier *and* an LLM
node *and* threshold signers, each trusting the others. We collapse the
**entire trust-to-answer path into one committed program in one VM,
disputed as one object:**

```
verify provenance (C2PA / DKIM / zkTLS sigs) → extract text → tokenize
        → run the model → decision/output
└──────────────────── one committed deterministic program ───────────────┘
        every stage = micro-ops; a single bisection covers ALL of it
```

The signature check and the matmul are **the same kind of thing**: lines
of the trace. Lie about a signature's validity → the bisection lands on
the signature-check micro-op. Lie about the matmul → it lands there. Same
machinery, **no module boundaries, no inter-component trust.** ECDSA-P256
/ RSA-modexp / hash verification are integer field arithmetic — they
compile into the exact ISA where we already proved IEEE-754 softfloat and
a full Qwen forward pass. A signature is ~thousands of micro-ops against
the model's billions: **authenticated I/O in the trace is essentially
free.**

## 3. The genuinely new idea: authenticated I/O as a VM primitive

Every verifiable-compute system (zkVM, optimistic VM) proves `f(x) = y`
but treats **"x is real"** as out of scope — someone trusted supplies x.
That is the gap every oracle falls into. We make *"this input is
admissible because attestation A by a committed anchor over these exact
bytes verifies"* part of the verified statement itself.

The committed program **includes** its input-authentication. There is no
separate trusted "evidence layer": evidence admissibility is just the
opening micro-ops of the same trace that runs the model. Verifying the
computation and authenticating its inputs become **one** fraud-provable
object. That fusion is the contribution.

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

What turns the built market into the general substrate:

1. **In-VM provenance verifiers** — ECDSA-P256 first (also a Sui-native
   dispute fast-path), then RSA-modexp / Groth16-wrapped DKIM. New
   committed *programs*, no new mechanism — exactly how softfloat was
   added.
2. **A `Claim` object** generalizing the market `Fact`:
   `assert(program_root, authenticated_inputs, output)` settled by the
   same bisection → one-micro-op verifier.
3. **Composition** — a claim's output may be another claim's authenticated
   input, yielding a *verifiable computation graph*: cheap to assert,
   cheap to check optimistically, expensive only to a liar, and the thing
   that makes this a platform rather than a feature.

## 8. One line

**The world already signs its facts; we are the trustless machine that
computes over those signatures and makes the answer an on-chain object no
one has to be trusted for.** Prediction markets are the first thing you
build on it — not the thing it is.
