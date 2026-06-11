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
  [native Sui: verify each corpus item's attestation — C2PA/DKIM/zkTLS/TEE]
                              │  (cheap, one-shot, off the trace)
                              ▼
   committed attested corpus  →  deterministic Agent (search · reason ·
        synthesize over the corpus)  →  decision / output
   └──────────── one committed deterministic program, bisectable ─────────┘
```

Attestation verification is native-chain and one-shot (NOT micro-ops —
Sui does it cheaper). What lives in the fraud-proven trace is the part
that genuinely cannot be trusted to a node: the agent's reasoning over the
corpus, and the deterministic completeness check. Lie about the reasoning
→ bisection lands on a model/control micro-op; omit a decision-relevant
attested item → the augmentation game flips and slashes. **No trusted
module boundaries inside the answer.**

## 3. The genuinely new idea: a deterministic agent over an attested corpus

NOT "verify signatures inside the VM" — Sui checks signatures natively,
so doing it as micro-ops is strictly worse (same trust, more gas, no
benefit). A thing belongs inside the deterministic VM only if it CANNOT
be done trustlessly outside. Signature verification fails that test. Two
things pass it, and they are the contribution:

**(a) The agentic reasoning.** "Deep research" naively means the model
fetches live web content — non-deterministic I/O (pages change, geo-vary,
rate-limit; the challenger can't reproduce the runner's bytes), which is
unbisectable. We flip it: the answer is a deterministic function

    answer = Agent(question, committed_attested_corpus, model)

The corpus is a set of evidence items each carrying a provenance
attestation, grown PERMISSIONLESSLY (anyone may add any genuinely-attested
item in the window). The Agent — the LLM doing the research-grade
reasoning, searching the corpus, following citations within it, weighing
sources, synthesizing — runs DETERMINISTICALLY (the part we proved
bit-exact). No fetch inside the deterministic core: "research" becomes
reasoning over a large authenticated pool, which is where the intelligence
actually lives. THIS is the in-VM win nobody else has — opML proves the
compute but not the reasoning's inputs; zkTLS-oracles authenticate one
fetch but the LLM step is a trusted node.

**(b) Completeness by adversarial augmentation.** The unsolved hole in
every research oracle — "did the searcher search honestly/completely?" —
is not a cryptographic property and cannot be proven directly. We do not
prove it. We make OMITTING a decision-relevant attested fact a
publicly-triggerable slashing condition: anyone may submit an additional
attested item; if re-running the deterministic Agent with it included
FLIPS the answer, the resolver is slashed and the answer corrected. "Does
adding E change the output" is itself deterministic and fraud-provable.
The resolver is forced to comprehensiveness by the threat that anyone can
expose a missing source — the adversarial logic of the dispute, applied to
the search. Griefing is priced (an item that doesn't flip costs the
submitter their bond).

Attestation verification itself stays where it belongs — NATIVE on Sui
(`ecdsa_r1` for C2PA/ES256, `groth16` for zkEmail/zkTLS,
`nitro_attestation` for TEE), one cheap check per corpus item, NOT in the
VM trace.

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

1. **Attested-corpus I/O.** A `Corpus` object: provenance-attested items,
   grown permissionlessly in a window; each attestation verified NATIVELY
   on Sui at submission (no VM burden, no committee where the source
   signs). The deterministic Agent reads only the committed corpus — I/O
   non-determinism eliminated.
2. **The deterministic Agent** = the committed program: searches/reasons
   over the corpus to an answer, bit-exact (the float/integer judges
   already proven; the agent loop is bounded control flow over them).
3. **Two games over one `Claim`** generalizing the market `Fact`:
   compute-correctness (bisection → one micro-op) AND completeness
   (adversarial augmentation → deterministic "does E flip it" → slash).
4. **Composition** — a claim's output may be another claim's attested
   input, yielding a *verifiable computation graph*: cheap to assert,
   cheap to check optimistically, expensive only to a liar.

**Trust menu for corpus items (cleanest first):** (i) publisher signs
natively (C2PA/DKIM/signed feed) — the publisher's own key, NO committee,
native verify; (ii) TEE-attested fetch for unsigned pages (AWS Nitro,
`sui::nitro_attestation` native) — hardware root, no committee;
(iii) zkTLS (Reclaim, live Sui verifier) — notary committee, last resort,
assumption priced. Irreducible truth: the open web does not authenticate
its own content (TLS leaves no transferable proof — why zkTLS needs a
witness), so fully-trustless fetch of UNSIGNED content is impossible for
anyone; we win by biasing the corpus toward natively-signed sources where
there is no committee at all.

## 8. One line

**The world already signs its facts; we are the trustless machine that
computes over those signatures and makes the answer an on-chain object no
one has to be trusted for.** Prediction markets are the first thing you
build on it — not the thing it is.
