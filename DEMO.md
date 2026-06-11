# DEMO.md — "Polymarket of the future": markets resolved by a fraud-provable LLM judge

*The full-featured demo design. Everything below runs on machinery that
already exists and is proven (see [ANALYSIS.md](ANALYSIS.md)) except the
items in §6, each scoped. The principle throughout: the market's outcome
is a **pure function of on-chain-committed bytes**, computable by anyone,
disputable down to one micro-op, with the chain as the only referee.*

## 1. The product in one paragraph

Anyone creates a market: a question, a resolution date, and a **judge
spec** (a committed model + prompt template + decode policy + source
policy). People trade YES/NO shares against an AMM. After the date, an
open **evidence window** lets anyone submit evidence — admissible ONLY
with a cryptographic provenance proof chaining to the committed source
policy (signed content from real publishers; no free text). Then any
**resolver** runs the judge — deterministically, on the committed evidence
— and asserts the verdict with a bond. Anyone can recompute and
challenge; fraud loses a bisection game that ends in one micro-op executed
by the Sui contract. After the window, winning shares redeem. No
committee, no multisig, no designated oracle: **the resolver role is
permissionless because being wrong is provable.**

## 2. Actors and lifecycle

```
 Creator          Traders        Evidence submitters     Resolver(s)      Challenger(s)
    │ create_market   │                  │                   │                │
    │ (judge_id,      │  buy/sell        │                   │                │
    │  question,      │  YES/NO via AMM  │                   │                │
    │  dates, fees) ──┼──────────────────┼───────────────────┼────────────────┤
    │                 │       [resolution date passes]       │                │
    │                 │                  │ submit_evidence   │                │
    │                 │                  │ (bytes + PROOF of │                │
    │                 │                  │  provenance)      │                │
    │                 │       [evidence window closes]       │                │
    │                 │                  │                   │ run judge      │
    │                 │                  │                   │ assert_verdict │
    │                 │                  │                   │ (bond, roots)  │
    │                 │       [challenge window]             │                │
    │                 │                  │                   │      challenge + bisection
    │                 │                  │                   │      → verify_step → slash
    │                 │       [finalize]                     │                │
    │                 │  redeem winning shares               │ resolver fee   │
```

## 3. The hard questions, answered

### 3.1 What exactly does the judge read? — PROVENANCE-GATED EVIDENCE

> The full literature-grounded design of this layer — who decides what to
> fetch (all discretion at creation time), trust-group independence
> accounting, zkTLS tiering, judge hardening against every published
> attack class, the extraction+symbolic-aggregation decomposition, and
> the pre-registered testing protocol — is in **[EVIDENCE.md](EVIDENCE.md)**.
> This section is the summary.

**Rejected design (v0): bonded free-text submission.** Anyone could
submit any paragraph; the "evidence set" degenerates into voting with
prose — the same human-discretion hole (UMA/Polymarket-style social
resolution) this system exists to close. Killed.

**The design: evidence is admissible ONLY with a cryptographic
provenance proof.** The market's judge spec commits a **source policy** —
a set of trust anchors — and a submission enters the evidence set iff it
carries a machine-verifiable attestation chaining to one of them:

| attestation class | proves | trust root | Sui verification | demo-real today? |
|---|---|---|---|---|
| **C2PA / Content Credentials** | publisher P published these exact image/video bytes | publisher cert on the C2PA / IPTC Origin trust list (pinned) | **`sui::ecdsa_r1` NATIVE** (manifests are COSE/ES256, P-256) | **YES** — BBC ships it on BBC Verify; IPTC signs news images in production; we downloaded + verified a real signed asset |
| **DKIM-signed news mail** | domain D's server sent these exact header+body bytes (breaking-news alerts/newsletters) | the domain's DNS DKIM key (pinned; archive.prove.email keeps timestamped history) | RSA not native → Move modexp (e=65537) OR zkEmail Groth16 via **`sui::groth16` NATIVE** | **YES** — BBC `50dkim1` (RSA-2048) verified live, stable 2024→2026; NYT, Reuters likewise |
| **zkTLS / web proof** | https://domain/… served these bytes (attestor-clock time T) | the proof system's attestor set (assumptions pinned) | **Reclaim has a LIVE Sui mainnet verifier** (`client::verify_proof`, `ecdsa_k1`) | YES (Reclaim); TLSNotary not production-ready |
| **signed on-chain feeds** | price/event facts | the chain / feed pubkey | **Pyth live on Sui** (ed25519 payloads); native | YES |
| **TEE attestation (bonus)** | this enclave ran this code | AWS Nitro PCRs | **`sui::nitro_attestation` NATIVE** | YES (soft-finality backstop only) |

Properties this buys:

- **Nobody can submit "whatever they want."** The admissible universe is
  exactly *what committed publishers actually published*, cryptographically.
  Submission stays permissionless — but permissionless within truth:
  anyone may bring any genuinely-published BBC/Reuters/AP item; no one
  can bring an invented one.
- **Selection attacks are symmetric and bounded**: an adversary can only
  choose *among real published items*; the other side can always submit
  the rest of the published record. The judge weighs real reporting, not
  fabrications.
- **Verification is itself fraud-provable.** Signature checks use Sui's
  native crypto where possible; anything heavier (manifest parsing,
  content-hash binding, proof verification) is a committed deterministic
  computation — exactly the class of thing our VM + bisection already
  adjudicates. Long-term unification: **the entire resolution function —
  verify attestations → extract text → assemble input → run the LLM →
  decision token — is ONE committed deterministic program**, disputable
  down to a single micro-op. The LLM is just the largest stage of a
  fully-verified pipeline. That is the product: an oracle with no oracle
  in it.
- The market question is evidence item 0; ordering/truncation of admitted
  items stays deterministic (by attestation hash, committed token budget).

**Source-policy registry**: market creators choose from registered
anchor sets (e.g., "C2PA news tier-1: BBC, AP, Reuters, AFP" — each entry
pins certificates/keys + the attestation format version). Anchors are
data, not trustees: they never act, sign nothing per-market, and cannot
censor a market — the worst a captured publisher can do is publish
falsehoods under its own cryptographic identity, which is exactly the
real-world trust people already place in a byline, now made explicit and
auditable.

### 3.2 The tokenization gap (named honestly)

The VM consumes token ids; tokenization (BPE) runs off-chain. v1 rule:
**evidence is submitted as (raw bytes, token ids) pairs and the token ids
are canonical**; the tokenizer hash is committed in the judge spec, so any
observer can detect a mismatched pair during the window — and a committed
rule says mismatched pairs are skipped by honest resolvers (a resolver who
includes a mismatched pair produces a different genesis than honest
recomputers ⇒ challenged at step 0 territory… **only if the genesis
insertion binds the ids**, which it does, §3.4). Residual risk: a
submitter whose (bytes, ids) mismatch goes unnoticed by every observer
during the window. Hardening options (post-demo, written down): tokenizer
as a committed VM program (BPE in micro-ops), or a zk-tokenization proof,
or byte-level judge input. This is the one trust seam in v1 and it is
windowed, observable, and small.

### 3.3 Verdict format: how text becomes a settlement

The judge spec commits a **decision protocol**: the prompt template ends
with an instruction to answer with exactly one token; the committed
output rule is "the FIRST generated token id" mapped through a committed
table `{yes_id → YES, no_id → NO, anything else → INVALID}`. INVALID
refunds collateral pro-rata (markets must price ambiguity). The output
region binding already exists on-chain (final-state output challenge,
SPEC §8.5) — the contract reads the token id from the proven final state,
not from the resolver's prose.

### 3.4 Binding the input on-chain (already designed, SPEC §7.2)

`genesis_F = insert(static_genesis_root, input_pages)` happens **on the
chain** at assertion time: the resolver supplies Merkle update proofs
that transform the audited model genesis into the per-question genesis by
writing exactly the committed evidence token ids into the input region.
The disputable trace therefore starts from an on-chain-derived root: a
resolver cannot run on different input than what the market committed.

### 3.5 Economics (sized from measured numbers)

| parameter | drives it | demo default |
|---|---|---|
| resolver fee | cost of one honest run (~minutes of CPU) + margin | 0.5% of pool, min fixed |
| resolver bond | must exceed challenger's full cost: recompute + ~38 dispute txs + capital lockup | ≥ 20× estimated challenge cost |
| challenger bond | anti-grief: lost if the assertion stands | ~½ resolver bond |
| evidence bond | anti-spam | small, refundable |
| challenge window | recompute time (minutes) + dispute time (measured: 25 rounds; localnet minutes, mainnet hours) + margin | 24h demo: 5 min localnet |
| max dispute span | one judgment ≈ 30–60M micro-ops → 25–26 rounds, verify_step at chain-minimum gas | bounded by window |

Security claim, precisely: **one honest party with a laptop and a
challenger bond makes every wrong verdict unprofitable.** That party can
be a trader with money in the market — the people with the strongest
incentive to check are already present.

### 3.6 Which judge for the demo

Both proven judges, registered side by side to show the architecture is
arithmetic-agnostic:
- **integer judge** (fast dispute demos: oracle ~1M steps/s),
- **committed-float judge** (the published-quality flagship: PPL 34.60).
The demo runs the happy path + a dispute on the integer judge (minutes),
and at least one assert+finalize on the float judge.

## 4. What the demo SHOWS (the script)

Scenario walk, end to end on localnet, one command:

1. **Create**: judge registry entry (program_root, genesis_root, template
   hash, decision table, tokenizer hash) → market "Will it rain in Paris
   tomorrow?" with dates and fees. AMM seeded.
2. **Trade**: three simulated traders move the price (CPMM swaps).
3. **Evidence (REAL provenance, real past event)**: a market on a settled
   past question, resolved from genuinely-attested sources:
   - a **C2PA-signed news image** (the verified IPTC asset, or a BBC
     Verify asset) → `ecdsa_r1` check on-chain;
   - a **DKIM-signed news alert** from a pinned domain (BBC `50dkim1`
     RSA-2048, key from archive.prove.email) → modexp/Groth16 check.
   Each verified AT SUBMISSION; rejected if the signature/trust-chain
   fails; admitted set ordered by attestation hash, committed.
4. **Resolve (honest)**: resolver runs the judge (native predictor for
   the answer + committed run for roots), inserts input on-chain,
   asserts YES with bond; window passes; market settles; winners redeem;
   resolver collects fee. *Show: every artifact hash matches a local
   recomputation.*
5. **Resolve (fraud)**: fresh market, dishonest resolver flips one weight
   bit (or asserts the wrong token) → challenger recomputes, challenges,
   **real bisection txs on localnet** → `verify_step` executes the one
   disputed micro-op in the Move VM → resolver slashed, challenger paid,
   market settles on the honest verdict.
6. **Stall**: resolver goes silent mid-dispute → clock-based timeout
   slash (already implemented in dispute.move).

## 5. What already exists (no work)

- dispute.move: Fact, bonds, windows, bisection entries, verify_step
  (integer + float ops), timeouts, output challenge — 54 Move tests.
- Judges: integer + float, both C-14-proven; fraud games + convictions.
- Localnet driver (client/), gas numbers, SuiJson plumbing.

## 6. Build list (the gap), in order

| # | item | est | notes |
|---|---|---|---|
| 1 | `market.move`: Market object, CPMM (x·y=k) over SUI collateral, YES/NO share balances, buy/sell/redeem, settle(Fact) | 1 day | simplest correct AMM; shares as table balances, not Coin types, for v1 |
| 2 | `evidence.move`: submission window, bonds, deterministic ordering, evidence-set commitment consumed by assert | 0.5 day | |
| 3 | `registry.move` + judge spec object (template hash, decision table, tokenizer hash) | 0.5 day | wraps existing JudgeParams |
| 4 | on-chain `insert_input` (Merkle update proofs static_genesis → genesis_F) | 1 day | fold exists; add update-path verify (~30 lines Move) + Rust proof builder |
| 5 | decision-table read from proven output region in settle path | 0.5 day | §8.5 machinery exists |
| 6 | `market_demo` client bin: the §4 script end-to-end on localnet | 1–1.5 days | reuses localnet.rs + qwen_dispute cursor parties |
| 7 | polish: README walkthrough, asciinema-style transcript, parameter table | 0.5 day | |

**Total: ~5–6 focused days to a full-featured, completely-secure demo.**

## 7. Postponed (written down, not forgotten)

- Non-interactive verification (one-shot ZK fault proofs → validity
  mode): SPEC **FW-8**, with model-pedigree decision gates.
- Tokenization-in-VM / zk-tokenization (closes §3.2's seam).
- Multi-outcome & scalar markets (same machinery, bigger decision table).
- Mainnet parameterization (real windows, real bonds, walrus/DA for
  evidence bytes).
