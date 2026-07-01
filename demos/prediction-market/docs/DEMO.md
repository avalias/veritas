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
> This section is the summary. The credential layer is now BUILT: `dispute/sources/credential.move` (ed25519 + native ES256/C2PA, real vector verified on-chain) and `dispute/sources/tee.move` (Nitro second layer). See [PROVENANCE.md](PROVENANCE.md).

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

The judge runs the committed model on a forced reply format — `VERDICT:
<YES|NO|UNKNOWN>` then a one-line REASON — and its **output token stream**
is what the dispute binds (SPEC §7.3 / final-state output challenge §8.5).
The judge identity commits three verdict-token ids `(yes, no, unknown)`, and
the on-chain reading (`market::decode_verdict`) is **earliest of the three
wins**: the first committed token to appear in the proven stream decides —
`yes → YES`, `no → NO`, `unknown → ABSTAIN` — and none-present is also
ABSTAIN. The contract reads the token ids from the **proven final state**,
never from the resolver's prose.

ABSTAIN means the judge supports neither side: an evidence item asserted as
YES or NO over a judge that actually abstained is a mis-extraction, droppable
by anyone (`drop_misextracted`), so ambiguity can't be laundered into a
confident settlement.

This token rule is the on-chain twin of the off-chain resolver's
`extract_verdict`, which scans the **decoded text** for the earliest
standalone word among {YES, NO, UNKNOWN}. Committing the UNKNOWN id (not just
yes/no) is what keeps the two faithful: without it, an `UNKNOWN` verdict whose
REASON sentence contained a stray "no"/"yes" token would settle on-chain as
NO/YES while the resolver and the dApp showed ABSTAIN.

### 3.4 Binding a counter-extraction to the item, on-chain (implemented)

`drop_misextracted` is what makes a wrong verdict slashable. It acts on a
counter-extraction Fact only when **every** link of the chain is bound
on-chain — no field of the Fact is taken on faith:

- **the Fact stood for real** — it is FINALIZED *and* committed a challenge
  window and per-move timeout ≥ the market's committed minima, so it could not
  be self-asserted and instantly finalized (a zero-window Fact proves nothing);
- **its verdict is its own computation** — `output` is opened against the Fact's
  final root `root_n` (state_root / halted / step / out_base fold), so lying
  about the verdict requires lying about a `root_n` the bisection game adjudicates;
- **it ran THIS market's judge** — `program_root` and memory depth `d` match the
  committed judge identity;
- **on exactly THIS item's input** — `genesis_F = insert(static_genesis_root,
  input_pages)` is reconstructed on-chain by folding the caller-supplied input
  pages into the audited static image root (so the resolver can't run on
  different input than committed), *and* that reconstructed genesis equals the
  item's committed `content_hash`. The signed-feed convention is that a droppable
  item's `content_hash` **is** the genesis of its tokenized judge input — so the
  item is welded to its own input, and a Fact over a fabricated input can't drop it;
- **and the proven verdict disagrees** with the item's asserted claim.

Only then is the item dropped (idempotently — the state mutation is the last
step). zkTLS/Reclaim items commit `content_hash` to the web claim, not a
judge-input genesis, so they are **corroborating-only**: they cannot be slashed
by `drop_misextracted`, matching their capped-tier trust role (EVIDENCE.md §3).

The one residual is inherent to any optimistic system: a Fact that **no honest
watcher challenges within the committed window** stands on a `root_n` that was
never adversarially re-executed. That trust is bounded by the real
window/timeout, the slashable bond, and the on-chain genesis reconstruction
above — a challenger who runs the public deterministic judge always wins — and
is removed entirely only by the postponed validity-proof backstop (§FW-8).

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
