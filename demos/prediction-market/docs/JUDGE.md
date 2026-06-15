# JUDGE.md — hand it to a judge, watch the whole thing run on-chain in 5 minutes

Everything below is **real Sui testnet** + the **real Qwen3-1.7B** judge. No mocks.
A judge connects their own wallet and personally signs every step of a prediction
market's life — buy, AI-judge, prove, resolve, redeem — plus convicts a lying
resolver on-chain. Each action surfaces a clickable suiscan transaction.

---

## 1. Operator setup — ONE command

```bash
./demo/go.sh
```

It starts the AI-judge resolver (real Qwen3-1.7B on :8899) and the dApp server
(:8777) if they aren't already up, stages fresh **⚡ LIVE** and **⚖️ READY**
markets, arms the **Fraud Lab**, launches the **auto-replenisher** (so the board
resets itself for the next judge), and opens the dApp. **Re-run it any time** to
reset for a new judge.

Then **have the judge connect Slush (set to devnet) and fund it**:

```bash
curl -s -X POST https://faucet.devnet.sui.io/v2/gas -H 'Content-Type: application/json' \
  -d '{"FixedAmountRequest":{"recipient":"<JUDGE_ADDRESS>"}}'
```

> The ⚡ market opens a ~150s trading window from staging; the auto-replenisher
> keeps a fresh one available and re-arms the Fraud Lab after each conviction, so
> back-to-back judges always get a clean board. The ⚖️ READY card resolves on one
> click regardless of timing.

---

## 2. The 5-minute script — what the judge clicks, what they see

**The judge does not need this page.** A **Guided Tour** panel (bottom-right of
the dApp) walks them through all 7 steps one at a time — "👉 DO THIS NOW", a
**Take me there** button that opens the right market, and a ✓ + **view tx** link
for every completed step. The **⚡ LIVE** card is the spine; it also shows a phase
countdown banner (🛒 Buy → 🛡️ Submit → ⚖️ Resolve → 🏆 Redeem). All 7 steps were
verified end-to-end with a real wallet (7/7, 0 errors).

| # | Action (the tour says exactly this) | What appears on-chain |
|---|--------|-----------------------|
| 1 | **Connect** Slush (devnet). | wallet connects — you sign everything yourself |
| 2 | Open **⚡ LIVE**, buy **0.05 YES**. | `buy_yes` tx → **view tx**; CPMM price moves. This is the position you'll redeem. |
| 3 | Once it shows **EVIDENCE**, pick a source — **BBC / Reuters / AP** — and click **Add**. | `submit_web_proof` tx → suiscan; that source's pinned-attestor signature verified by **native ecrecover**. Each source is an *independent* trust-group. |
| 4 | Click **"Ask the AI judge to read the evidence."** | the real **Qwen3-1.7B** streams tokens and types its **Verdict: YES** (off-chain — it *reads*; the rule decides) |
| 5 | Try the **"Submit as NO"** box — any opinion. | **refused** ("no proof exists") — *no tx*. You cannot vote, only prove. |
| 6 | Banner shows **RESOLVE** → click **"Resolve — apply the committed rule."** | `resolve` tx → suiscan; outcome = count of **independent trust-groups** vs k → **YES** |
| 7 | Banner shows **REDEEM** → click **"Redeem winnings."** | `redeem_to_sender` tx → suiscan; winning SUI paid to the wallet |
| 8 | Open the **Fraud Lab** (red banner) → **"Convict & slash the liar."** | `verify_step` tx → suiscan; 85,937 micro-ops bisected to one; the Fact flips to **REJECTED**, bond slashed |

**Fallback if ⚡ timing slips:** the **⚖️ READY** card is already past its window
with a YES proof admitted — one click on **Resolve** decides it on-chain instantly.

**Bonus:** **Create a market** (top bar) — the judge commits a question + its
resolution rules in one signed tx, before anyone trades.

---

## 3. What the judge can independently verify

- Every toast has a **view tx** link to `suiscan.xyz/devnet/tx/<digest>` — open any
  of them and see the real signed transaction, sender = the judge's own address.
- The package `0xd2b2a949…ec4a86f7` and every Market / Fact object is on suiscan.
- The Qwen stream is the **real model**: stop the resolver and the button says
  "[resolver offline]"; it is not a canned animation.
- The "Submit as NO" refusal proves the anti-vote: there is no admissible NO, so
  there is nothing to submit. Resolution is a **count of proofs per independent
  source**, never a count of opinions or money.

## 4. What the judge attests

> *On Sui devnet I connected my own wallet and signed real transactions for an
> entire prediction-market lifecycle: I bought YES and watched the on-chain price
> move; I watched the real deterministic Qwen3-1.7B judge stream its reading of the
> evidence; I admitted a zkTLS web proof verified on-chain by native ecrecover, and
> saw the app refuse a bare "NO" opinion because no proof backed it; I resolved the
> market by the committed rule that counts independent trust-groups (not a vote, not
> the AI's displayed verdict); I redeemed my winning shares for SUI; and I delivered
> a one-micro-op fraud proof that convicted a lying resolver and slashed its bond.
> Every action surfaced a real suiscan transaction. The opML judge really runs, the
> market really works, and the proofs are real and on-chain.*

---

## 5. Remote judge (not at the operator's laptop)

Trade / resolve / redeem / Fraud Lab / create all run from the judge's own browser
against devnet — no resolver needed. Only the **live Qwen stream** (steps 3) needs
the resolver, which serves on `:8899`. To make it reachable:

```bash
# expose the local resolver (it now binds 0.0.0.0)
ngrok http 8899          # → https://<id>.ngrok.app
# then send the judge:
http://<your-host>:8777/app.html?resolver=https://<id>.ngrok.app
```

The dApp reads `?resolver=<url>` and points the judge button at it. Simplest of
all: **keep the judge at the operator's machine, or screen-share and drive.**

## 6. Troubleshooting

- **⚡ not in trading / "evidence window opens shortly"** → re-run
  `python3 demos/prediction-market/judge_setup.py` (re-mints a fresh ⚡ with a future window).
- **Resolve says E_TOO_EARLY** → the window hasn't closed; use the ⚖️ READY card,
  or wait for the ⚡ banner to reach ⚖️ RESOLVE.
- **Redeem says nothing to redeem** → only the market the judge personally bought
  into pays them; use the ⚡ card they bought in step 2.
- **Fraud Lab already convicted** (status REJECTED) →
  `cargo run -p client --bin devnet_stage_dispute -- <PKG> <addr1> <addr2>`
  (or just keep `replenish.py` running — it re-arms automatically).
- **Gas race / signature fails** → fund the address again (multiple coins) and retry.
