# JUDGE.md — hand it to a judge, watch the whole thing run on-chain in 5 minutes

Everything below is **real Sui devnet** + the **real Qwen-0.6B** judge. No mocks.
A judge connects their own wallet and personally signs every step of a prediction
market's life — buy, AI-judge, prove, resolve, redeem — plus convicts a lying
resolver on-chain. Each action surfaces a clickable suiscan transaction.

---

## 1. Operator setup — three terminals, ~60 seconds

```bash
# 1) the AI judge (loads the real committed-float Qwen-0.6B, serves :8899)
cargo run -p qwen --release --bin resolver

# 2) the dApp
python3 demo/web/serve.py                     # → http://127.0.0.1:8777/app.html

# 3) stage the live markets  ← run this LAST, right before handing over
python3 demo/judge_setup.py
```

Optional 4th terminal — keep the board fresh for back-to-back judges (it stages a
new ⚡/⚖️ the instant one is resolved, and re-arms the Fraud Lab after a conviction):

```bash
python3 demo/replenish.py
```

Then **have the judge connect Slush (set to devnet) and fund it** before they start:

```bash
curl -s -X POST https://faucet.devnet.sui.io/v2/gas -H 'Content-Type: application/json' \
  -d '{"FixedAmountRequest":{"recipient":"<JUDGE_ADDRESS>"}}'
```

> Run `judge_setup.py` **last**: the ⚡ market opens a ~100s trading window from
> that moment, so the judge catches it in trading. If they take longer, just
> re-run `judge_setup.py` (or let `replenish.py` do it) — the ⚖️ market resolves
> on one click regardless of timing.

---

## 2. The 5-minute script — what the judge clicks, what they see

The **⚡ LIVE** card is the spine: one market the judge walks through its entire
lifecycle, live, each step a real signed tx. The board guides them with a phase
countdown banner (🛒 Buy → 🛡️ Submit → ⚖️ Resolve → 🏆 Redeem).

| # | Action | Market | What appears on-chain |
|---|--------|--------|-----------------------|
| 1 | Connect Slush (devnet). Open any trading market, buy **0.05 YES**. Watch the price tick up. | BTC $150k (or any of 8) | `buy_yes` tx → **view tx** suiscan link; CPMM price moves |
| 2 | Open the **⚡ LIVE** card (in trading). Buy **0.05 YES** — this is the position you'll redeem. | ⚡ LIVE | `buy_yes` tx → suiscan; "your position: 0.05 YES" |
| 3 | While it trades, click **"Ask the AI judge to read the evidence."** Watch the real **Qwen-0.6B** stream tokens and type its verdict. | ⚡ LIVE (or the Evidence market) | live SSE token stream → **Verdict: YES** (off-chain by design — this is the judge *reading*, not deciding) |
| 4 | Banner flips to 🛡️ **EVIDENCE**. Click **"Admit this zkTLS proof on-chain."** | ⚡ LIVE | `submit_web_proof` tx → suiscan; a pinned-attestor signature verified by **native ecrecover** |
| 5 | Try the **"Submit as NO"** box — type any opinion. It's **refused** ("no proof exists"). | ⚡ LIVE | *no tx* — you literally cannot vote; you can only prove |
| 6 | Open the **Fraud Lab** (red banner). Watch 85,937 micro-ops bisect to one. Click **"Convict & slash the liar."** | Fraud Lab | `verify_step` tx → suiscan; the Fact's status flips to **REJECTED**, bond slashed |
| 7 | Banner flips to ⚖️ **RESOLVE**. Click **"Resolve — apply the committed rule."** | ⚡ LIVE | `resolve` tx → suiscan; outcome computed by counting **independent trust-groups** (1 YES ≥ k=1) → **YES** |
| 8 | Banner flips to 🏆 **REDEEM**. Click **"Redeem winnings."** Winning SUI lands in the wallet. | ⚡ LIVE | `redeem_to_sender` tx → suiscan; SUI paid out |

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
> move; I watched the real deterministic Qwen-0.6B judge stream its reading of the
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
  `python3 demo/judge_setup.py` (re-mints a fresh ⚡ with a future window).
- **Resolve says E_TOO_EARLY** → the window hasn't closed; use the ⚖️ READY card,
  or wait for the ⚡ banner to reach ⚖️ RESOLVE.
- **Redeem says nothing to redeem** → only the market the judge personally bought
  into pays them; use the ⚡ card they bought in step 2.
- **Fraud Lab already convicted** (status REJECTED) →
  `cargo run -p client --bin devnet_stage_dispute -- <PKG> <addr1> <addr2>`
  (or just keep `replenish.py` running — it re-arms automatically).
- **Gas race / signature fails** → fund the address again (multiple coins) and retry.
