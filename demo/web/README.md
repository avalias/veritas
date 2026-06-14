# Veritas — live demo

Two things live here:

- **`app.html`** — the real dApp. Connect a Sui wallet and trade on
  **live markets deployed to Sui devnet**, submit a real zkTLS evidence
  proof (verified on-chain), and redeem. Real transactions, real wallet
  signing.
- **`index.html`** — the guided walkthrough (no wallet needed) that
  explains the whole system.

## Run it

It's all static — no build step.

```
python3 demo/web/serve.py        # → http://127.0.0.1:8777/app.html
```

(Open `app.html` over the local server, not `file://`, so it can fetch
`markets.json`.)

## To actually trade (real on-chain)

1. Install the **[Slush](https://slush.app)** wallet (or any Sui
   wallet-standard wallet).
2. Switch the wallet network to **devnet**.
3. Fund your address from the devnet faucet:
   `curl -s -X POST https://faucet.devnet.sui.io/v2/gas -H 'Content-Type: application/json' -d '{"FixedAmountRequest":{"recipient":"<YOUR_ADDRESS>"}}'`
4. Open `app.html`, click **Connect wallet**, pick a market, and buy YES/NO.

## What's deployed

`demo/web/config.json` / `markets.json` hold the live devnet package and the
seeded markets:

- 4 open markets to trade (Starship, GPT-6, BTC $150k, Fed cut),
- 1 market in its evidence window with a real zkTLS proof ready to submit.

Package and markets are on a real public network — verify any transaction
on [suiscan devnet](https://suiscan.xyz/devnet).

## Why this isn't a vote (the UMA problem, solved)

You can't submit an opinion as evidence. You can only submit a **zkTLS
proof of what a pinned source actually served** — verified on-chain by
Sui's native `ecrecover`. Confirmations are counted **per independent
source**, not per submission, so spamming doesn't add up to a vote. A fixed
public AI reads the real content and extracts the answer, and a wrong
verdict is **provable on-chain and slashes the liar**. Money prices the
market; it never decides the outcome.
