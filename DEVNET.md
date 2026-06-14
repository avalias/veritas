# DEVNET.md — it's live, here's how to run and test it

The whole product is deployed and running on **Sui devnet** (a real public
network), with a real wallet dApp. Nothing here is a mock.

## Run the dApp

```
python3 demo/web/serve.py        # → http://127.0.0.1:8777/app.html
```

- Browse 10 live markets (read straight from devnet, no wallet needed).
- Install [Slush](https://slush.app), set it to **devnet**, fund it:
  `curl -s -X POST https://faucet.devnet.sui.io/v2/gas -H 'Content-Type: application/json' -d '{"FixedAmountRequest":{"recipient":"<YOUR_ADDRESS>"}}'`
- Connect, and you can really: **trade YES/NO**, **submit a zkTLS evidence
  proof**, **resolve**, **redeem**, and **create your own market** — each a
  real signed transaction.

## What's deployed (verify any of it on [suiscan devnet](https://suiscan.xyz/devnet))

- **Package**: `0xd2b2a9493f91dc61d4dcc9f20f973600f39cb1e1cce1a155fe93ac16ec4a86f7`
  (modules: `market`, `dispute`, `credential`, `reclaim`, `tee`).
- **10 markets**: 8 open for trading (Starship, GPT-6, BTC $150k, Fed cut,
  Real Madrid, Avatar, hottest year, AI Math Olympiad), 1 in its evidence
  window, 1 **resolved YES** (full lifecycle). IDs in `demo/web/markets.json`.
- A real zkTLS proof admitted on-chain:
  [tx F3DtJMvf…](https://suiscan.xyz/devnet/tx/F3DtJMvfMG2QJQPu4JUKtZDH3C6uh6NKzrNKb2RyQPv1).

## What was verified end-to-end on devnet

- **Trading** moves the on-chain CPMM price (real `buy_yes`/`buy_no`).
- **zkTLS evidence**: a real attestor signature verified on-chain by native
  `ecrecover` (`submit_web_proof`), admitted into the trust-group count.
- **Full lifecycle**: create → evidence window → proof admitted → resolve =
  **YES** (a completed market is in the grid).
- **Every dApp transaction** (buy, submit_web_proof, resolve, redeem,
  create_market) **builds and dry-runs successfully** against devnet; the
  position display reads real on-chain state via `position_of`.

## The clever evidence design (why this isn't a vote)

This is the UMA problem, solved — and the dApp makes it visceral (open the
evidence market and try the "vote NO" box):

- You **cannot submit an opinion**. The only admissible evidence is a
  **zkTLS proof of what a pinned source actually served**, verified
  on-chain. There is no proof of a source saying NO, so there is nothing to
  submit. You can't vote — you can only prove.
- Confirmations are counted **per independent source**, never per
  submission, so copies and spam don't add up to a vote (the contract
  rejects duplicate issuer keys and counts at the trust-group level).
- A fixed public **AI judge reads the real content** and extracts the
  answer; a wrong verdict is **provable on-chain and slashes the liar**.
- Money **prices** the market; it never **decides** it.

## Re-seed / redeploy

```
./dispute/deploy.sh devnet                       # publish (prints PACKAGE_ID)
python3 demo/seed_all.py                          # curated markets + prices
python3 demo/add_markets.py                       # a few more
python3 demo/show_resolved.py                     # a resolved-YES showcase
```

(For mainnet: `./dispute/deploy.sh mainnet` with a funded address — the
package is 70-tests-green and hardened.)
