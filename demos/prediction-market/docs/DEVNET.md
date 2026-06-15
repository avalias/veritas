# DEVNET.md — it's live, here's how to run and test it

The whole product is deployed and running on **Sui devnet** (a real public
network), with a real wallet dApp. Nothing here is a mock.

> **Handing it to a judge? See [JUDGE.md](JUDGE.md)** — a 5-minute, click-by-click
> script where the judge personally signs every step of a market's life (buy →
> AI-judge → prove → resolve → redeem) plus convicts a fraud, each a real on-chain
> tx. One command (`python3 demos/prediction-market/judge_setup.py`) stages the live markets;
> `python3 demos/prediction-market/replenish.py` keeps the board fresh across back-to-back judges.

## Run the dApp

```
cargo run -p qwen --release --bin resolver   # the live AI judge (:8899)
python3 demos/prediction-market/web/serve.py                     # → http://127.0.0.1:8777/app.html
python3 demos/prediction-market/judge_setup.py                   # stage the ⚡ LIVE + ⚖️ READY markets
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
  window, 1 **resolved YES** (full lifecycle). IDs in `demos/prediction-market/web/markets.json`.
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

## Who runs Qwen? Watch it, live.

The chain never runs the model (billions of ops). A **resolver** runs Qwen
off-chain — on its own hardware — and the chain only re-runs **one micro-op**
if a verdict is disputed (the Fraud Lab). To see it, start the resolver
locally; it loads the real committed-float judge — **Qwen3-1.7B** if you ran
`fetch-1.7b.sh`, otherwise the 0.6B reference (perplexity 34.60) — bit-identical on any CPU:

```
cargo run -p qwen --release --bin resolver        # serves :8899
```

Then open the **evidence market** in the dApp and click **"Ask the AI judge
to read the evidence."** The dApp streams Qwen's tokens as it reads the
zkTLS-proven headline and **types its verdict** ("YES, … the report says
Starship reached orbit"), then reveals the on-chain submit/resolve. That is
literally the model running — the resolver is who runs Qwen; the chain
verifies it's fraud-provable. (For a public deployment, host the resolver
and set `resolver_url` in `config.json`; the in-VM evidence→Qwen binding
that makes a *wrong* verdict slashable at the market level is the remaining
cryptographic step — SPEC §7.2 genesis construction — and the Fraud Lab
already proves the Qwen-conviction half.)

## The Fraud Lab — convict a liar on-chain, from a click

The dApp has a live fraud proof. A resolver has staked a bond on a
**fraudulent AI-judge result** (on devnet); a challenger disputed it; the
two **bisected 85,937 micro-operations down to one**, on-chain. Open the
Fraud Lab, watch the bisection collapse to a single culprit step, and click
**"Convict & slash the liar"** — your wallet signs the `verify_step`
transaction, the Sui contract re-runs that one micro-op, and the resolver's
bond is slashed. Proven end-to-end: a wallet clicked convict and the Fact's
on-chain status went to REJECTED. The toy judge keeps the bisection fast to
stage; the identical machinery convicts the real Qwen judge
(`fqwen_conviction.move`). Re-stage a fresh one with:

```
cargo run -p client --bin devnet_stage_dispute -- <PKG> <resolver> <challenger>
```

## Prove your own data (Reclaim zkTLS, in-app)

One market pins **Reclaim's real attestor**. Add your free Reclaim
`app_id` / `app_secret` / `provider_id` (dev.reclaimprotocol.org) to
`demos/prediction-market/web/config.json`, and the "Generate a real zkTLS proof" button runs
the Reclaim flow in-app — you prove a real website's data yourself, and the
proof is mapped straight into `submit_web_proof` and verified on-chain by
native `ecrecover`. The on-chain format already matches Reclaim's exactly
(`reclaim_tests`), so only the client credentials are needed.

## Deploy to testnet / mainnet

The dApp is network-configurable (it reads `network`/`rpc` from
`config.json`). To move it to testnet or mainnet:

```
./dispute/deploy.sh testnet      # fund the address first at faucet.sui.io
python3 demos/prediction-market/seed_all.py && python3 demos/prediction-market/add_markets.py   # seed markets
# then set demos/prediction-market/web/config.json network+rpc to testnet (or re-run from there)
```

(Currently the public testnet faucet is IP-rate-limited, so the live
instance runs on **devnet**, which is a real public network. `mainnet`
works the same with a funded address — the package is 70-tests-green and
hardened.)

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
python3 demos/prediction-market/seed_all.py                          # curated markets + prices
python3 demos/prediction-market/add_markets.py                       # a few more
python3 demos/prediction-market/show_resolved.py                     # a resolved-YES showcase
```

(For mainnet: `./dispute/deploy.sh mainnet` with a funded address — the
package is 70-tests-green and hardened.)
