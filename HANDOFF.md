# HANDOFF — resume from here

Everything below is committed to git on the `main` branch. There is no remote
yet: to give this to someone else, `git push` to a GitHub repo first (the weights
and the attestor clone are gitignored; the fetch scripts re-create them).

## What this is

Two things in one repo:

- **`opml/`** — verifiable LLM inference for Sui. A compiler turns any quantized
  LLM (float activations) into a program for a small fraud-proof VM. The model
  runs off-chain at near-native speed and is bit-for-bit deterministic; anyone can
  dispute an output, and a ~330-line Sui Move contract re-runs the single disputed
  micro-op and slashes the liar. Read `opml/README.md`.
- **`demos/prediction-market/`** — a market on Sui whose outcome is decided only
  by zkTLS web proofs (native ecrecover) read by the deterministic Qwen judge.
  No vote. Read `demos/prediction-market/README.md`.

## Repo map

```
opml/
  vm/ kernels/ gpu/ compiler/ game/ client/ benches/   Rust crates (the engine)
  models/qwen/        committed-float Qwen runtime; artifacts/ = 0.6B, artifacts-1.7b/ = 1.7B
  models/qwen/src/bin/resolver.rs   the judge HTTP/SSE server (:8899)
  move/               the opml verifier Move package (dispute, interp, merkle, softfloat, signed, tee)
  docs/               SPEC, ANALYSIS, PRIOR_ART
demos/prediction-market/
  move/               the veritas market package (market, credential, reclaim)
  web/                the demos: index + slashing/judge/evidence/market .html + app.html, on veritas.js
  *.py, go.sh         staging, auto-replenish, real-zkTLS submit
  docs/               DEVNET, JUDGE, EVIDENCE, PROVENANCE, WEBPROOFS, DEMO
tools/
  reclaim/            zkTLS client (gen.mjs) + the SDK (node_modules gitignored)
  zktls/              self-hosted attestor runbook (the attestor-core clone is gitignored)
  shot/               playwright verification scripts
```

## Deployed on Sui testnet

| | |
|---|---|
| opml verifier package | `0xbe6d4a7e3b569854b6cdebc4d6a2d8dae75049db50dfe6e309f643b589e23068` |
| veritas market package | `0x128f9a3c051da659ba7bb94f18d77c553f08603d96a35b3d8cea8905e610c19a` |
| operator / deployer | `0xb0d94005a671368abac192c6d89f1ceb164934a6d01710d94f1937b72cee3f55` (≈0.28 SUI — **top up**) |
| fraud challenger | `0xb01503bef9a3acaab095a9269d21a5a8def0069478d5d4f8c5fbc6b0a4a650c9` |
| our zkTLS attestor key | `0x42…42`, address `0x17c5185167401ed00cf5f5b2fc97d9bbfdb7d025` (= pinned source 0) |
| a real zkTLS proof admitted on-chain | market `0x2654205963c555787be91fda0bbfda187edba3f6b02caa2d5b91cdb521368bd8` |

Config the dApp/scripts read: `demos/prediction-market/web/config.json` (network,
rpc, both package ids, gas_budget/seed_liq for lean testnet staging) and
`web/markets.json` (the staged markets + attestor addresses, written by the seed scripts).

## Start everything (cold)

```bash
# 1. once: the smarter judge model (4 GB, gitignored)
./opml/models/qwen/fetch-1.7b.sh

# 2. the demo stack (judge resolver on :8899 + dApp on :8777 + stage markets + replenisher)
./demos/prediction-market/go.sh
#    go.sh uses the 1.7B if fetched, else the 0.6B reference.

# 3. the self-hosted zkTLS attestor on :8001 (Node 20/22, NOT 24; no Docker)
nvm use 22
git clone --depth 1 https://github.com/reclaimprotocol/attestor-core tools/zktls/attestor-core
cd tools/zktls/attestor-core && npm install && npm run download:zk-files
PRIVATE_KEY=0x4242424242424242424242424242424242424242424242424242424242424242 PORT=8001 npm run start
#    full runbook: tools/zktls/README.md
```

Then open <http://127.0.0.1:8777/> and pick a demo. Tests:
`cargo test` (Rust), `sui move test` in each `move/` dir (54 + 16 green).

## The judge model

- The runtime is model-agnostic (reads all dims from config; needs tied
  embeddings, which 0.6B and 1.7B have, 4B+ do not). `resolver.rs` picks the dir
  from `QWEN_DIR` (defaults to the 0.6B reference). `fetch-1.7b.sh` downloads and
  merges the 1.7B shards into the single `model.safetensors` the loader reads.
- The prompt is a strict fact-checker that must answer **YES / NO / UNKNOWN** with
  a one-sentence reason. The resolver emits the exact prompt as the first SSE
  event so the UI shows it. Verified: orbit → YES, scrubbed → NO, irrelevant →
  UNKNOWN, "BTC > $100k" with price $66k → NO.
- To go smarter still, fetch a bigger tied-embedding Qwen3 and point `QWEN_DIR` at
  it. 4B+ would need the loader's `assert!(tie_word_embeddings)` relaxed and a
  separate lm_head.

## Real zkTLS (unlimited, self-hosted, no Docker)

Proven end to end. Our attestor signs with the key whose address the markets pin,
so its proofs are admitted by `reclaim.move` unchanged. `tools/reclaim/gen.mjs`
generates a proof of any URL through it; `demos/prediction-market/submit_real_zktls.py`
creates a market and submits one on-chain. Working sources (Qwen reads them well):

```bash
python3 demos/prediction-market/submit_real_zktls.py coinbase   # live BTC price
python3 demos/prediction-market/submit_real_zktls.py football   # a TheSportsDB result
```

Gotchas: run the attestor on Node 20/22 (not 24); `npm rebuild re2` in
`tools/reclaim` to match your Node; the client pins webcrypto (already done in
gen.mjs); copy the zk files into the client if missing (see tools/zktls/README.md).

## The demos (web/)

- `slashing.html` — the story: a resolver bonded a faked verdict; you run the real
  deterministic judge, it disagrees, and you slash him on-chain. Comparison table
  vs opML/zkML/TEE.
- `judge.html` — the real Qwen3-1.7B reads evidence and types YES/NO/UNKNOWN with a
  reason; shows the exact prompt; free-text box proves it's live, not canned.
- `evidence.html` — add a zkTLS proof (ecrecover). Two attacks shown failing:
  an opinion (no signature), and a forgery (the page runs the real ecrecover and
  the recovered address breaks when you change the claim).
- `zktls.html` — pick a real API (Coinbase price, USD/EUR, a Hacker News headline,
  a match result), generate a real zkTLS proof live through the attestor, see it
  recover the pinned attestor, and the judge reads the proven value.
- `market.html` — a market resolved by counting independent proofs, then paid out.
- `app.html` — all of it combined, with a guided tour.

## Open threads / what's next

1. **In-browser live zkTLS is DONE** (zktls.html + tools/reclaim/gen_server.mjs on
   :8788). Next, if wanted: wire the same live-gen into evidence.html's market
   submit so a user proves-and-submits a fresh proof to a real market in one flow.
2. **On-chain evidence→Qwen binding** (the genesis-construction step, SPEC §7.2):
   makes a *wrong judge verdict* slashable at the market level. The Fraud Lab
   already proves the conviction machinery; binding live evidence to it is the
   remaining cryptographic build. This is the deepest open item.
4. **Top up testnet gas** on the deployer and challenger; **push to a git remote**.
5. Optional: re-measure committed-float PPL for the 1.7B; deeper code-comment
   de-slop pass on the older Rust/Move (the human-facing surfaces are done).

## Operational notes

- Three long-running processes: judge `:8899`, attestor `:8001`, dApp `:8777`.
  `go.sh` starts the first and third; the attestor is separate (Node 22).
- Testnet faucet is rate-limited; fund the deployer from an existing wallet if the
  faucet 429s. Gas budgets in config are kept low because faucet coins are small;
  `judge_lib` merges coins when they fragment.
- The fraud-staging bin switches the active sui address per call and restores it;
  if a run dies mid-way, `sui client switch --address 0xb0d9…` to get back.
- Slush wallet must be set to **testnet** to sign in the dApp.
