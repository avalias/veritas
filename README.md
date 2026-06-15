# Veritas

Two things live here.

**`opml/` is verifiable LLM inference for Sui.** A compiler that turns any
quantized language model (with float activations) into a program for a small
fraud-proof VM, run off-chain at near-native speed (with some simple kernel tweaks it will be native soon), with every output disputable
down to a single arithmetic operation that a ~330-line Sui Move contract
re-executes and slashes. We tested everything extensively to match bit to bit.

No trusted hardware, no
committee, just a contract anyone can read. This is a big inovation, read more in
[opml/README.md](opml/README.md), we explain what made this opML breakthorugh on Sui finally feasible, and compare to older solutions. This is our gift to Sui community,
we think that opML LLM inference will drive the next wave of innovations on Sui. There are countless applications, once you can verify LLM execution on Sui, with no proving ovehead, and almost instant verification. Feel free to use it in your projects. A new wave of LLM using blochain apps will come to Sui. Because, we solved this theoretical quetion!

**`demos/prediction-market/` is a market that resolves on proof, not on votes.**
Its outcome is decided only by zkTLS web proofs, verified on-chain by native
ecrecover and read by the deterministic AI judge from `opml/`. Nobody votes; you
can only submit a proof of what a real source served. It's one example of what
the engine makes possible.

```
opml/                      the engine (Rust crates + Move verifier + docs)
demos/prediction-market/   the demo dApp, market contract, and judge resolver
VISION.md                  why route existing crypto trust instead of inventing oracles
HANDOFF.md                 full resume doc: testnet addresses, open threads, gotchas
```

## Run it end to end (cold start)

Three processes when everything is up: the **AI judge** `:8899`, the **dApp** `:8777`,
and the self-hosted **zkTLS attestor** `:8001` (plus a small proof-gen server `:8788`).
Run every command from the repo root.

**Prerequisites:** Rust (`rustup`, `~/.cargo/bin` on PATH) · **Node 20 or 22 via nvm —
not 24** (24 has a crypto-backend bug the zkTLS client works around) · the **Sui CLI**
(set the Slush wallet to **testnet** to sign in the dApp) · **Python 3**.

```bash
# 1. Fetch the Qwen judge weights (~4 GB, gitignored). Skip this and it falls back to
#    the bundled 0.6B reference; fetch it for the smarter Qwen3-1.7B the demo uses.
./opml/models/qwen/fetch-1.7b.sh

# 2. Start the demo stack: AI judge resolver (:8899) + dApp (:8777), stage fresh markets,
#    arm the Fraud Lab, and keep the board fresh. Re-run any time to reset.
./demos/prediction-market/go.sh          # opens http://127.0.0.1:8777/

# 3. The self-hosted zkTLS attestor (:8001) — unlimited, no Docker, Node 20/22. It signs
#    with the key the demo markets pin (address 0x17c5…d025), so its proofs are admitted
#    on-chain unchanged. Full runbook: tools/zktls/README.md
nvm use 22
git clone --depth 1 https://github.com/reclaimprotocol/attestor-core tools/zktls/attestor-core
cd tools/zktls/attestor-core && npm install && npm run download:zk-files
PRIVATE_KEY=0x4242424242424242424242424242424242424242424242424242424242424242 PORT=8001 npm run start

# 4. The browser proof-gen server the live demo calls (go.sh auto-starts it if the
#    attestor is already up on :8001). From a fresh shell at the repo root:
cd tools/reclaim && npm rebuild re2 && node gen_server.mjs     # :8788
```

Then open <http://127.0.0.1:8777/> and pick a demo:

| page | what it shows |
|------|---------------|
| **Slash a lying LLM** | a resolver bonded a fake verdict; run the real deterministic judge, watch it disagree, re-run the one disputed micro-op on-chain and take the bond. |
| **Watch the AI judge read the news** | the real Qwen3-1.7B reads a real news report and settles a bettable yes/no question ("did the Fed raise rates?", "did the merger go through?") where the wording never matches the question. |
| **Prove live web data** | generate a real zkTLS proof of a live source (BTC price, a football result), watch it recover the pinned attestor, then try to forge the value and watch ecrecover break. |
| **A market, end to end** | one real market on Sui: trade, admit a zkTLS proof, the judge reads it, resolve, redeem. |

`app.html` combines all of it with a guided tour. Each on-chain action links its suiscan tx.

## Tests

```bash
cargo test                                         # Rust: VM, compiler, kernels, fraud game
cd opml/move && sui move test                       # the on-chain verifier (54 tests)
cd demos/prediction-market/move && sui move test    # the market contract (16 tests)
```

## Continue development

The Qwen weights (`opml/models/qwen/artifacts-1.7b/`) and the attestor clone
(`tools/zktls/attestor-core/`) are **gitignored and recreated by the fetch scripts**
above. The judge model is **Qwen3-1.7B**; the runtime is model-agnostic (reads all
dims from config, needs tied embeddings) and **0.6B is the bundled reference** whose
quality we measured (committed-float perplexity 34.5974 = the published model).

[HANDOFF.md](HANDOFF.md) is the full resume doc: the deployed testnet package IDs and
operator address, the open threads (the deepest being on-chain evidence→Qwen genesis
binding, which makes a *wrong verdict* slashable at the market level), and the
operational gotchas. See [opml/README.md](opml/README.md) for the engine and
[demos/prediction-market/docs/DEVNET.md](demos/prediction-market/docs/DEVNET.md) for
the running demo.

Apache-2.0.
