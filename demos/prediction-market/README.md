# Veritas prediction market

A prediction market on Sui that resolves on proof instead of votes. A market's
outcome is decided only by zkTLS web proofs of what a real source published,
verified on-chain by native ecrecover and read by the deterministic Qwen judge
from [`opml/`](../../opml). You cannot submit an opinion. There is no proof that
a source said "no", so there is nothing to vote with. That is how it avoids
becoming the UMA-style token vote it replaces.

This is a demo of the engine, not the engine itself. The fraud-provable AI judge
lives in `opml/`; this directory shows one product built on it.

## Four demos, each showing one thing

Open `web/index.html` (the four are deployed on Sui testnet):

| page | what it shows |
|------|---------------|
| `web/slashing.html` | Slash a lying LLM. Re-run the one disputed micro-op on-chain and take the resolver's bond. The point of the whole project. |
| `web/judge.html` | Watch the real Qwen3-0.6B read a proven fact and type its verdict, deterministically. Off-chain, no wallet. |
| `web/evidence.html` | Add evidence as a real zkTLS proof, verified by native ecrecover. Try to submit an opinion instead; it refuses. |
| `web/market.html` | A market resolved by counting independent proofs against a committed rule, then paying out. |

`web/app.html` is the same capabilities combined into one guided walkthrough.

```
move/        the veritas Move package: market (CPMM) + credential + reclaim (16 tests)
web/         the demos: index + the four pages + app.html, on a shared veritas.js
*.py, go.sh  staging + auto-replenish scripts for a live demo
docs/        DEVNET (run it), JUDGE (the 5-minute script), EVIDENCE/PROVENANCE (why it's not a vote)
```

## Run it

```bash
cd move && sui move test          # 16 tests
./go.sh                           # judge resolver + dApp server + staged markets
```

Then open http://127.0.0.1:8777/ and pick a demo. The live judge stream (judge.html
and app.html) needs the resolver from `opml/`; everything else runs from the
browser against testnet. Operator notes are in [docs/DEVNET.md](docs/DEVNET.md);
the guided 5-minute walkthrough is in [docs/JUDGE.md](docs/JUDGE.md).
