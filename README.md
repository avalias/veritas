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
```

## Quick start

The engine:

```bash
cargo test                          # Rust: VM, compiler, kernels, fraud game
cd opml/move && sui move test       # Move: the on-chain verifier (54 tests)
```

The demo:

```bash
cd demos/prediction-market && sui move test   # the market contract (16 tests)
./demos/prediction-market/go.sh               # run the live dApp on devnet
```

See [opml/README.md](opml/README.md) for the engine and
[demos/prediction-market/docs/DEVNET.md](demos/prediction-market/docs/DEVNET.md)
for the running demo.

Apache-2.0.
