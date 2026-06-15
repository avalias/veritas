# Veritas

Two things live here.

**`opml/` is verifiable LLM inference for Sui.** A compiler that turns any
quantized language model (with float activations) into a program for a small
fraud-proof VM, run off-chain at near-native speed, with every output disputable
down to a single arithmetic operation that a ~330-line Sui Move contract
re-executes and slashes. No zero-knowledge proving, no trusted hardware, no
committee, just a contract anyone can read. This is the part we think is worth
keeping. See [opml/README.md](opml/README.md).

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
