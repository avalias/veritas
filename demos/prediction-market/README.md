# Veritas prediction market

A prediction market on Sui that resolves on proof instead of votes. The outcome
of a market is decided only by zkTLS web proofs of what a real source published,
verified on-chain by native ecrecover and read by the deterministic Qwen judge
from [`opml/`](../../opml). You cannot submit an opinion. There is no proof that
a source said "no", so there is nothing to vote with. This is how it avoids
becoming the UMA-style token vote it's built to replace.

It's a demo of the engine, not the engine itself. The interesting part (the
fraud-provable AI judge) lives in `opml/`; this directory shows one product you
can build with it.

```
move/        the veritas Move package: market (CPMM) + credential + reclaim
web/         the dApp (wallet signing, the guided tour, the live judge stream)
*.py, go.sh  staging + auto-replenish scripts for a live devnet demo
docs/        DEVNET (run it), JUDGE (the 5-minute script), EVIDENCE/PROVENANCE (why it's not a vote)
```

## Run it

```bash
sui move test            # in move/ — 16 tests
./go.sh                  # starts the judge resolver + dApp + stages markets on devnet
```

Then open http://127.0.0.1:8777/app.html and follow the on-screen guided tour.
Full operator notes are in [docs/DEVNET.md](docs/DEVNET.md); the judge walkthrough
is in [docs/JUDGE.md](docs/JUDGE.md).
