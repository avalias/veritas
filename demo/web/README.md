# Veritas — the human-facing demo

A single, self-contained web page that walks anyone through the whole
product: a real market with locked rules, provenance-gated evidence (with
live signed-source badges), the deterministic AI judge resolving it, the
fraud-proof that slashes a liar, and the side-by-side contrast with the
UMA token-vote that lost $7M on this exact question.

## Run it

It is one static file with no build step and no dependencies.

```
open demo/web/index.html          # macOS: just double-click it
# or serve it (any static server):
python3 demo/web/serve.py         # → http://127.0.0.1:8777
```

Everything is interactive: tap YES/NO to move the price along the bonding
curve, try submitting unsigned "evidence" and watch it get refused, press
"run the judge" to resolve, and "challenge a dishonest verdict" to watch
the dispute bisect a 29.5M-step trace down to one micro-op and slash the
liar.

## What's real underneath

The page is a faithful simulation of the deployed system, which is real:

- the on-chain market is `dispute/sources/market.move` (66 Move tests green);
- evidence admission is `dispute/sources/credential.move` — `ed25519` and
  native `ES256`/C2PA verification (the news-photo Content Credentials
  standard), with a real ES256 vector proven on-chain;
- the full lifecycle has been driven live on a Sui localnet by
  `dispute/demo/market_e2e.py`, with publisher signatures verified
  on-chain;
- the judge fault conviction is `dispute/tests/fqwen_conviction.move`.

See [PRODUCT.md](../../PRODUCT.md), [VISION.md](../../VISION.md), and
[EVIDENCE.md](../../EVIDENCE.md).
