# Headless verification of the live dApp

Reproduces the checks used to validate `demo/web/app.html` against Sui
devnet (rendering, console errors, transaction build+dry-run, and a full
wallet-signed buy via an injected wallet-standard wallet).

```
npm install playwright && npx playwright install chromium
python3 ../../demo/web/serve.py &        # serve the dApp on :8777
node inject.mjs    # ephemeral funded wallet → Connect → Buy YES (real tx on devnet)
node txbuild.mjs   # buy / submit_web_proof / create_market all build + dry-run
node modal.mjs     # screenshot the trade + evidence modals
```

`inject.mjs` proved the end-to-end signing path: a wallet following the Sui
wallet-standard connects, the dApp builds the tx, the wallet signs and
executes it on devnet, and the UI confirms with a tx digest. A real wallet
(Slush) implements the same standard, so it behaves identically.
