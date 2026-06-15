# Self-hosted zkTLS attestor (real, unlimited, no Docker)

The hosted Reclaim service caps the free tier at ~25 proofs/month. That cap is
billing on their API, not a property of the attestor. Run your own attestor and
it is unlimited, and you control the key it signs with. Our `reclaim.move`
verifies the attestor signature on-chain by native ecrecover, so a proof from
our own attestor is accepted with no change.

We sign with the key `0x42…42`, whose address is
`0x17c5185167401ed00cf5f5b2fc97d9bbfdb7d025` — the source the demo markets
already pin (source 0). So real proofs this attestor produces are admissible
on-chain out of the box.

This runs on **Node 20 or 22** (not 24, which has a crypto-backend bug the
client works around). No Docker.

## Run the attestor

```bash
nvm use 22
git clone --depth 1 https://github.com/reclaimprotocol/attestor-core tools/zktls/attestor-core
cd tools/zktls/attestor-core
npm install
npm run download:zk-files
PRIVATE_KEY=0x4242424242424242424242424242424242424242424242424242424242424242 PORT=8001 npm run start
# logs: "WS server listening" with signerAddress 0x17c5…d025 on :8001 /ws
```

## Generate a real proof of a live website

The client lives in `tools/reclaim` (the SDK is installed there). It pins the
webcrypto implementation before the claim, which is the one Node gotcha to
respect. `gen.mjs` defaults to the live Coinbase BTC price; change `TARGET_URL`
and `TARGET_REGEX` for any site.

```bash
cd tools/reclaim
npm rebuild re2            # once, to match your Node version
ATTESTOR_BASE_URL=ws://localhost:8001/ws node gen.mjs > proof.json
```

The proof's signature recovers our attestor address through the exact
`reclaim.move` algorithm (verified end to end).

## Submit a real proof on-chain

`demos/prediction-market/submit_real_zktls.py` creates a market that pins our
attestor, generates a fresh proof during the evidence window, and submits it.
Verified on testnet: a live BTC price was admitted on-chain
(market `0x2654205963c555787be91fda0bbfda187edba3f6b02caa2d5b91cdb521368bd8`).

```bash
python3 demos/prediction-market/submit_real_zktls.py
# → ✅ ADMITTED ON-CHAIN
```

The cloned `attestor-core/` and its downloaded zk files are gitignored; clone
and install them locally with the steps above.
