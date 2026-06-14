#!/usr/bin/env bash
# Publish the dispute+market package to a Sui network and print the IDs the
# demo / clients need. Works for localnet, testnet, or mainnet.
#
#   ./deploy.sh local      # needs `sui start --with-faucet` running
#   ./deploy.sh testnet    # fund first: https://faucet.sui.io  (or the HTTP API)
#   ./deploy.sh mainnet    # uses your funded mainnet address — real SUI
#
# The package is mainnet-ready: 70 Move tests green, hardened against an
# adversarial review. Output: PACKAGE_ID for the front-end / market_e2e.py.
set -euo pipefail
ENV="${1:-local}"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "→ switching to env: $ENV"
sui client switch --env "$ENV" >/dev/null 2>&1 || {
  echo "env '$ENV' not configured. add it, e.g.:"
  echo "  sui client new-env --alias testnet --rpc https://fullnode.testnet.sui.io:443"
  exit 1
}

ADDR="$(sui client active-address)"
echo "→ active address: $ADDR"

BAL="$(sui client gas --json 2>/dev/null | python3 -c 'import json,sys;d=json.load(sys.stdin);print(sum(int(c["mistBalance"]) for c in d))' 2>/dev/null || echo 0)"
if [ "$BAL" -lt 200000000 ]; then
  echo "✗ insufficient gas ($BAL MIST). fund $ADDR:"
  echo "    testnet: https://faucet.sui.io/?address=$ADDR"
  echo "    or:      curl -s -X POST https://faucet.testnet.sui.io/v2/gas -H 'Content-Type: application/json' -d '{\"FixedAmountRequest\":{\"recipient\":\"$ADDR\"}}'"
  exit 1
fi

echo "→ publishing…"
OUT="$(sui client publish --gas-budget 500000000 --json "$HERE")"
PKG="$(echo "$OUT" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(next(c["packageId"] for c in d["objectChanges"] if c["type"]=="published"))')"
echo
echo "════════════════════════════════════════════════════════════"
echo "  PACKAGE_ID = $PKG"
echo "  network    = $ENV"
echo "════════════════════════════════════════════════════════════"
echo "  modules: market · dispute · credential · reclaim · tee"
echo "  next: point demo/web or market_e2e.py at PACKAGE_ID."
