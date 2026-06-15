#!/usr/bin/env python3
"""Seed a curated set of example markets on the deployed devnet package, so
the dApp shows real, on-chain markets people can trade and resolve.

Creates 4 open (trading) markets + 1 in its evidence window (for the live
zkTLS submission). Writes demos/prediction-market/web/markets.json for the front-end.

Run after deploy.sh devnet (active env = devnet, address funded).
"""
import json
import subprocess
import sys
import time

ROOT = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True).stdout.strip()
CFG = json.load(open(f"{ROOT}/demos/prediction-market/web/config.json"))
PKG = CFG["package"]
CLOCK = "0x6"
GB = "300000000"

# 3 independent zkTLS attestors (distinct 20-byte addresses). The first is
# our real test attestor (0x17c5…) so the live evidence proof verifies; the
# others stand in for independent witness networks.
ATTESTORS = [
    "0x17c5185167401ed00cf5f5b2fc97d9bbfdb7d025",
    "0xda11c9da04ab02c4af9374b27a5e727944d3e1dd",  # the real Reclaim attestor
    "0x2222222222222222222222222222222222222222",
]
KEYS_JSON = json.dumps(ATTESTORS)
SCHEMES = "[3,3,3]"   # SCHEME_RECLAIM
GROUPS = "[0,1,2]"    # three independent trust groups
JUDGE_ROOT = "0x" + "a7" * 32   # committed judge program root (placeholder hash)

DAY = 86400 * 1000

# (emoji, question, category) — curated to be exciting + news-resolvable.
TRADING = [
    ("🚀", "Will SpaceX's Starship reach orbit before October 1, 2026?", "Space"),
    ("🤖", "Will OpenAI release a model it calls GPT-6 before 2027?", "AI"),
    ("📈", "Will Bitcoin trade above $150,000 before 2027?", "Markets"),
    ("🏛️", "Will the US Federal Reserve cut rates at its next meeting?", "Macro"),
]
# the market used for the live evidence demo (already in its evidence window)
EVIDENCE = ("🛰️", "Did Starship reach orbit, as reported by the wires?", "Space · resolving now")


def sh(args, check=True):
    r = subprocess.run(args, capture_output=True, text=True)
    if check and r.returncode != 0:
        print("FAILED:", " ".join(args[:6]), "\n", r.stderr[-800:], file=sys.stderr)
        sys.exit(1)
    return r


def gas_coin():
    coins = json.loads(sh(["sui", "client", "gas", "--json"]).stdout)
    return max(coins, key=lambda c: int(c["mistBalance"]))["gasCoinId"]


def split(amount):
    out = json.loads(sh(["sui", "client", "split-coin", "--coin-id", gas_coin(),
                         "--amounts", str(amount), "--gas-budget", GB, "--json"]).stdout)
    for c in out["objectChanges"]:
        if c["type"] == "created" and "Coin" in c["objectType"]:
            return c["objectId"]
    sys.exit("split failed")


def create(question, resolve_after_ms, window_ms, k):
    seed = split(20_000_000)  # 0.02 SUI liquidity
    out = json.loads(sh([
        "sui", "client", "call", "--package", PKG, "--module", "market",
        "--function", "create_market_entry", "--gas-budget", GB, "--json",
        "--args",
        "0x" + question.encode().hex(), JUDGE_ROOT, "12",
        KEYS_JSON, SCHEMES, GROUPS, str(k), "0",
        str(resolve_after_ms), str(window_ms), "100", seed, CLOCK,
    ]).stdout)
    for c in out["objectChanges"]:
        if c["type"] == "created" and c.get("objectType", "").endswith("::market::Market"):
            return c["objectId"]
    sys.exit("market id not found")


def main():
    now = int(time.time() * 1000)
    markets = []
    for emoji, q, cat in TRADING:
        mid = create(q, now + 30 * DAY, 7 * DAY, 2)
        markets.append({"id": mid, "emoji": emoji, "question": q, "category": cat,
                        "phase": "trading", "resolve_after": now + 30 * DAY})
        print("trading:", emoji, mid)

    # evidence-phase market: opened in the recent past, window covers now
    emoji, q, cat = EVIDENCE
    mid = create(q, now - DAY, 3 * DAY, 1)  # k=1 so one zkTLS proof resolves it
    markets.append({"id": mid, "emoji": emoji, "question": q, "category": cat,
                    "phase": "evidence", "resolve_after": now - DAY})
    print("evidence:", emoji, mid)

    json.dump({"package": PKG, "network": "devnet", "rpc": CFG["rpc"],
               "clock": CLOCK, "attestors": ATTESTORS, "markets": markets},
              open(f"{ROOT}/demos/prediction-market/web/markets.json", "w"), indent=2)
    print(f"\nwrote demos/prediction-market/web/markets.json with {len(markets)} markets")


if __name__ == "__main__":
    main()
