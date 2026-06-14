#!/usr/bin/env python3
"""market_e2e.py — the whole product, live, on a Sui localnet.

Drives the REAL `sui` CLI against a running localnet and walks a market
through its entire life: create (with committed rules) -> trade YES/NO ->
provenance-gated evidence (real ed25519 Web Credentials signed here, in
this script) -> resolve by the committed rule -> redeem. Prints a clean,
human narrative at each step.

Prereqs:
    sui start --with-faucet --force-regenesis      # in another terminal
    python3 -m pip install cryptography            # for ed25519 signing

Run:
    python3 dispute/demo/market_e2e.py

This is intentionally a transparent script (not a compiled binary): you can
read exactly what every transaction does and re-sign every credential.
"""
import hashlib
import json
import subprocess
import sys
import time

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

PKG_DIR = subprocess.run(
    ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
).stdout.strip() + "/dispute"

GAS_BUDGET = "200000000"  # 0.2 SUI, generous for a demo


# ----------------------------------------------------------------------------
# tiny sui CLI helpers
# ----------------------------------------------------------------------------
def sh(args, check=True):
    r = subprocess.run(args, capture_output=True, text=True)
    if check and r.returncode != 0:
        print("CMD FAILED:", " ".join(args), file=sys.stderr)
        print(r.stdout[-2000:], r.stderr[-2000:], file=sys.stderr)
        sys.exit(1)
    return r


def sui_json(args):
    r = sh(["sui", "client"] + args + ["--json"])
    return json.loads(r.stdout)


def active_address():
    return sh(["sui", "client", "active-address"]).stdout.strip()


def split_coin(amount):
    """Return a fresh coin object id holding `amount` MIST."""
    out = sui_json(["split-coin", "--coin-id", gas_coin(), "--amounts", str(amount),
                    "--gas-budget", GAS_BUDGET])
    for c in out.get("objectChanges", []):
        if c.get("type") == "created" and "Coin" in c.get("objectType", ""):
            return c["objectId"]
    sys.exit("could not split a coin")


def gas_coin():
    coins = sui_json(["gas"])
    # pick the largest gas coin so splits never drain the first one
    best = max(coins, key=lambda c: int(c["mistBalance"]) if isinstance(c, dict) and "mistBalance" in c else 0)
    return best["gasCoinId"] if "gasCoinId" in best else best["coinObjectId"]


def call(pkg, module, function, args):
    cmd = ["call", "--package", pkg, "--module", module, "--function", function,
           "--gas-budget", GAS_BUDGET]
    if args:
        cmd += ["--args"] + args  # single --args followed by every value (SuiJSON)
    resp = sui_json(cmd)
    status = resp.get("effects", {}).get("status", {}).get("status")
    if status != "success":
        sys.exit(f"{function} failed: {resp.get('effects', {}).get('status')}")
    return resp


def created_of_type(resp, type_frag):
    for c in resp.get("objectChanges", []):
        if c.get("type") == "created" and type_frag in c.get("objectType", ""):
            return c["objectId"]
    return None


def now_ms():
    return int(time.time() * 1000)


# Sui system Clock is a shared object at 0x6
CLOCK = "0x6"


# ----------------------------------------------------------------------------
# the demo
# ----------------------------------------------------------------------------
def banner(title):
    print("\n" + "=" * 68)
    print("  " + title)
    print("=" * 68)


def main():
    me = active_address()
    print(f"resolver / trader address: {me}")

    banner("PUBLISH  ·  the dispute+market package")
    pub = sui_json(["publish", "--gas-budget", "500000000", PKG_DIR])
    pkg = None
    for c in pub.get("objectChanges", []):
        if c.get("type") == "published":
            pkg = c["packageId"]
    print(f"package: {pkg}")

    # two independent issuers (trust groups 0 and 1): e.g. "AP" and "Reuters".
    issuers = []
    for seed_byte in (0xA1, 0xB2):
        sk = Ed25519PrivateKey.from_private_bytes(bytes([seed_byte]) * 32)
        pk = sk.public_key().public_bytes(serialization.Encoding.Raw,
                                           serialization.PublicFormat.Raw)
        issuers.append((sk, pk))
    # SuiJSON encodes vector<vector<u8>> as a JSON array of 0x-hex strings.
    keys_json = json.dumps(["0x" + pk.hex() for _, pk in issuers])

    # short windows so the demo runs in seconds
    t0 = now_ms()
    resolve_after = t0 + 6000      # 6s of trading
    window = 8000                  # 8s evidence window

    banner("CREATE  ·  a market with rules fixed up front")
    print('  Q: "Did the agency confirm event E before the deadline?"')
    print(f"  rule: OCCURRENCE, k=2 independent issuer groups; window fixed at creation")
    seed = split_coin(50_000_000)  # 0.05 SUI liquidity
    resp = call(pkg, "market", "create_market_entry", [
        "Did the agency confirm event E before the deadline?",  # question
        "0xaabbccdd",        # judge program root (placeholder)
        "12",                # judge depth
        keys_json,           # issuer_keys  vector<vector<u8>>
        "[0,0]",             # issuer_schemes: both ed25519 (scheme 0)
        "[0,1]",             # issuer_groups
        "2",                 # k
        "0",                 # burden = OCCURRENCE
        str(resolve_after),  # resolve_after_ms
        str(window),         # evidence_window_ms
        "100",               # fee_bps (1%)
        seed,                # seed coin
        CLOCK,
    ])
    market = created_of_type(resp, "::market::Market")
    print(f"  market: {market}")

    banner("TRADE  ·  price discovery on a solvent CPMM")
    pay_yes = split_coin(10_000_000)
    call(pkg, "market", "buy_yes", [market, pay_yes, CLOCK])
    pay_no = split_coin(4_000_000)
    call(pkg, "market", "buy_no", [market, pay_no, CLOCK])
    print("  bought YES (10M) and NO (4M); YES is now the favored side")

    banner("WAIT  ·  trading closes, evidence window opens")
    while now_ms() < resolve_after + 200:
        time.sleep(0.5)
    print("  evidence window is open")

    banner("EVIDENCE  ·  provenance-gated Web Credentials only")
    # both independent issuers sign a YES confirmation (2 groups => meets k=2)
    for idx, (sk, pk) in enumerate(issuers):
        claim = 1  # CLAIM_YES
        content_hash = hashlib.sha256(f"article-from-issuer-{idx}".encode()).digest()
        signed_ms = resolve_after + 1000 + idx * 100
        market_bytes = bytes.fromhex(market[2:])  # 32-byte address
        preimage = market_bytes + bytes([claim]) + content_hash + signed_ms.to_bytes(8, "little")
        msg = hashlib.blake2b(preimage, digest_size=32).digest()
        sig = sk.sign(msg)
        call(pkg, "market", "submit_evidence", [
            market, str(idx), str(claim),
            "0x" + content_hash.hex(), str(signed_ms), "0x" + sig.hex(), CLOCK,
        ])
        print(f"  issuer {idx} (group {idx}) signed YES  ->  admitted (ed25519 verified on-chain)")

    banner("WAIT  ·  evidence window closes")
    while now_ms() < resolve_after + window + 200:
        time.sleep(0.5)

    banner("RESOLVE  ·  pure function of the signed evidence")
    call(pkg, "market", "resolve", [market, CLOCK])
    fields = sui_json(["object", market])["content"]["fields"]
    outcome = int(fields["outcome"])
    names = {0: "OPEN", 1: "YES", 2: "NO", 3: "UNRESOLVED"}
    print(f"  2 independent groups confirmed YES  ->  outcome = {names[outcome]}")

    banner("REDEEM  ·  winners paid 1 SUI per share from collateral")
    call(pkg, "market", "redeem_to_sender", [market])
    print("  YES shares redeemed at 1:1. NO shares are worth 0.")
    print("\nDone. No human decided this outcome — the signed evidence did.\n")


if __name__ == "__main__":
    main()
