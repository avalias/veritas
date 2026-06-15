#!/usr/bin/env python3
"""Shared helpers for staging judge-demo markets on Sui devnet.

A "live" market walks the full lifecycle in real time so a judge can personally
witness trade -> evidence -> AI-judge -> resolve -> redeem, each a real signed tx.
"""
import json, subprocess, sys, time
from eth_account import Account
from eth_account.messages import encode_defunct
from eth_hash.auto import keccak

ROOT = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True).stdout.strip()
CFG = json.load(open(f"{ROOT}/demo/web/config.json"))
PKG, CLOCK, GB = CFG["package"], "0x6", "400000000"
# attestor[0] is the address of the demo signing key b"\x42"*32; [1],[2] add
# independent trust groups. Resolution here uses one YES group (k=1 occurrence).
ATT = ["0x17c5185167401ed00cf5f5b2fc97d9bbfdb7d025",
       "0xda11c9da04ab02c4af9374b27a5e727944d3e1dd",
       "0x2222222222222222222222222222222222222222"]
SIGNER = Account.from_key(b"\x42" * 32)
assert SIGNER.address.lower() == ATT[0], "signing key must match pinned attestor[0]"


def sh(args, check=True):
    r = subprocess.run(args, capture_output=True, text=True)
    if check and r.returncode != 0:
        print("FAIL:", " ".join(args[:4]), "\n", r.stderr[-500:], file=sys.stderr)
        sys.exit(1)
    return r


def gas_coin():
    g = json.loads(sh(["sui", "client", "gas", "--json"]).stdout)
    return max(g, key=lambda c: int(c["mistBalance"]))["gasCoinId"]


def split(mist):
    o = json.loads(sh(["sui", "client", "split-coin", "--coin-id", gas_coin(),
                       "--amounts", str(mist), "--gas-budget", GB, "--json"]).stdout)
    return next(c["objectId"] for c in o["objectChanges"]
               if c["type"] == "created" and "Coin" in c["objectType"])


def build_proof(timestamp_s,
                headline="SpaceX's Starship reached orbit on its latest flight, wire services reported",
                source="bbc.com", url="https://www.bbc.com/news",
                match="Starship reached orbit", claim=1):
    """A Reclaim-format zkTLS web proof, signed by the pinned attestor exactly
    as reclaim::verify checks on-chain (keccak identifier + EIP-191 + ecrecover)."""
    provider = "http"
    parameters = json.dumps({"url": url, "method": "GET",
                             "responseMatches": [{"type": "contains", "value": match}]},
                            separators=(",", ":"))
    context = json.dumps({"extractedParameters": {"headline": headline},
                          "providerHash": "0xbbc-live"}, separators=(",", ":"))
    owner = SIGNER.address.lower()
    identifier = "0x" + keccak(f"{provider}\n{parameters}\n{context}".encode()).hex()
    sig = Account.sign_message(encode_defunct(text=f"{identifier}\n{owner}\n{timestamp_s}\n1"), SIGNER.key)
    return {"attestor_idx": 0, "claim": claim, "provider": "0x" + provider.encode().hex(),
            "parameters": "0x" + parameters.encode().hex(), "context": "0x" + context.encode().hex(),
            "owner": "0x" + owner.encode().hex(), "timestamp_s": timestamp_s, "epoch": 1,
            "signature": "0x" + bytes(sig.signature).hex(), "source": source, "headline": headline}


def create_market(question, resolve_after_ms, window_ms, k=1, seed_mist=300_000_000, fee_bps=100):
    seed = split(seed_mist)
    o = json.loads(sh(["sui", "client", "call", "--package", PKG, "--module", "market",
        "--function", "create_market_entry", "--gas-budget", GB, "--json",
        "--args", "0x" + question.encode().hex(), "0x" + "a7" * 32, "12",
        json.dumps(ATT), "[3,3,3]", "[0,1,2]", str(k), "0",
        str(resolve_after_ms), str(window_ms), str(fee_bps), seed, CLOCK]).stdout)
    return next(c["objectId"] for c in o["objectChanges"]
                if c["type"] == "created" and c.get("objectType", "").endswith("::market::Market"))


def submit_proof(market_id, proof):
    """Operator-side submit (used to pre-stage a resolve-ready market)."""
    sh(["sui", "client", "call", "--package", PKG, "--module", "market",
        "--function", "submit_web_proof", "--gas-budget", GB, "--json",
        "--args", market_id, str(proof["attestor_idx"]), str(proof["claim"]),
        proof["provider"], proof["parameters"], proof["context"], proof["owner"],
        str(proof["timestamp_s"]), str(proof["epoch"]), proof["signature"], CLOCK])


def resolve(market_id):
    sh(["sui", "client", "call", "--package", PKG, "--module", "market",
        "--function", "resolve", "--gas-budget", GB, "--json",
        "--args", market_id, CLOCK])


def now_ms():
    return int(time.time() * 1000)


def load_markets():
    return json.load(open(f"{ROOT}/demo/web/markets.json"))


def save_markets(m):
    json.dump(m, open(f"{ROOT}/demo/web/markets.json", "w"), indent=2)
