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
CFG = json.load(open(f"{ROOT}/demos/prediction-market/web/config.json"))
PKG, CLOCK, GB = CFG["package"], "0x6", str(CFG.get("gas_budget", 150_000_000))
# attestor[0] is the address of the demo signing key b"\x42"*32; [1],[2] add
# independent trust groups. Resolution here uses one YES group (k=1 occurrence).
# Three independent "sources" the demo controls end-to-end — each a distinct
# pinned attestor in its own trust group. Each proof is a real Reclaim-format
# zkTLS proof verified on-chain by native ecrecover; the only difference from a
# live fetch is that we pre-sign it (no fragile external call mid-demo).
SOURCES = [
    {"name": "BBC News", "seed": b"\x42" * 32, "domain": "bbc.com",
     "url": "https://www.bbc.com/news/science-environment",
     "headline": "BBC News: SpaceX's Starship completes its first full orbital flight"},
    {"name": "Reuters", "seed": b"\x43" * 32, "domain": "reuters.com",
     "url": "https://www.reuters.com/technology/space",
     "headline": "Reuters: SpaceX Starship reaches orbit for the first time"},
    {"name": "Associated Press", "seed": b"\x44" * 32, "domain": "apnews.com",
     "url": "https://apnews.com/hub/spacex",
     "headline": "AP: SpaceX Starship reaches orbit, company confirms"},
]
for _s in SOURCES:
    _s["acct"] = Account.from_key(_s["seed"])
    _s["addr"] = _s["acct"].address.lower()
ATT = [s["addr"] for s in SOURCES]
SIGNER = SOURCES[0]["acct"]  # back-compat


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


def build_source_proof(idx, timestamp_s, claim=1):
    """A real Reclaim-format zkTLS web proof from SOURCES[idx], signed by that
    pinned attestor exactly as reclaim::verify checks on-chain (keccak
    identifier + EIP-191 + ecrecover). Distinct URL per source ⇒ distinct
    content hash ⇒ counts as an independent confirmation."""
    s = SOURCES[idx]
    provider = "http"
    parameters = json.dumps({"url": s["url"], "method": "GET",
                             "responseMatches": [{"type": "contains", "value": "Starship reaches orbit"}]},
                            separators=(",", ":"))
    context = json.dumps({"extractedParameters": {"headline": s["headline"]},
                          "providerHash": "0x" + s["domain"]}, separators=(",", ":"))
    owner = s["addr"]
    identifier = "0x" + keccak(f"{provider}\n{parameters}\n{context}".encode()).hex()
    sig = Account.sign_message(encode_defunct(text=f"{identifier}\n{owner}\n{timestamp_s}\n1"), s["acct"].key)
    return {"attestor_idx": idx, "claim": claim, "provider": "0x" + provider.encode().hex(),
            "parameters": "0x" + parameters.encode().hex(), "context": "0x" + context.encode().hex(),
            "owner": "0x" + owner.encode().hex(), "timestamp_s": timestamp_s, "epoch": 1,
            "signature": "0x" + bytes(sig.signature).hex(), "source": s["name"],
            "domain": s["domain"], "headline": s["headline"]}


def build_proof(timestamp_s, **_kw):  # back-compat: the first source
    return build_source_proof(0, timestamp_s)


def all_source_proofs(timestamp_s, claim=1):
    return [build_source_proof(i, timestamp_s, claim) for i in range(len(SOURCES))]


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
    return json.load(open(f"{ROOT}/demos/prediction-market/web/markets.json"))


def save_markets(m):
    # atomic: write a temp file then rename, so the dApp (which polls this file
    # every 15s) never reads a half-written JSON during a re-stage.
    import os
    path = f"{ROOT}/demos/prediction-market/web/markets.json"
    tmp = path + ".tmp"
    json.dump(m, open(tmp, "w"), indent=2)
    os.replace(tmp, path)


def gas_total_sui():
    g = json.loads(sh(["sui", "client", "gas", "--json"]).stdout)
    return sum(int(c["mistBalance"]) for c in g) / 1e9


def object_fields(oid):
    r = subprocess.run(["sui", "client", "object", oid, "--json"], capture_output=True, text=True)
    try:
        return json.loads(r.stdout)["content"]["fields"]
    except Exception:
        return None


def fact_status():
    """Status of the current Fraud-Lab Fact: 1=CHALLENGED (armed), 4=REJECTED (convicted)."""
    try:
        d = json.load(open(f"{ROOT}/demos/prediction-market/web/dispute.json"))
        f = object_fields(d["fact"])
        return int(f["status"]) if f else None
    except Exception:
        return None


def two_addresses():
    out = subprocess.run(["sui", "client", "addresses", "--json"], capture_output=True, text=True).stdout
    try:
        return [a[1] if isinstance(a, list) else a for a in json.loads(out)["addresses"]]
    except Exception:
        return []


def stage_fraud():
    """Re-arm the Fraud Lab: stage a fresh CHALLENGED dispute (writes dispute.json)."""
    al = two_addresses()
    if len(al) < 2:
        print("  🔪 fraud: need two funded addresses to stage; skipping", file=sys.stderr)
        return False
    # the dispute lives in the opml verifier package, not the market package
    opml_pkg = CFG.get("opml_package", PKG)
    active = subprocess.run(["sui", "client", "active-address"], capture_output=True, text=True).stdout.strip()
    r = subprocess.run(["cargo", "run", "-q", "-p", "client", "--bin", "devnet_stage_dispute",
                        "--", opml_pkg, al[0], al[1]], capture_output=True, text=True, cwd=ROOT)
    # the staging bin switches the active address per call; restore it
    subprocess.run(["sui", "client", "switch", "--address", active], capture_output=True, text=True)
    ok = r.returncode == 0
    print("  🔪 Fraud Lab re-armed" if ok else f"  🔪 fraud staging FAILED: {r.stderr[-200:]}")
    return ok


def ensure_fraud():
    """Arm the Fraud Lab only if the current Fact is missing or already convicted."""
    if fact_status() == 1:
        print("  🔪 Fraud Lab already armed (Fact CHALLENGED).")
        return True
    return stage_fraud()
