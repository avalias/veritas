#!/usr/bin/env python3
"""End-to-end proof that a REAL zkTLS proof from our self-hosted attestor is
admitted on-chain. Creates a market that pins our attestor, generates a fresh
proof of a live website during the evidence window, and submits it.

Prereq: the attestor is running (tools/zktls/attestor-core, port 8001) and the
client deps are installed (tools/reclaim). Run on the configured network.
"""
import json, subprocess, time, os
import judge_lib as L

ATTESTOR = "ws://localhost:8001/ws"
TARGET = "https://api.coinbase.com/v2/prices/BTC-USD/spot"
REGEX = '"amount":"(?<value>[0-9.]+)"'


def gen_proof():
    """Run the Node client to produce a fresh real proof; return the claim dict."""
    cmd = (f'source ~/.nvm/nvm.sh >/dev/null 2>&1; nvm use 22 >/dev/null 2>&1; '
           f'cd "{L.ROOT}/tools/reclaim"; '
           f'ATTESTOR_BASE_URL={ATTESTOR} TARGET_URL={TARGET} TARGET_REGEX=\'{REGEX}\' node gen.mjs')
    out = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True)
    # the client interleaves pino log lines; the proof is the pretty-printed object
    lines = out.stdout.splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip() == "{")
    return json.loads("\n".join(lines[start:]))


def hx(s):
    return "0x" + s.encode().hex()


def main():
    now = L.now_ms()
    ra = now + 12_000
    win = 2 * 3600 * 1000
    print("creating a market that pins our attestor…")
    mid = L.create_market("Did Coinbase serve this BTC price (proven by self-hosted zkTLS)?",
                          ra, win, k=1, seed_mist=70_000_000)
    print("  market:", mid)
    print("waiting for the evidence window to open…")
    while L.now_ms() < ra + 2_000:
        time.sleep(1)

    print("generating a fresh real zkTLS proof (live MPC-TLS to Coinbase)…")
    d = gen_proof()
    cd = d["claim"]
    sigd = d["signatures"]["claimSignature"]
    sig = bytes(sigd[str(i)] for i in range(len(sigd)))
    price = json.loads(cd["context"]).get("extractedParameters", {}).get("value")
    print(f"  proven live BTC price: {price}  (attestor-signed, ts {cd['timestampS']})")

    proof = {"attestor_idx": 0, "claim": 1,
             "provider": hx(cd["provider"]), "parameters": hx(cd["parameters"]),
             "context": hx(cd["context"]), "owner": hx(cd["owner"]),
             "timestamp_s": int(cd["timestampS"]), "epoch": int(cd["epoch"]),
             "signature": "0x" + sig.hex()}
    print("submitting it on-chain (verified by native ecrecover in reclaim.move)…")
    L.submit_proof(mid, proof)

    f = L.object_fields(mid)
    n = len(f.get("evidence", []))
    print(f"\nRESULT: {'✅ ADMITTED ON-CHAIN' if n >= 1 else '❌ not admitted'} — market has {n} evidence item(s)")
    print(f"  market {mid}")


if __name__ == "__main__":
    main()
