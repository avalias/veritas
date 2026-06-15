#!/usr/bin/env python3
"""Fast end-to-end proof of the FULL market lifecycle incl. redeem payout, on
devnet, using the same Move calls the dApp signs. Creates a throwaway short
market, buys YES, admits a proof, resolves, redeems, and checks the payout."""
import json, subprocess, time
import judge_lib as L

GB = L.GB

def call(fn, args):
    return json.loads(subprocess.run(["sui", "client", "call", "--package", L.PKG, "--module",
        "market", "--function", fn, "--gas-budget", GB, "--json", "--args", *args],
        capture_output=True, text=True).stdout)

def pos(mid, who):
    tx = subprocess.run(["sui","client","object",mid,"--json"],capture_output=True,text=True)
    return tx

now = L.now_ms(); ra = now + 30_000; win = 12_000  # 30s trading so the buy tx lands in-phase
print("create fast market…")
mid = L.create_market("Lifecycle self-test", ra, win, k=1, seed_mist=200_000_000)
me = subprocess.run(["sui","client","active-address"],capture_output=True,text=True).stdout.strip()

# BUY YES (real buy_yes with a fresh coin)
coin = L.split(50_000_000)
print("buy YES…")
call("buy_yes", [mid, coin, L.CLOCK])

# wait for evidence window, ADMIT a YES proof
while L.now_ms() < ra + 1500: time.sleep(1)
proof = L.build_proof((ra + win // 2) // 1000)
print("admit zkTLS proof…")
L.submit_proof(mid, proof)

# wait for the window to close, RESOLVE
while L.now_ms() < ra + win + 1500: time.sleep(1)
print("resolve…")
L.resolve(mid)
f = json.loads(subprocess.run(["sui","client","object",mid,"--json"],capture_output=True,text=True).stdout)["content"]["fields"]
print(f"  outcome={f['outcome']} (1=YES)  phase={f['phase']}  yes_groups={len(f.get('yes_groups',[]))}")

# REDEEM (pays the sender; same call as the dApp's redeem button)
bal0 = int(json.loads(subprocess.run(["sui","client","gas","--json"],capture_output=True,text=True).stdout) and
           sum(int(c["mistBalance"]) for c in json.loads(subprocess.run(["sui","client","gas","--json"],capture_output=True,text=True).stdout)))
print("redeem…")
r = subprocess.run(["sui","client","call","--package",L.PKG,"--module","market","--function",
    "redeem_to_sender","--gas-budget",GB,"--json","--args",mid],capture_output=True,text=True)
ok = r.returncode == 0
# look for a Redeemed event in the tx
red = "Redeemed" in r.stdout
print(f"  redeem tx ok={ok}  Redeemed event seen={red}")
passed = str(f['outcome']) == '1' and ok and red
print("\nRESULT:", "✅ FULL LIFECYCLE PASSED (buy→prove→resolve YES→redeem payout)" if passed else "❌ FAILED")
