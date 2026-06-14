#!/usr/bin/env python3
"""Create the evidence-phase market (short trading buffer so it enters its
evidence window quickly), generate a matching real zkTLS proof signed by
the pinned attestor, and write demo/web/markets.json with all markets.
"""
import json, subprocess, sys, time
from eth_account import Account
from eth_account.messages import encode_defunct
from eth_hash.auto import keccak

ROOT = subprocess.run(["git","rev-parse","--show-toplevel"],capture_output=True,text=True).stdout.strip()
CFG = json.load(open(f"{ROOT}/demo/web/config.json"))
PKG, CLOCK, GB = CFG["package"], "0x6", "300000000"
ATTESTORS = ["0x17c5185167401ed00cf5f5b2fc97d9bbfdb7d025",
             "0xda11c9da04ab02c4af9374b27a5e727944d3e1dd",
             "0x2222222222222222222222222222222222222222"]
JUDGE_ROOT = "0x" + "a7"*32

TRADING = [
  {"id":"0x9a0326ad75cb8814fcf67e5875a19bb9f6471ed5bb6c8524cee3813e56b98296","emoji":"🚀","question":"Will SpaceX's Starship reach orbit before October 1, 2026?","category":"Space"},
  {"id":"0x233ea93c4fa03d2d08d7462d60540ed736aaa267a650e8011aada20ea346487a","emoji":"🤖","question":"Will OpenAI release a model it calls GPT-6 before 2027?","category":"AI"},
  {"id":"0x1a0be332c443f1d9015b4fff69260a713745488097a3df9f3b769462067911dc","emoji":"📈","question":"Will Bitcoin trade above $150,000 before 2027?","category":"Markets"},
  {"id":"0x1a7e04b8c20a4e01e108d91d8b0361ed11c28c10e14df5be6c6e6b8b50bb5a83","emoji":"🏛️","question":"Will the US Federal Reserve cut rates at its next meeting?","category":"Macro"},
]

def sh(a):
    r=subprocess.run(a,capture_output=True,text=True)
    if r.returncode!=0: print("FAIL",a[:6],r.stderr[-600:],file=sys.stderr); sys.exit(1)
    return r
def gas(): return max(json.loads(sh(["sui","client","gas","--json"]).stdout),key=lambda c:int(c["mistBalance"]))["gasCoinId"]
def split(n):
    o=json.loads(sh(["sui","client","split-coin","--coin-id",gas(),"--amounts",str(n),"--gas-budget",GB,"--json"]).stdout)
    return next(c["objectId"] for c in o["objectChanges"] if c["type"]=="created" and "Coin" in c["objectType"])

now=int(time.time()*1000)
resolve_after = now + 90_000          # 90s trading buffer, then evidence opens
window = 3600_000                     # 1 hour evidence window
seed=split(20_000_000)
q="Did Starship reach orbit, as reported by the wires?"
out=json.loads(sh(["sui","client","call","--package",PKG,"--module","market","--function","create_market_entry",
  "--gas-budget",GB,"--json","--args","0x"+q.encode().hex(),JUDGE_ROOT,"12",
  json.dumps(ATTESTORS),"[3,3,3]","[0,1,2]","1","0",str(resolve_after),str(window),"100",seed,CLOCK]).stdout)
ev_id=next(c["objectId"] for c in out["objectChanges"] if c["type"]=="created" and c.get("objectType","").endswith("::market::Market"))
print("evidence market:",ev_id)

# real zkTLS proof, witnessed inside the evidence window, signed by attestor[0]
acct=Account.from_key(b"\x42"*32)        # -> 0x17c5… == ATTESTORS[0]
provider="http"
parameters=json.dumps({"url":"https://www.reuters.com","method":"GET","responseMatches":[{"type":"contains","value":"Starship reached orbit"}]},separators=(",",":"))
context=json.dumps({"extractedParameters":{"headline":"Starship reached orbit"},"providerHash":"0xreuters"},separators=(",",":"))
owner=acct.address.lower()
ts=(resolve_after+120_000)//1000          # 2 min into the evidence window
epoch=1
identifier="0x"+keccak(f"{provider}\n{parameters}\n{context}".encode()).hex()
sig=Account.sign_message(encode_defunct(text=f"{identifier}\n{owner}\n{ts}\n{epoch}"),acct.key)

evidence={"id":ev_id,"emoji":"🛰️","question":q,"category":"Space · evidence window",
  "phase":"evidence","resolve_after":resolve_after,"window":window,
  "proof":{"attestor_idx":0,"claim":1,
    "provider":"0x"+provider.encode().hex(),"parameters":"0x"+parameters.encode().hex(),
    "context":"0x"+context.encode().hex(),"owner":"0x"+owner.encode().hex(),
    "timestamp_s":ts,"epoch":epoch,"signature":"0x"+bytes(sig.signature).hex(),
    "source":"reuters.com","headline":"Starship reached orbit"}}

json.dump({"package":PKG,"network":"devnet","rpc":CFG["rpc"],"clock":CLOCK,
  "attestors":ATTESTORS,"markets":TRADING,"evidence_market":evidence},
  open(f"{ROOT}/demo/web/markets.json","w"),indent=2)
print("wrote markets.json; evidence opens at +90s, proof ts =",ts)
