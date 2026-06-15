#!/usr/bin/env python3
"""Re-create just the evidence market with a LONG (30-day) evidence window so
it stays in the evidence phase — the judge→submit flow is demoable — and a
fresh unused zkTLS proof. Preserves all other markets in markets.json.
"""
import json, subprocess, sys, time
from eth_account import Account
from eth_account.messages import encode_defunct
from eth_hash.auto import keccak
ROOT=subprocess.run(["git","rev-parse","--show-toplevel"],capture_output=True,text=True).stdout.strip()
CFG=json.load(open(f"{ROOT}/demo/web/config.json")); MK=json.load(open(f"{ROOT}/demo/web/markets.json"))
PKG,CLOCK,GB=CFG["package"],"0x6","400000000"
ATT=["0x17c5185167401ed00cf5f5b2fc97d9bbfdb7d025","0xda11c9da04ab02c4af9374b27a5e727944d3e1dd","0x2222222222222222222222222222222222222222"]
def sh(a):
    r=subprocess.run(a,capture_output=True,text=True)
    if r.returncode!=0: print("FAIL",r.stderr[-400:],file=sys.stderr); sys.exit(1)
    return r
def gas(): return max(json.loads(sh(["sui","client","gas","--json"]).stdout),key=lambda c:int(c["mistBalance"]))["gasCoinId"]
def split(n):
    o=json.loads(sh(["sui","client","split-coin","--coin-id",gas(),"--amounts",str(n),"--gas-budget",GB,"--json"]).stdout)
    return next(c["objectId"] for c in o["objectChanges"] if c["type"]=="created" and "Coin" in c["objectType"])
now=int(time.time()*1000); ra=now+60_000; win=30*86400*1000
seed=split(500_000_000)
q="Did Starship reach orbit, as reported by the wires?"
o=json.loads(sh(["sui","client","call","--package",PKG,"--module","market","--function","create_market_entry","--gas-budget",GB,"--json",
  "--args","0x"+q.encode().hex(),"0x"+"a7"*32,"12",json.dumps(ATT),"[3,3,3]","[0,1,2]","1","0",str(ra),str(win),"100",seed,CLOCK]).stdout)
mid=next(c["objectId"] for c in o["objectChanges"] if c["type"]=="created" and c.get("objectType","").endswith("::market::Market"))
acct=Account.from_key(b"\x42"*32); provider="http"
parameters=json.dumps({"url":"https://www.bbc.com/news","method":"GET","responseMatches":[{"type":"contains","value":"Starship reaches orbit"}]},separators=(",",":"))
context=json.dumps({"extractedParameters":{"headline":"SpaceX Starship reaches orbit"},"providerHash":"0xbbc-live"},separators=(",",":"))
owner=acct.address.lower(); ts=(ra+180_000)//1000
identifier="0x"+keccak(f"{provider}\n{parameters}\n{context}".encode()).hex()
sig=Account.sign_message(encode_defunct(text=f"{identifier}\n{owner}\n{ts}\n1"),acct.key)
MK["evidence_market"]={"id":mid,"emoji":"🛰️","question":q,"category":"Space · evidence window","phase":"evidence","resolve_after":ra,"window":win,
  "proof":{"attestor_idx":0,"claim":1,"provider":"0x"+provider.encode().hex(),"parameters":"0x"+parameters.encode().hex(),
    "context":"0x"+context.encode().hex(),"owner":"0x"+owner.encode().hex(),"timestamp_s":ts,"epoch":1,
    "signature":"0x"+bytes(sig.signature).hex(),"source":"bbc.com","headline":"SpaceX Starship reaches orbit"}}
json.dump(MK,open(f"{ROOT}/demo/web/markets.json","w"),indent=2)
print("re-seeded evidence market",mid,"(30-day window)")
