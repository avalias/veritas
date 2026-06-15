#!/usr/bin/env python3
"""Create one market that goes the full distance — trade buffer → evidence
→ resolve YES — so the live grid shows a completed lifecycle. Appends it to
markets.json as a resolved showcase. Takes ~2.5 min (short windows).
"""
import json, subprocess, sys, time
from eth_account import Account
from eth_account.messages import encode_defunct
from eth_hash.auto import keccak
ROOT=subprocess.run(["git","rev-parse","--show-toplevel"],capture_output=True,text=True).stdout.strip()
CFG=json.load(open(f"{ROOT}/demos/prediction-market/web/config.json")); MK=json.load(open(f"{ROOT}/demos/prediction-market/web/markets.json"))
PKG,CLOCK,GB=CFG["package"],"0x6","400000000"
ATT=["0x17c5185167401ed00cf5f5b2fc97d9bbfdb7d025","0xda11c9da04ab02c4af9374b27a5e727944d3e1dd","0x2222222222222222222222222222222222222222"]
def sh(a):
    r=subprocess.run(a,capture_output=True,text=True)
    if r.returncode!=0: print("FAIL",r.stderr[-400:],r.stdout[-200:],file=sys.stderr); sys.exit(1)
    return r
def gas(): return max(json.loads(sh(["sui","client","gas","--json"]).stdout),key=lambda c:int(c["mistBalance"]))["gasCoinId"]
def split(n):
    o=json.loads(sh(["sui","client","split-coin","--coin-id",gas(),"--amounts",str(n),"--gas-budget",GB,"--json"]).stdout)
    return next(c["objectId"] for c in o["objectChanges"] if c["type"]=="created" and "Coin" in c["objectType"])

now=int(time.time()*1000); ra=now+90_000; win=90_000
q="Did Starship reach orbit? (resolved showcase)"
seed=split(500_000_000)
o=json.loads(sh(["sui","client","call","--package",PKG,"--module","market","--function","create_market_entry","--gas-budget",GB,"--json",
  "--args","0x"+q.encode().hex(),"0x"+"a7"*32,"12",json.dumps(ATT),"[3,3,3]","[0,1,2]","1","0",str(ra),str(win),"100",seed,CLOCK]).stdout)
mid=next(c["objectId"] for c in o["objectChanges"] if c["type"]=="created" and c.get("objectType","").endswith("::market::Market"))
print("market",mid,"— waiting for evidence window…")

while int(time.time()*1000) < ra+3000: time.sleep(3)
# submit the real zkTLS proof inside the window
acct=Account.from_key(b"\x42"*32); provider="http"
parameters=json.dumps({"url":"https://www.reuters.com","method":"GET","responseMatches":[{"type":"contains","value":"Starship reached orbit"}]},separators=(",",":"))
context=json.dumps({"extractedParameters":{"headline":"Starship reached orbit"},"providerHash":"0xshow"},separators=(",",":"))
owner=acct.address.lower(); ts=(ra+15_000)//1000
identifier="0x"+keccak(f"{provider}\n{parameters}\n{context}".encode()).hex()
sig=Account.sign_message(encode_defunct(text=f"{identifier}\n{owner}\n{ts}\n1"),acct.key)
sh(["sui","client","call","--package",PKG,"--module","market","--function","submit_web_proof","--gas-budget",GB,"--json","--args",
  mid,"0","1","0x"+provider.encode().hex(),"0x"+parameters.encode().hex(),"0x"+context.encode().hex(),"0x"+owner.encode().hex(),str(ts),"1","0x"+bytes(sig.signature).hex(),CLOCK])
print("zkTLS proof admitted — waiting for window to close…")

while int(time.time()*1000) < ra+win+2000: time.sleep(3)
sh(["sui","client","call","--package",PKG,"--module","market","--function","resolve","--gas-budget",GB,"--json","--args",mid,CLOCK])
f=json.loads(sh(["sui","client","object",mid,"--json"]).stdout)["content"]["fields"]
print("RESOLVED — outcome:",["OPEN","YES","NO","VOID"][int(f["outcome"])])

MK["markets"].append({"id":mid,"emoji":"🛰️","question":"Did Starship reach orbit?","category":"Resolved · zkTLS","phase":"resolved","price_yes":100})
json.dump(MK,open(f"{ROOT}/demos/prediction-market/web/markets.json","w"),indent=2)
print("appended resolved showcase to markets.json")
