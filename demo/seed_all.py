#!/usr/bin/env python3
"""(Re)seed the curated devnet markets with HEALTHY liquidity and varied
opening prices so the demo feels like a live market, plus a fresh evidence
market with an unused zkTLS proof. Writes demo/web/markets.json.
"""
import json, subprocess, sys, time
from eth_account import Account
from eth_account.messages import encode_defunct
from eth_hash.auto import keccak

ROOT = subprocess.run(["git","rev-parse","--show-toplevel"],capture_output=True,text=True).stdout.strip()
CFG = json.load(open(f"{ROOT}/demo/web/config.json"))
PKG, CLOCK, GB = CFG["package"], "0x6", "400000000"
ATTESTORS = ["0x17c5185167401ed00cf5f5b2fc97d9bbfdb7d025",
             "0xda11c9da04ab02c4af9374b27a5e727944d3e1dd",
             "0x2222222222222222222222222222222222222222"]
JUDGE_ROOT = "0x" + "a7"*32
DAY = 86400*1000
LIQ = 500_000_000   # 0.5 SUI liquidity per market → gentle price impact

def sh(a):
    r=subprocess.run(a,capture_output=True,text=True)
    if r.returncode!=0: print("FAIL",a[:6],"\n",r.stderr[-500:],"\n",r.stdout[-300:],file=sys.stderr); sys.exit(1)
    return r
def gas(): return max(json.loads(sh(["sui","client","gas","--json"]).stdout),key=lambda c:int(c["mistBalance"]))["gasCoinId"]
def split(n):
    o=json.loads(sh(["sui","client","split-coin","--coin-id",gas(),"--amounts",str(n),"--gas-budget",GB,"--json"]).stdout)
    return next(c["objectId"] for c in o["objectChanges"] if c["type"]=="created" and "Coin" in c["objectType"])
def create(q,ra,win,k):
    seed=split(LIQ)
    o=json.loads(sh(["sui","client","call","--package",PKG,"--module","market","--function","create_market_entry",
      "--gas-budget",GB,"--json","--args","0x"+q.encode().hex(),JUDGE_ROOT,"12",
      json.dumps(ATTESTORS),"[3,3,3]","[0,1,2]",str(k),"0",str(ra),str(win),"100",seed,CLOCK]).stdout)
    return next(c["objectId"] for c in o["objectChanges"] if c["type"]=="created" and c.get("objectType","").endswith("::market::Market"))
def nudge(mid,which,amt):
    coin=split(amt)
    sh(["sui","client","call","--package",PKG,"--module","market","--function","buy_"+which,
        "--gas-budget",GB,"--json","--args",mid,coin,CLOCK])
def price(mid):
    f=json.loads(sh(["sui","client","object",mid,"--json"]).stdout)["content"]["fields"]
    ry,rn=int(f["reserve_yes"]),int(f["reserve_no"]); return round(rn/(ry+rn)*100)

now=int(time.time()*1000)
# (emoji, question, category, nudge side, nudge amount) — to vary opening odds
SPEC=[
 ("🚀","Will SpaceX's Starship reach orbit before October 1, 2026?","Space","yes",170_000_000),
 ("🤖","Will OpenAI release a model it calls GPT-6 before 2027?","AI","no",120_000_000),
 ("📈","Will Bitcoin trade above $150,000 before 2027?","Markets","yes",60_000_000),
 ("🏛️","Will the US Federal Reserve cut rates at its next meeting?","Macro","yes",230_000_000),
]
markets=[]
for emoji,q,cat,side,amt in SPEC:
    mid=create(q, now+30*DAY, 7*DAY, 2)
    nudge(mid,side,amt)
    p=price(mid)
    markets.append({"id":mid,"emoji":emoji,"question":q,"category":cat,"phase":"trading","price_yes":p})
    print(f"{emoji} {mid[:12]} YES {p}%")

# evidence market (short trading buffer) + fresh proof
ra=now+90_000; win=3600_000
mid=create("Did Starship reach orbit, as reported by the wires?",ra,win,1)
acct=Account.from_key(b"\x42"*32)
provider="http"
parameters=json.dumps({"url":"https://www.bbc.com/news","method":"GET","responseMatches":[{"type":"contains","value":"Starship reaches orbit"}]},separators=(",",":"))
context=json.dumps({"extractedParameters":{"headline":"SpaceX Starship reaches orbit"},"providerHash":"0xbbc-live"},separators=(",",":"))
owner=acct.address.lower(); ts=(ra+180_000)//1000
identifier="0x"+keccak(f"{provider}\n{parameters}\n{context}".encode()).hex()
sig=Account.sign_message(encode_defunct(text=f"{identifier}\n{owner}\n{ts}\n1"),acct.key)
ev={"id":mid,"emoji":"🛰️","question":"Did Starship reach orbit, as reported by the wires?","category":"Space · evidence window",
    "phase":"evidence","resolve_after":ra,"window":win,
    "proof":{"attestor_idx":0,"claim":1,"provider":"0x"+provider.encode().hex(),"parameters":"0x"+parameters.encode().hex(),
      "context":"0x"+context.encode().hex(),"owner":"0x"+owner.encode().hex(),"timestamp_s":ts,"epoch":1,
      "signature":"0x"+bytes(sig.signature).hex(),"source":"bbc.com","headline":"SpaceX Starship reaches orbit"}}
print("🛰️ evidence",mid[:12],"opens +90s")

json.dump({"package":PKG,"network":"devnet","rpc":CFG["rpc"],"clock":CLOCK,
  "attestors":ATTESTORS,"markets":markets,"evidence_market":ev},
  open(f"{ROOT}/demo/web/markets.json","w"),indent=2)
print("wrote markets.json")
