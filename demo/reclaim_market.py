#!/usr/bin/env python3
"""Create a market that pins Reclaim's REAL attestor, so a user-generated
zkTLS proof (from the Reclaim app) verifies on-chain via reclaim::verify and
submit_web_proof. Adds it to markets.json and a `reclaim` config block.
"""
import json, subprocess, sys, time
ROOT=subprocess.run(["git","rev-parse","--show-toplevel"],capture_output=True,text=True).stdout.strip()
CFG=json.load(open(f"{ROOT}/demo/web/config.json")); MK=json.load(open(f"{ROOT}/demo/web/markets.json"))
PKG,CLOCK,GB=CFG["package"],"0x6","400000000"
# Reclaim's real attestor address (attestor-core); pin whichever your app uses.
RECLAIM_ATTESTOR="0xda11c9da04ab02c4af9374b27a5e727944d3e1dd"
def sh(a):
    r=subprocess.run(a,capture_output=True,text=True)
    if r.returncode!=0: print("FAIL",r.stderr[-400:],file=sys.stderr); sys.exit(1)
    return r
def gas(): return max(json.loads(sh(["sui","client","gas","--json"]).stdout),key=lambda c:int(c["mistBalance"]))["gasCoinId"]
def split(n):
    o=json.loads(sh(["sui","client","split-coin","--coin-id",gas(),"--amounts",str(n),"--gas-budget",GB,"--json"]).stdout)
    return next(c["objectId"] for c in o["objectChanges"] if c["type"]=="created" and "Coin" in c["objectType"])
now=int(time.time()*1000); ra=now+60_000; win=7*86400*1000  # 7-day evidence window
q="Will an OpenAI model top a public coding benchmark this year?"
seed=split(500_000_000)
o=json.loads(sh(["sui","client","call","--package",PKG,"--module","market","--function","create_market_entry","--gas-budget",GB,"--json",
  "--args","0x"+q.encode().hex(),"0x"+"a7"*32,"12",json.dumps([RECLAIM_ATTESTOR]),"[3]","[0]","1","0",str(ra),str(win),"100",seed,CLOCK]).stdout)
mid=next(c["objectId"] for c in o["objectChanges"] if c["type"]=="created" and c.get("objectType","").endswith("::market::Market"))
print("reclaim market",mid)
MK["reclaim_market"]={"id":mid,"emoji":"🔗","question":q,"category":"zkTLS · prove it yourself",
  "phase":"evidence","resolve_after":ra,"window":win,"attestor":RECLAIM_ATTESTOR,"attestor_idx":0}
json.dump(MK,open(f"{ROOT}/demo/web/markets.json","w"),indent=2)
# reclaim creds block (user fills app_id/secret/provider_id from dev.reclaimprotocol.org)
CFG["reclaim"]={"app_id":"","app_secret":"","provider_id":"","attestor":RECLAIM_ATTESTOR}
json.dump(CFG,open(f"{ROOT}/demo/web/config.json","w"),indent=2)
print("added reclaim_market + reclaim config block (fill app_id/secret/provider_id to enable in-app proofs)")
