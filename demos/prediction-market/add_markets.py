#!/usr/bin/env python3
"""Append a few more curated markets to the live devnet set (keeps existing
ones; nudges opening odds for variety). Updates demos/prediction-market/web/markets.json.
"""
import json, subprocess, sys, time
ROOT=subprocess.run(["git","rev-parse","--show-toplevel"],capture_output=True,text=True).stdout.strip()
CFG=json.load(open(f"{ROOT}/demos/prediction-market/web/config.json"))
MK=json.load(open(f"{ROOT}/demos/prediction-market/web/markets.json"))
PKG,CLOCK,GB="0x"+CFG["package"].replace("0x",""),"0x6","400000000"
ATT=["0x17c5185167401ed00cf5f5b2fc97d9bbfdb7d025","0xda11c9da04ab02c4af9374b27a5e727944d3e1dd","0x2222222222222222222222222222222222222222"]
JR="0x"+"a7"*32; DAY=86400*1000; LIQ=500_000_000
def sh(a):
    r=subprocess.run(a,capture_output=True,text=True)
    if r.returncode!=0: print("FAIL",r.stderr[-400:],file=sys.stderr); sys.exit(1)
    return r
def gas(): return max(json.loads(sh(["sui","client","gas","--json"]).stdout),key=lambda c:int(c["mistBalance"]))["gasCoinId"]
def split(n):
    o=json.loads(sh(["sui","client","split-coin","--coin-id",gas(),"--amounts",str(n),"--gas-budget",GB,"--json"]).stdout)
    return next(c["objectId"] for c in o["objectChanges"] if c["type"]=="created" and "Coin" in c["objectType"])
def create(q):
    seed=split(LIQ); now=int(time.time()*1000)
    o=json.loads(sh(["sui","client","call","--package",PKG,"--module","market","--function","create_market_entry",
      "--gas-budget",GB,"--json","--args","0x"+q.encode().hex(),JR,"12",json.dumps(ATT),"[3,3,3]","[0,1,2]","2","0",
      str(now+30*DAY),str(7*DAY),"100",seed,CLOCK]).stdout)
    return next(c["objectId"] for c in o["objectChanges"] if c["type"]=="created" and c.get("objectType","").endswith("::market::Market"))
def nudge(mid,side,amt):
    coin=split(amt); sh(["sui","client","call","--package",PKG,"--module","market","--function","buy_"+side,"--gas-budget",GB,"--json","--args",mid,coin,CLOCK])
def price(mid):
    f=json.loads(sh(["sui","client","object",mid,"--json"]).stdout)["content"]["fields"];ry,rn=int(f["reserve_yes"]),int(f["reserve_no"]);return round(rn/(ry+rn)*100)

MORE=[
 ("⚽","Will Real Madrid reach the 2026 Champions League final?","Sports","yes",90_000_000),
 ("🎬","Will the next Avatar film gross over $2B worldwide?","Culture","no",100_000_000),
 ("🌡️","Will 2026 be confirmed the hottest year on record?","Climate","yes",260_000_000),
 ("🧠","Will an AI win a gold medal at the 2026 Math Olympiad?","AI","no",70_000_000),
]
have={m["id"] for m in MK["markets"]}
for emoji,q,cat,side,amt in MORE:
    if any(m["question"]==q for m in MK["markets"]): continue
    mid=create(q); nudge(mid,side,amt); p=price(mid)
    MK["markets"].append({"id":mid,"emoji":emoji,"question":q,"category":cat,"phase":"trading","price_yes":p})
    print(f"{emoji} {mid[:12]} YES {p}%")
json.dump(MK,open(f"{ROOT}/demos/prediction-market/web/markets.json","w"),indent=2)
print("total trading markets:",len(MK["markets"]))
