#!/usr/bin/env python3
"""Demonstrate the anti-voting rule on-chain: a market that resolves YES
ONLY when TWO INDEPENDENT zkTLS sources confirm (k=2, two distinct pinned
attestors / trust groups). Proves resolution counts independent proofs, not
submissions — the opposite of token voting. Appends to markets.json.
"""
import json, subprocess, sys, time
from eth_account import Account
from eth_account.messages import encode_defunct
from eth_hash.auto import keccak
ROOT=subprocess.run(["git","rev-parse","--show-toplevel"],capture_output=True,text=True).stdout.strip()
CFG=json.load(open(f"{ROOT}/demo/web/config.json")); MK=json.load(open(f"{ROOT}/demo/web/markets.json"))
PKG,CLOCK,GB=CFG["package"],"0x6","400000000"
# two INDEPENDENT attestors (distinct keys → distinct trust groups)
A0=Account.from_key(b"\x42"*32); A1=Account.from_key(b"\x43"*32)
ATT=[A0.address.lower(),A1.address.lower()]
def sh(a):
    r=subprocess.run(a,capture_output=True,text=True)
    if r.returncode!=0: print("FAIL",r.stderr[-400:],r.stdout[-200:],file=sys.stderr); sys.exit(1)
    return r
def gas(): return max(json.loads(sh(["sui","client","gas","--json"]).stdout),key=lambda c:int(c["mistBalance"]))["gasCoinId"]
def split(n):
    o=json.loads(sh(["sui","client","split-coin","--coin-id",gas(),"--amounts",str(n),"--gas-budget",GB,"--json"]).stdout)
    return next(c["objectId"] for c in o["objectChanges"] if c["type"]=="created" and "Coin" in c["objectType"])
def proof(acct,idx,src,head,ra):
    provider="http"
    parameters=json.dumps({"url":f"https://www.{src}","method":"GET","responseMatches":[{"type":"contains","value":head}]},separators=(",",":"))
    context=json.dumps({"extractedParameters":{"headline":head},"providerHash":"0x"+src[:6]},separators=(",",":"))
    owner=acct.address.lower(); ts=(ra+15_000)//1000
    ident="0x"+keccak(f"{provider}\n{parameters}\n{context}".encode()).hex()
    sig=Account.sign_message(encode_defunct(text=f"{ident}\n{owner}\n{ts}\n1"),acct.key)
    return [str(idx),"1","0x"+provider.encode().hex(),"0x"+parameters.encode().hex(),"0x"+context.encode().hex(),"0x"+owner.encode().hex(),str(ts),"1","0x"+bytes(sig.signature).hex()]

now=int(time.time()*1000); ra=now+90_000; win=90_000
seed=split(500_000_000)
q="Did the wires confirm Starship reached orbit? (needs 2 independent sources)"
o=json.loads(sh(["sui","client","call","--package",PKG,"--module","market","--function","create_market_entry","--gas-budget",GB,"--json",
  "--args","0x"+q.encode().hex(),"0x"+"a7"*32,"12",json.dumps(ATT),"[3,3]","[0,1]","2","0",str(ra),str(win),"100",seed,CLOCK]).stdout)
mid=next(c["objectId"] for c in o["objectChanges"] if c["type"]=="created" and c.get("objectType","").endswith("::market::Market"))
print("k=2 market",mid,"— waiting for evidence window…")
while int(time.time()*1000) < ra+3000: time.sleep(3)

# source 1 (Reuters, attestor 0) — one confirmation, NOT enough alone
for args in [proof(A0,0,"reuters.com","Starship reached orbit",ra)]:
    sh(["sui","client","call","--package",PKG,"--module","market","--function","submit_web_proof","--gas-budget",GB,"--json","--args",mid]+args+[CLOCK])
print("source 1 (Reuters) admitted — 1 of 2 groups (not enough)")
# source 2 (BBC, attestor 1) — second INDEPENDENT confirmation → meets k=2
for args in [proof(A1,1,"bbc.com","SpaceX Starship reaches orbit",ra)]:
    sh(["sui","client","call","--package",PKG,"--module","market","--function","submit_web_proof","--gas-budget",GB,"--json","--args",mid]+args+[CLOCK])
print("source 2 (BBC) admitted — 2 of 2 independent groups")

while int(time.time()*1000) < ra+win+2000: time.sleep(3)
sh(["sui","client","call","--package",PKG,"--module","market","--function","resolve","--gas-budget",GB,"--json","--args",mid,CLOCK])
f=json.loads(sh(["sui","client","object",mid,"--json"]).stdout)["content"]["fields"]
print("RESOLVED:",["OPEN","YES","NO","VOID"][int(f["outcome"])],"· yes_groups",len(f["yes_groups"]))
MK["markets"].append({"id":mid,"emoji":"🛰️","question":"Wires confirm Starship orbit (2 independent sources)","category":"Resolved · k=2 zkTLS","phase":"resolved","price_yes":100})
json.dump(MK,open(f"{ROOT}/demo/web/markets.json","w"),indent=2)
print("appended k=2 showcase")
