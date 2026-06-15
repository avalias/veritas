#!/usr/bin/env python3
"""ONE command to make the dApp judge-ready. Run it right before handing over.

It stages two purpose-built markets so a judge can witness the WHOLE lifecycle
on-chain in ~5 minutes:

  ⚡ LIVE   — fresh, in trading. The judge personally does buy → AI-judge →
             submit zkTLS proof → resolve → redeem, each a real signed tx, as
             the market walks its phases live (guided by an on-screen countdown).
  ⚖️ READY  — already past its evidence window with a YES proof admitted, so the
             judge can click Resolve and see an outcome decided on-chain INSTANTLY
             (a robust fallback that never depends on timing).

Everything else (8 trading markets, the open evidence market + live Qwen judge,
the Fraud Lab, create-your-own) is already live. Prints the URL + the 5-min script.

  python3 demo/judge_setup.py
"""
import subprocess
import time
import judge_lib as L

import os
SEED = int(L.CFG.get("seed_liq", 150_000_000))     # liquidity per staged market
T_TRADE = int(os.environ.get("T_TRADE", 150_000))  # ⚡ trading window (ms) — judge buys here
T_WIN = int(os.environ.get("T_WIN", 75_000))        # ⚡ evidence window (ms) — judge submits here
# → ⚡ becomes resolvable ~225s after creation; redeemable right after. The
# instant features (Qwen judge, Fraud Lab, create, other trades) fill the waits.
# (T_TRADE / T_WIN env overrides let the E2E test run the whole arc fast.)

def stage_live():
    now = L.now_ms()
    ra = now + T_TRADE
    win = T_WIN
    mid = L.create_market("Will independent wires confirm Starship reached orbit?", ra, win, k=1, seed_mist=SEED)
    proofs = L.all_source_proofs((ra + win // 2) // 1000)  # one per source, mid-window
    return {"id": mid, "emoji": "⚡", "question": "Will independent wires confirm Starship reached orbit?",
            "category": "LIVE · trade → judge → resolve → redeem", "kind": "live",
            "resolve_after": ra, "window": win, "k": 1, "proofs": proofs, "proof": proofs[0]}

def stage_resolve_ready():
    now = L.now_ms()
    ra = now + 8_000
    win = 22_000
    mid = L.create_market("Did Starship reach orbit? (ready to resolve)", ra, win, k=1, seed_mist=SEED)
    proofs = L.all_source_proofs((ra + 5_000) // 1000)
    print("  ⚖️  waiting for the evidence window to open…")
    while L.now_ms() < ra + 1_500:
        time.sleep(1)
    L.submit_proof(mid, proofs[0])                   # pre-admit the first source
    print("  ⚖️  YES proof admitted; waiting for the window to close…")
    while L.now_ms() < ra + win + 1_500:
        time.sleep(1)
    return {"id": mid, "emoji": "⚖️", "question": "Did Starship reach orbit?",
            "category": "READY · click Resolve now", "kind": "live",
            "resolve_after": ra, "window": win, "k": 1, "proofs": proofs, "proof": proofs[0],
            "preadmitted": [L.SOURCES[0]["name"]]}

def main():
    bal = L.gas_total_sui()
    print(f"Staging judge-ready markets on devnet…  (gas: {bal:.2f} SUI)")
    if bal < 3:
        print("  ⚠️  LOW GAS — fund the operator address before the demo:\n"
              "      curl -s -X POST https://faucet.devnet.sui.io/v2/gas -H 'Content-Type: application/json' \\\n"
              f"        -d '{{\"FixedAmountRequest\":{{\"recipient\":\"{subprocess.run(['sui','client','active-address'],capture_output=True,text=True).stdout.strip()}\"}}}}'")
    # do the slow steps FIRST (⚖️ wait + fraud staging), then mint ⚡ LAST so its
    # full trading window is fresh when the operator hands over.
    ready = stage_resolve_ready()
    print("  ⚖️  resolve-ready market:", ready["id"])
    L.ensure_fraud()  # arm the Fraud Lab (fresh un-convicted Fact) if needed
    live = stage_live()
    print("  ⚡  live market:", live["id"])

    m = L.load_markets()
    m["live_market"] = live
    m["resolve_ready_market"] = ready
    L.save_markets(m)

    url = "http://127.0.0.1:8777/app.html"
    print(f"""
────────────────────────────────────────────────────────────────────
  READY.  Open {url}
  (resolver must be up:  cargo run -p qwen --release --bin resolver)

  Just hand it over — the on-screen GUIDED TOUR (bottom-right) walks the
  judge through all 7 steps, one at a time, and collects every tx receipt:
   1 connect wallet · 2 buy YES on ⚡ · 3 add a source (BBC/Reuters/AP) →
   zkTLS proof · 4 watch the Qwen judge read it · 5 resolve · 6 redeem ·
   7 convict a fraud.  Each step shows a clickable suiscan transaction.
  Fallback if ⚡ timing slips: the ⚖️ READY card resolves on one click.
  Tip: run  ./demo/go.sh  to start everything + auto-replenish in one go.
────────────────────────────────────────────────────────────────────""")

if __name__ == "__main__":
    main()
