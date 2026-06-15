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
import time
import judge_lib as L

T_TRADE = 150_000   # ⚡ trading window (ms) — judge buys here (generous margin)
T_WIN = 75_000      # ⚡ evidence window (ms) — judge submits the proof here
# → ⚡ becomes resolvable ~225s after creation; redeemable right after. The
# instant features (Qwen judge, Fraud Lab, create, other trades) fill the waits.

def stage_live():
    now = L.now_ms()
    ra = now + T_TRADE
    win = T_WIN
    mid = L.create_market("Will the wires confirm Starship reached orbit (live demo)?", ra, win, k=1)
    proof = L.build_proof((ra + win // 2) // 1000)  # timestamp mid evidence-window
    return {"id": mid, "emoji": "⚡", "question": "Will the wires confirm Starship reached orbit?",
            "category": "LIVE · trade → judge → resolve → redeem", "kind": "live",
            "resolve_after": ra, "window": win, "proof": proof}

def stage_resolve_ready():
    now = L.now_ms()
    ra = now + 8_000
    win = 22_000
    mid = L.create_market("Did Starship reach orbit? (ready to resolve)", ra, win, k=1)
    proof = L.build_proof((ra + 5_000) // 1000)
    print("  ⚖️  waiting for the evidence window to open…")
    while L.now_ms() < ra + 1_500:
        time.sleep(1)
    L.submit_proof(mid, proof)                       # admit a YES proof on-chain
    print("  ⚖️  YES proof admitted; waiting for the window to close…")
    while L.now_ms() < ra + win + 1_500:
        time.sleep(1)
    return {"id": mid, "emoji": "⚖️", "question": "Did Starship reach orbit?",
            "category": "READY · click Resolve now", "kind": "live",
            "resolve_after": ra, "window": win, "proof": proof, "preadmitted": True}

def main():
    print("Staging judge-ready markets on devnet…")
    live = stage_live()
    print("  ⚡  live market:", live["id"])
    ready = stage_resolve_ready()
    print("  ⚖️  resolve-ready market:", ready["id"])

    m = L.load_markets()
    m["live_market"] = live
    m["resolve_ready_market"] = ready
    L.save_markets(m)

    url = "http://127.0.0.1:8777/app.html"
    print(f"""
────────────────────────────────────────────────────────────────────
  READY.  Open {url}
  (make sure the resolver is up:  cargo run -p qwen --release --bin resolver)

  Hand to the judge — the ⚡ LIVE card is the spine (≈3 min):
   1. Connect Slush (devnet) · buy YES on ⚡            → tx
   2. While ⚡ trades: open the evidence market, click
      "Ask the AI judge" — watch Qwen-0.6B type its verdict
   3. ⚡ flips to EVIDENCE → "Submit the judge's verdict"  → EvidenceAdmitted tx
   4. Try "Submit as NO" — it's refused (you can't vote)
   5. Open the Fraud Lab → "Convict & slash the liar"     → verify_step tx
   6. ⚡ flips to RESOLVABLE → "Resolve"                   → Resolved YES tx
   7. "Redeem winnings"                                    → payout tx
  Fallback if ⚡ timing slips: the ⚖️ READY card resolves on one click.
────────────────────────────────────────────────────────────────────""")

if __name__ == "__main__":
    main()
