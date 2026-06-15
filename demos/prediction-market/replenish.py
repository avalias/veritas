#!/usr/bin/env python3
"""Auto-replenisher — keeps the board fresh across back-to-back judges.

Watches devnet. The moment a judge RESOLVES the ⚡ live market or the ⚖️
resolve-ready market (or CONVICTS the Fraud Lab dispute), it stages a fresh one
and rewrites markets.json. The dApp re-reads markets.json every 15s, so the next
judge always finds a live market in trading and an un-convicted fraud proof — no
operator intervention between runs.

  python3 demo/replenish.py            # run alongside the resolver + server

Safe: it only replaces the ⚡ once it is RESOLVED (judge finished its lifecycle)
or clearly ABANDONED (window closed >5 min, never resolved), so it never yanks a
market a judge is mid-flow on.
"""
import json, subprocess, sys, time
import judge_lib as L
import judge_setup as S

POLL = 12          # seconds between checks
STALE_MS = 300_000  # ⚡ abandoned if window closed this long ago, unresolved
PHASE_RESOLVED = 2


def fields(obj_id):
    r = subprocess.run(["sui", "client", "object", obj_id, "--json"], capture_output=True, text=True)
    try:
        return json.loads(r.stdout)["content"]["fields"]
    except Exception:
        return None


def main():
    print(f"replenisher watching devnet every {POLL}s …  (Ctrl-C to stop)")
    while True:
        try:
            m = L.load_markets()
            changed = False

            lv = m.get("live_market")
            if lv:
                f = fields(lv["id"])
                if f is not None:
                    resolved = int(f.get("phase", 0)) == PHASE_RESOLVED
                    closed_at = int(f["resolve_after_ms"]) + int(f["evidence_window_ms"])
                    abandoned = (not resolved) and L.now_ms() > closed_at + STALE_MS
                    if resolved or abandoned:
                        print(f"  ⚡ {'resolved' if resolved else 'abandoned'} → staging fresh live market")
                        m["live_market"] = S.stage_live()
                        changed = True
                        print("     new ⚡", m["live_market"]["id"])

            rr = m.get("resolve_ready_market")
            if rr:
                f = fields(rr["id"])
                if f is not None and int(f.get("phase", 0)) == PHASE_RESOLVED:
                    print("  ⚖️ resolved → staging a fresh resolve-ready market")
                    m["resolve_ready_market"] = S.stage_resolve_ready()
                    changed = True
                    print("     new ⚖️", m["resolve_ready_market"]["id"])

            if changed:
                L.save_markets(m)

            if L.fact_status() == 4:  # REJECTED == convicted → re-arm
                print("  🔪 fraud convicted → re-staging a fresh dispute…")
                L.stage_fraud()

        except KeyboardInterrupt:
            print("\nstopped.")
            return
        except Exception as e:
            print("  (transient)", e, file=sys.stderr)
        time.sleep(POLL)


if __name__ == "__main__":
    main()
