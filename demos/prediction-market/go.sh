#!/usr/bin/env bash
# ONE command to make everything judge-ready and keep it fresh across judges.
#   ./demos/prediction-market/go.sh
# Starts the AI-judge resolver + the dApp server (if not already up), stages
# fresh ⚡/⚖️ markets + arms the Fraud Lab, launches the auto-replenisher, and
# opens the dApp. Re-run any time to reset the board for the next judge.
set -e
cd "$(git rev-parse --show-toplevel)"
export PATH="$HOME/.cargo/bin:$PATH"

# 1. the AI judge (real Qwen-0.6B) on :8899
if ! curl -s -m1 http://127.0.0.1:8899/ >/dev/null 2>&1; then
  echo "· starting the AI judge (loading Qwen-0.6B)…"
  [ -x target/release/resolver ] || cargo build -q -p qwen --release --bin resolver
  nohup ./target/release/resolver >/tmp/resolver.log 2>&1 &
  for i in $(seq 1 90); do curl -s -m1 http://127.0.0.1:8899/ >/dev/null 2>&1 && break; sleep 1; done
fi
curl -s -m2 http://127.0.0.1:8899/ >/dev/null 2>&1 && echo "· AI judge up (:8899)" || { echo "!! resolver failed — see /tmp/resolver.log"; exit 1; }

# 2. the dApp server on :8777
if ! curl -s -m1 -o /dev/null http://127.0.0.1:8777/app.html 2>/dev/null; then
  echo "· starting the dApp server (:8777)…"
  nohup python3 demos/prediction-market/web/serve.py >/tmp/serve.log 2>&1 &
  sleep 2
fi
echo "· dApp up (:8777)"

# 3. stage fresh markets + arm the Fraud Lab
python3 demos/prediction-market/judge_setup.py

# 4. keep the board fresh across back-to-back judges
pkill -f 'demos/prediction-market/replenish.py' 2>/dev/null || true
nohup python3 demos/prediction-market/replenish.py >/tmp/replenish.log 2>&1 &
echo "· auto-replenisher running (re-stages after each judge; log: /tmp/replenish.log)"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  READY → http://127.0.0.1:8777/app.html"
echo "  Hand it over. The on-screen Guided Tour walks the judge"
echo "  through all 7 steps; every action shows a suiscan tx."
echo "════════════════════════════════════════════════════════════"
command -v open >/dev/null 2>&1 && open "http://127.0.0.1:8777/app.html" || true
