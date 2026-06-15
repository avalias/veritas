#!/usr/bin/env bash
# ONE command to make everything judge-ready and keep it fresh across judges.
#   ./demos/prediction-market/go.sh
# Starts the AI-judge resolver + the dApp server (if not already up), stages
# fresh ⚡/⚖️ markets + arms the Fraud Lab, launches the auto-replenisher, and
# opens the dApp. Re-run any time to reset the board for the next judge.
set -e
cd "$(git rev-parse --show-toplevel)"
export PATH="$HOME/.cargo/bin:$PATH"

# 1. the AI judge on :8899. Use the smarter Qwen3-1.7B if it has been fetched
# (opml/models/qwen/fetch-1.7b.sh); otherwise fall back to the 0.6B reference.
if ! curl -s -m1 http://127.0.0.1:8899/ >/dev/null 2>&1; then
  QDIR=""
  if [ -f opml/models/qwen/artifacts-1.7b/model.safetensors ]; then
    QDIR="$PWD/opml/models/qwen/artifacts-1.7b"; echo "· starting the AI judge (Qwen3-1.7B)…"
  else
    echo "· starting the AI judge (Qwen3-0.6B; run opml/models/qwen/fetch-1.7b.sh for the smarter 1.7B)…"
  fi
  [ -x target/release/resolver ] || cargo build -q -p qwen --release --bin resolver
  if [ -n "$QDIR" ]; then QWEN_DIR="$QDIR" nohup ./target/release/resolver >/tmp/resolver.log 2>&1 &
  else nohup ./target/release/resolver >/tmp/resolver.log 2>&1 & fi
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

# 2b. if the self-hosted zkTLS attestor is up (separate, Node 20/22), also start
# the live-proof gen server so zktls.html works. The attestor itself is started
# separately — see tools/zktls/README.md.
if lsof -nP -iTCP:8001 -sTCP:LISTEN >/dev/null 2>&1 && ! lsof -nP -iTCP:8788 -sTCP:LISTEN >/dev/null 2>&1; then
  ( source ~/.nvm/nvm.sh >/dev/null 2>&1; nvm use 22 >/dev/null 2>&1
    cd tools/reclaim && nohup node gen_server.mjs >/tmp/genserver.log 2>&1 & )
  echo "· zkTLS gen server up (:8788)"
fi

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
