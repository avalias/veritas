#!/usr/bin/env bash
# Fetch Qwen3-1.7B (instruct) and merge its two shards into the single
# model.safetensors the committed-float runtime reads. The judge runs this
# bigger model; the engine is model-agnostic, so nothing else changes.
# ~4 GB download. The weights are gitignored.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DIR="$HERE/artifacts-1.7b"
BASE="https://huggingface.co/Qwen/Qwen3-1.7B/resolve/main"
mkdir -p "$DIR"; cd "$DIR"

[ -f config.json ]    || curl -sL -o config.json    "$BASE/config.json"
[ -f tokenizer.json ] || curl -sL -o tokenizer.json "$BASE/tokenizer.json"

if [ ! -f model.safetensors ]; then
  echo "downloading shards (~4 GB)…"
  curl -L -o s1 "$BASE/model-00001-of-00002.safetensors"
  curl -L -o s2 "$BASE/model-00002-of-00002.safetensors"
  echo "merging shards…"
  python3 - "$DIR/s1" "$DIR/s2" <<'PY'
import json, struct, os, sys
shards = sys.argv[1:3]
headers, datalens = [], []
for sh in shards:
    sz = os.path.getsize(sh)
    with open(sh, 'rb') as f:
        hlen = struct.unpack('<Q', f.read(8))[0]
        headers.append(json.loads(f.read(hlen)))
        datalens.append(sz - 8 - hlen)
combined, base = {}, 0
for h, dl in zip(headers, datalens):
    for name, meta in h.items():
        if name == '__metadata__':
            continue
        b, e = meta['data_offsets']
        combined[name] = {'dtype': meta['dtype'], 'shape': meta['shape'], 'data_offsets': [base + b, base + e]}
    base += dl
hjson = json.dumps(combined, separators=(',', ':')).encode()
with open('model.safetensors', 'wb') as out:
    out.write(struct.pack('<Q', len(hjson)))
    out.write(hjson)
    for sh in shards:
        with open(sh, 'rb') as f:
            hl = struct.unpack('<Q', f.read(8))[0]
            f.seek(8 + hl)
            while (chunk := f.read(8 << 20)):
                out.write(chunk)
print('merged', len(combined), 'tensors')
PY
  rm -f s1 s2
fi
echo "ready: $DIR/model.safetensors ($(du -h "$DIR/model.safetensors" | cut -f1))"
echo "run the judge on it:  QWEN_DIR=$DIR cargo run -p qwen --release --bin resolver"
