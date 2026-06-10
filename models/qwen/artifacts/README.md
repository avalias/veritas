# Qwen3-0.6B artifacts (Apache 2.0, Qwen team / Alibaba)

`config.json` and `tokenizer.json` are vendored in-repo. The weights are
NOT committed (1.5 GB); fetch and verify:

```
curl -L -o model.safetensors \
  "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/model.safetensors"
shasum -a 256 -c <<< "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b  model.safetensors"
```

Pinned hashes (sha256, see ARTIFACT_HASHES.txt):
- config.json      `660db3b7…442f27dd`
- tokenizer.json   `aeb13307…4492dae4`
- model.safetensors `f47f7117…96874b`

Read by `qwen::config` / `qwen::tensors`; quantized on load by
`qwen::quant` (the repo's only float code, offline-quarantined).
