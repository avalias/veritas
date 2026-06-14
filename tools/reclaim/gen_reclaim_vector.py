#!/usr/bin/env python3
"""Generate the on-chain test vector for dispute::reclaim — a REAL
Reclaim-format attestor signature (EIP-191 secp256k1 over the canonical
claim), produced with eth-account exactly as a Reclaim attestor signs.

In production you do NOT generate the attestor key: a Reclaim attestor in
the witness network signs, and you pin its real address (e.g.
0xDa11C9Da04Ab02C4AF9374B27A5E727944D3E1dD). Here we use a deterministic
key so the Move test is reproducible; the verification path is identical.

    pip install eth-account 'eth-hash[pycryptodome]'
    python3 gen_reclaim_vector.py
"""
import json
from eth_account import Account
from eth_account.messages import encode_defunct
from eth_hash.auto import keccak

acct = Account.from_key(b"\x42" * 32)  # deterministic test attestor

provider = "http"
parameters = json.dumps(
    {"url": "https://www.bbc.com/news", "method": "GET",
     "responseMatches": [{"type": "contains", "value": "Starship reaches orbit"}]},
    separators=(",", ":"))
context = json.dumps(
    {"extractedParameters": {"headline": "Starship reaches orbit"}, "providerHash": "0xbbc"},
    separators=(",", ":"))
owner = acct.address.lower()
timestamp_s = 1734200000
epoch = 1

identifier = "0x" + keccak(f"{provider}\n{parameters}\n{context}".encode()).hex()
signed_data = f"{identifier}\n{owner}\n{timestamp_s}\n{epoch}"
sig = Account.sign_message(encode_defunct(text=signed_data), acct.key)

out = {
    "provider": provider.encode().hex(),
    "parameters": parameters.encode().hex(),
    "context": context.encode().hex(),
    "owner": owner.encode().hex(),
    "timestamp_s": timestamp_s,
    "epoch": epoch,
    "signature": bytes(sig.signature).hex(),
    "attestor_addr": acct.address[2:].lower(),
}
json.dump(out, open("reclaim_vector.json", "w"), indent=2)
print("attestor:", acct.address)
print("identifier:", identifier)
print("wrote reclaim_vector.json")
