#!/usr/bin/env python3
"""Generate the pinned ed25519 Web-Credential vector for market_tests.move.

Mirrors `dispute::market::canonical_message` exactly:
    msg = blake2b256( market_addr(32, big-endian) || claim(1)
                      || content_hash(32) || signed_ms(u64 little-endian) )
then signs `msg` with a deterministic ed25519 key (RFC 8032, as Sui's
`ed25519::ed25519_verify` expects). Re-run to regenerate; paste the printed
hex into the `ed25519_admission_matches_offchain_signer` test.
"""
import hashlib
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

SEED = bytes(range(32))  # deterministic test key (0x00..0x1f)
MARKET = (0xCAFE).to_bytes(32, "big")  # @0xCAFE as Sui address bytes
CLAIM = 1
CONTENT_HASH = bytes([0x11]) * 32
SIGNED_MS = 1500

preimage = MARKET + bytes([CLAIM]) + CONTENT_HASH + SIGNED_MS.to_bytes(8, "little")
msg = hashlib.blake2b(preimage, digest_size=32).digest()

sk = Ed25519PrivateKey.from_private_bytes(SEED)
pk = sk.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
sig = sk.sign(msg)

print("pubkey   ", pk.hex())
print("signature", sig.hex())
print("hash     ", CONTENT_HASH.hex())
print("msg      ", msg.hex())
