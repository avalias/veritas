#!/usr/bin/env python3
"""Generate REAL credential vectors for the multi-scheme verifier
(dispute::credential) and pin REAL news-org DKIM keys from DNS.

Schemes:
  ES256  (SCHEME_ES256=1)  — P-256 / SHA-256, as C2PA Content Credentials
                            use (COSE ES256). Verified on Sui NATIVELY via
                            ecdsa_r1::secp256r1_verify (compressed 33-byte
                            pubkey, 64-byte r||s sig, hash=SHA256).
  RSA2048(SCHEME_RSA2048=2) — PKCS#1 v1.5 / SHA-256, exactly DKIM's scheme.

All crypto here is real. We can't hold a news org's PRIVATE key, so the
sample email/photo is signed by a demo key; the verification math and the
key-provenance path are identical to the real thing, and we ALSO pin the
real DNS-published public keys so the registry path is real.
"""
import base64
import hashlib
import json
import subprocess
import sys

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa, utils

out = {}

# ---- ES256 / C2PA -------------------------------------------------------
# deterministic key from a fixed scalar
scalar = int.from_bytes(hashlib.sha256(b"demo-c2pa-signer").digest(), "big")
sk = ec.derive_private_key(scalar, ec.SECP256R1())
pk = sk.public_key()
# compressed point (33 bytes) — what secp256r1_verify wants
pk_comp = pk.public_bytes(serialization.Encoding.X962, serialization.PublicFormat.CompressedPoint)
# message = the bytes a C2PA manifest binds (here: a news photo's caption+hash)
es_msg = b"AP Photo: Starship completes soft landing, 2026-09-28T14:03Z; sha256=" + hashlib.sha256(b"image-bytes").hexdigest().encode()
der_sig = sk.sign(es_msg, ec.ECDSA(hashes.SHA256()))
r, s = utils.decode_dss_signature(der_sig)
# Sui (like most chains) requires the non-malleable LOW-S form.
N = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551
if s > N // 2:
    s = N - s
rs_sig = r.to_bytes(32, "big") + s.to_bytes(32, "big")  # raw r||s, 64 bytes
out["es256"] = {
    "pubkey_compressed": pk_comp.hex(),
    "message": es_msg.hex(),
    "signature_rs": rs_sig.hex(),
}

# ---- RSA-2048 / DKIM ----------------------------------------------------
rsa_sk = rsa.generate_private_key(public_exponent=65537, key_size=2048)
rsa_pk = rsa_sk.public_key()
n = rsa_pk.public_numbers().n
e = rsa_pk.public_numbers().e
# a realistic DKIM "signed headers" blob
dkim_msg = (b"from:Reuters Wire <alerts@reuters.com>\r\n"
            b"subject:Starship completes soft landing\r\n"
            b"date:Mon, 28 Sep 2026 14:05:00 GMT\r\n")
rsa_sig = rsa_sk.sign(dkim_msg, padding.PKCS1v15(), hashes.SHA256())
out["rsa2048"] = {
    "n": n.to_bytes(256, "big").hex(),
    "e": e,
    "message": dkim_msg.hex(),
    "signature": rsa_sig.hex(),
    "sha256_msg": hashlib.sha256(dkim_msg).hexdigest(),
}

# ---- REAL news-org DKIM public keys from DNS (key provenance is real) ----
def dns_dkim_modulus(selector, domain):
    try:
        r = subprocess.run(["nslookup", "-type=TXT", f"{selector}._domainkey.{domain}"],
                           capture_output=True, text=True, timeout=8)
        txt = r.stdout.replace('"', "").replace("\n", " ").replace("\t", " ")
        if "p=" not in txt:
            return None
        p = txt.split("p=", 1)[1].split(";")[0].strip().replace(" ", "")
        der = base64.b64decode(p + "=" * (-len(p) % 4))
        pub = serialization.load_der_public_key(der)
        if isinstance(pub, rsa.RSAPublicKey):
            return {"bits": pub.key_size, "n_sha256": hashlib.sha256(
                pub.public_numbers().n.to_bytes(pub.key_size // 8, "big")).hexdigest()}
    except Exception as ex:
        return {"error": str(ex)}
    return None

out["real_dkim_keys"] = {
    "reuters.com/selector1": dns_dkim_modulus("selector1", "reuters.com"),
    "nytimes.com/google": dns_dkim_modulus("google", "nytimes.com"),
    "bbc.co.uk/50dkim1": dns_dkim_modulus("50dkim1", "bbc.co.uk"),
}

json.dump(out, sys.stdout, indent=2)
print()
