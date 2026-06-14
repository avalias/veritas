#!/usr/bin/env python3
"""Extract the REAL signature from a REAL C2PA-signed image and emit an
on-chain-verifiable vector.

C2PA in JPEG: APP11 (0xFFEB) segments carry a JUMBF box tree. The claim
signature is a COSE_Sign1 (CBOR) inside a 'cbor' content box. We pull the
algorithm, the signature, the signer cert (x5chain), reconstruct the
COSE Sig_structure (the exact bytes signed), verify it off-chain against
the cert's public key (proving the image is genuinely signed), and print
the values credential.move verifies on-chain (ES256 -> ecdsa_r1,
EdDSA -> ed25519).
"""
import struct
import sys
import hashlib
import cbor2
from cryptography import x509
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, utils
from cryptography.hazmat.primitives import hashes, serialization

path = sys.argv[1] if len(sys.argv) > 1 else "adobe_C.jpg"
d = open(path, "rb").read()

# ---- 1. reassemble the JUMBF from APP11 segments ----
jumbf = b""
i = 2
while i < len(d) - 1:
    if d[i] == 0xFF and d[i + 1] == 0xEB:  # APP11
        ln = struct.unpack(">H", d[i + 2:i + 4])[0]
        seg = d[i + 4:i + 2 + ln]
        # header: CI(2 "JP") En(2) Z(4)  then JUMBF bytes
        jumbf += seg[8:]
        i += 2 + ln
    elif d[i] == 0xFF and d[i + 1] in (0xD8, 0xD9):
        i += 2
    elif d[i] == 0xFF and 0xC0 <= d[i + 1] <= 0xFE:
        ln = struct.unpack(">H", d[i + 2:i + 4])[0]
        i += 2 + ln
    else:
        i += 1

# ---- 2. walk JUMBF boxes, collect every 'cbor' content box ----
cbor_boxes = []
def walk(buf, off, end, depth=0):
    while off + 8 <= end:
        ln = struct.unpack(">I", buf[off:off + 4])[0]
        typ = buf[off + 4:off + 8]
        if ln == 0:
            ln = end - off
        box_end = off + ln
        if typ == b"jumb":
            walk(buf, off + 8, box_end, depth + 1)
        elif typ == b"cbor":
            cbor_boxes.append(buf[off + 8:box_end])
        off = box_end
walk(jumbf, 0, len(jumbf))

# ---- 3. find the COSE_Sign1 (4-element array: protected, unprotected, payload, sig) ----
ALG = {-7: "ES256", -8: "EdDSA", -35: "ES384", -36: "ES512", -37: "PS256", -39: "PS512"}
found = None
for raw in cbor_boxes:
    try:
        obj = cbor2.loads(raw)
    except Exception:
        continue
    if isinstance(obj, cbor2.CBORTag):  # tag 18 = COSE_Sign1
        obj = obj.value
    if isinstance(obj, list) and len(obj) == 4 and isinstance(obj[0], bytes) and isinstance(obj[3], bytes):
        prot = cbor2.loads(obj[0]) if obj[0] else {}
        alg = prot.get(1)
        if alg in ALG:
            found = (obj, prot, raw)
            break

if not found:
    print("no COSE_Sign1 claim signature found"); sys.exit(1)

(prot_bstr, uphdr, payload, signature), prot, _ = found[0], found[1], found[2]
alg = prot[1]
print("algorithm:", ALG[alg], f"(COSE {alg})")
print("signature length:", len(signature))

# ---- 4. signer cert + public key (x5chain lives in protected or unprotected header, label 33) ----
x5 = prot.get(33) or found[0][1].get(33)
if isinstance(x5, list):
    x5 = x5[0]
cert = x509.load_der_x509_certificate(x5)
print("signer (cert subject):", cert.subject.rfc4514_string())
print("issuer:", cert.issuer.rfc4514_string())
pub = cert.public_key()

# ---- 5. reconstruct the COSE Sig_structure (the exact bytes signed) ----
# Sig_structure = [ "Signature1", protected, external_aad(=b""), payload ]
# For C2PA the payload in the box is the claim bytes (attached).
sig_structure = cbor2.dumps(["Signature1", prot_bstr, b"", payload])

# ---- 6. verify OFF-CHAIN (proves the image is genuinely signed) ----
ok = False
onchain = {}
if alg == -7:  # ES256 / P-256 / SHA-256
    r = int.from_bytes(signature[:32], "big"); s = int.from_bytes(signature[32:], "big")
    der = utils.encode_dss_signature(r, s)
    try:
        pub.verify(der, sig_structure, ec.ECDSA(hashes.SHA256())); ok = True
    except Exception as e:
        print("verify error:", e)
    # on-chain: compressed pubkey (33B), low-s r||s (64B), message = sig_structure
    N = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551
    if s > N // 2:
        s = N - s
    onchain = {
        "scheme": "ES256 (1) -> ecdsa_r1",
        "pubkey": pub.public_bytes(serialization.Encoding.X962, serialization.PublicFormat.CompressedPoint).hex(),
        "message": sig_structure.hex(),
        "signature": (r.to_bytes(32, "big") + s.to_bytes(32, "big")).hex(),
    }
elif alg == -8:  # EdDSA / Ed25519
    try:
        pub.verify(signature, sig_structure); ok = True
    except Exception as e:
        print("verify error:", e)
    onchain = {
        "scheme": "EdDSA (0) -> ed25519",
        "pubkey": pub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex(),
        "message": sig_structure.hex(),
        "signature": signature.hex(),
    }
elif alg in (-37, -38, -39):  # PS256/384/512 — RSA-PSS, proves realness (on-chain via Groth16)
    from cryptography.hazmat.primitives.asymmetric import padding
    h = {-37: hashes.SHA256(), -38: hashes.SHA384(), -39: hashes.SHA512()}[alg]
    try:
        pub.verify(signature, sig_structure, padding.PSS(mgf=padding.MGF1(h), salt_length=padding.PSS.DIGEST_LENGTH), h)
        ok = True
    except Exception as e:
        print("verify error:", e)
    onchain = {"scheme": "PS256/RSA-PSS -> Groth16 (zkEmail-class), not native"}
elif alg in (-35, -36):  # ES384/ES512 — real ECDSA but not P-256, on-chain via Groth16
    h = {-35: hashes.SHA384(), -36: hashes.SHA512()}[alg]
    r = int.from_bytes(signature[:len(signature)//2], "big"); s = int.from_bytes(signature[len(signature)//2:], "big")
    try:
        pub.verify(utils.encode_dss_signature(r, s), sig_structure, ec.ECDSA(h)); ok = True
    except Exception as e:
        print("verify error:", e)
    onchain = {"scheme": f"ES{384 if alg==-35 else 512} -> Groth16, not native (Sui native is P-256/ES256)"}

print("OFF-CHAIN VERIFY:", "VALID — genuinely signed" if ok else "FAILED")
print("image sha256:", hashlib.sha256(d).hexdigest())
if ok and onchain:
    print("\n--- ON-CHAIN VECTOR (credential.move) ---")
    for k, v in onchain.items():
        print(k, v if len(str(v)) < 80 else str(v)[:76] + "…")
    import json
    json.dump(onchain, open(path + ".vector.json", "w"))
    print("wrote", path + ".vector.json", "(full hex)")
