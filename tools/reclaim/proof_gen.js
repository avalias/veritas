// Generate a REAL zkTLS proof of a REAL public endpoint via Reclaim's
// public attestor, and dump the attestor-signed claim so we can verify the
// secp256k1 attestor signature on-chain (Sui-native ecdsa_k1).
import nodeCrypto from 'node:crypto'
// Node 24's global `crypto` is Web Crypto (no randomBytes); attestor-core
// expects node-style randomBytes. Bridge it.
try {
  if (typeof globalThis.crypto.randomBytes !== 'function') {
    globalThis.crypto.randomBytes = (...a) => nodeCrypto.randomBytes(...a)
  }
} catch {
  Object.defineProperty(globalThis, 'crypto', { value: Object.assign(Object.create(globalThis.crypto), { randomBytes: nodeCrypto.randomBytes }), configurable: true })
}
import { createClaimOnAttestor } from '@reclaimprotocol/attestor-core'

const ATTESTOR = process.env.ATTESTOR_BASE_URL || 'wss://attestor.reclaimprotocol.org/ws'
const URL = process.env.TARGET_URL || 'https://api.coinbase.com/v2/prices/BTC-USD/spot'
const REGEX = process.env.TARGET_REGEX || '"amount":"(?<value>[0-9.]+)"'
const ownerPrivateKey = '0x' + '11'.repeat(32) // any user key; identifies the proof owner

console.error('attestor:', ATTESTOR, '\ntarget  :', URL)

const res = await createClaimOnAttestor({
  name: 'http',
  params: {
    url: URL,
    method: 'GET',
    responseMatches: [{ type: 'regex', value: REGEX }],
    responseRedactions: [{ regex: REGEX }],
  },
  secretParams: { cookieStr: '' },
  ownerPrivateKey,
  client: { url: ATTESTOR },
})

const sig = res?.signatures
const claim = res?.claim
console.error('=== RESULT KEYS ===', Object.keys(res || {}))
console.error('claim:', JSON.stringify(claim))
console.error('attestor addr:', sig?.attestorAddress || sig?.witnessAddresses)
console.log(JSON.stringify(res, (k, v) => (v?.type === 'Buffer' ? Buffer.from(v.data).toString('hex') : v), 2))
process.exit(0)
