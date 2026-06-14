// Patch the global crypto BEFORE loading attestor-core: give it node-style
// randomBytes while keeping web-style subtle/getRandomValues.
const nc = require('node:crypto')
const wc = nc.webcrypto
const merged = new Proxy(wc, { get(t, p) { return p === 'randomBytes' ? nc.randomBytes : t[p] } })
Object.defineProperty(globalThis, 'crypto', { value: merged, configurable: true, writable: true })

const { createClaimOnAttestor } = require('@reclaimprotocol/attestor-core')
const ATTESTOR = process.env.ATTESTOR_BASE_URL || 'wss://attestor.reclaimprotocol.org/ws'
const TURL = process.env.TARGET_URL || 'https://api.coinbase.com/v2/prices/BTC-USD/spot'
const REGEX = process.env.TARGET_REGEX || '"amount":"(?<value>[0-9.]+)"'
;(async () => {
  console.error('attestor:', ATTESTOR, '| target:', TURL)
  const res = await createClaimOnAttestor({
    name: 'http',
    params: { url: TURL, method: 'GET',
      responseMatches: [{ type: 'regex', value: REGEX }],
      responseRedactions: [{ regex: REGEX }] },
    secretParams: { cookieStr: '' },
    ownerPrivateKey: '0x' + '11'.repeat(32),
    client: { url: ATTESTOR },
  })
  console.error('RESULT KEYS:', Object.keys(res || {}))
  console.log(JSON.stringify(res, (k, v) => (v && v.type === 'Buffer' ? Buffer.from(v.data).toString('hex') : v), 2))
})().then(() => process.exit(0)).catch(e => { console.error('ERR:', (e && e.stack || e).toString().slice(0, 500)); process.exit(1) })
