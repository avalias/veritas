// Generate a real zkTLS proof against a live website, through our self-hosted
// attestor. The one Node-version gotcha (the bundled crypto backend lacks
// randomBytes) is fixed the official way: pin the webcrypto implementation
// before any claim runs.
import { setCryptoImplementation } from '@reclaimprotocol/tls';
import { webcryptoCrypto } from '@reclaimprotocol/tls/webcrypto';
setCryptoImplementation(webcryptoCrypto);

import { createClaimOnAttestor } from '@reclaimprotocol/attestor-core';

const ATTESTOR = process.env.ATTESTOR_BASE_URL || 'ws://localhost:8001/ws';
const URL = process.env.TARGET_URL || 'https://api.coinbase.com/v2/prices/BTC-USD/spot';
const REGEX = process.env.TARGET_REGEX || '"amount":"(?<value>[0-9.]+)"';

console.error('attestor:', ATTESTOR, '| target:', URL);
const res = await createClaimOnAttestor({
  name: 'http',
  params: { url: URL, method: 'GET', responseMatches: [{ type: 'regex', value: REGEX }], responseRedactions: [{ regex: REGEX }] },
  // the http provider requires at least one of cookieStr/authHeader/headers;
  // this public endpoint needs no auth, just a User-Agent.
  secretParams: { headers: { 'User-Agent': 'Mozilla/5.0 (compatible; Veritas/1.0)' } },
  ownerPrivateKey: '0x' + '11'.repeat(32),
  client: { url: ATTESTOR },
});
// serialize Buffers as hex so the proof is plain JSON
console.log(JSON.stringify(res, (k, v) => (v && v.type === 'Buffer' ? Buffer.from(v.data).toString('hex') : v), 2));
process.exit(0);
