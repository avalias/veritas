// A tiny local HTTP server that generates a REAL zkTLS proof on demand, through
// our self-hosted attestor, so the dApp can do it live from the browser.
//   GET /prove?source=<name>  ->  { value, provider, parameters, context, owner,
//                                   timestampS, epoch, signature }  (all real)
// Run on Node 20/22 with the attestor up on :8001.  Listens on :8788.
import { setCryptoImplementation } from '@reclaimprotocol/tls';
import { webcryptoCrypto } from '@reclaimprotocol/tls/webcrypto';
setCryptoImplementation(webcryptoCrypto);
import { createClaimOnAttestor } from '@reclaimprotocol/attestor-core';
import http from 'http';

const ATTESTOR = process.env.ATTESTOR_BASE_URL || 'ws://localhost:8001/ws';

// real, stable, free, no-key endpoints the judge can read
export const SOURCES = {
  coinbase: { url: 'https://api.coinbase.com/v2/prices/BTC-USD/spot', regex: '"amount":"(?<value>[0-9.]+)"' },
  forex:    { url: 'https://open.er-api.com/v6/latest/USD', regex: '"EUR":(?<value>[0-9.]+)' },
  news:     { url: 'https://hacker-news.firebaseio.com/v0/item/8863.json', regex: '"title":"(?<value>[^"]+)"' },
  football: { url: 'https://www.thesportsdb.com/api/v1/json/123/lookupevent.php?id=2052705', regex: '"strEvent":"(?<value>[^"]+)"' },
};

async function prove(src) {
  const r = await createClaimOnAttestor({
    name: 'http',
    params: { url: src.url, method: 'GET', responseMatches: [{ type: 'regex', value: src.regex }], responseRedactions: [{ regex: src.regex }] },
    // identity encoding: stop servers gzipping the body (which the attestor returns
    // base64-wrapped, breaking the regex). plain text keeps the regex working.
    secretParams: { headers: { 'User-Agent': 'Mozilla/5.0 (compatible; Veritas/1.0)', 'Accept-Encoding': 'identity' } },
    ownerPrivateKey: '0x' + '11'.repeat(32),
    client: { url: ATTESTOR },
  });
  const cd = r.claim;
  const sig = '0x' + Buffer.from(r.signatures.claimSignature).toString('hex');
  let value = null; try { value = JSON.parse(cd.context).extractedParameters?.value; } catch {}
  return { value, provider: cd.provider, parameters: cd.parameters, context: cd.context, owner: cd.owner, timestampS: cd.timestampS, epoch: cd.epoch, signature: sig };
}

http.createServer(async (req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  const u = new URL(req.url, 'http://x');
  if (!u.pathname.startsWith('/prove')) { res.end('zktls gen server up'); return; }
  // either a named preset, or an arbitrary https url + regex chosen in the dApp.
  // Never silently fall back to a different source: a wrong/unknown request must
  // error, so a page can never show one source's question over another's value.
  const url = u.searchParams.get('url'), regex = u.searchParams.get('regex'), source = u.searchParams.get('source');
  let src;
  if (url && regex && url.startsWith('https://')) src = { url, regex };
  else if (source) src = SOURCES[source];               // named preset (or undefined if unknown)
  else if (!url && !regex) src = SOURCES.coinbase;       // bare /prove smoke test
  if (!src) {
    res.statusCode = 400; res.setHeader('Content-Type', 'application/json');
    res.end(JSON.stringify({ error: 'need an https url+regex, or a known source name (' + Object.keys(SOURCES).join(', ') + ')' }));
    return;
  }
  console.error('proving', src.url);
  try {
    // a server can occasionally hand the attestor a gzip/chunked body it returns
    // base64-wrapped, so the regex finds nothing; retry a couple times for a clean read.
    let out;
    for (let i = 0; i < 3; i++) { out = await prove(src); if (out.value != null) break; }
    res.setHeader('Content-Type', 'application/json');
    res.end(JSON.stringify(out));
  } catch (e) {
    res.statusCode = 500; res.setHeader('Content-Type', 'application/json');
    res.end(JSON.stringify({ error: String(e && e.message || e).slice(0, 300) }));
  }
}).listen(8788, () => console.error('zktls gen server on http://localhost:8788  (attestor', ATTESTOR + ')'));
