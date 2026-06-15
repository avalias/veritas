import { chromium } from 'playwright';
const b = await chromium.launch(); const p = await b.newPage();
const errs = []; p.on('pageerror', e => errs.push(e.message));
p.on('console', m => { if (m.type() === 'error') errs.push('C:' + m.text()); });

await p.goto('http://127.0.0.1:8777/zktls.html', { waitUntil: 'networkidle' });
await p.waitForTimeout(800);

// 1. cards rendered
const cards = await p.$$eval('#cards .demo h3', els => els.map(e => e.textContent));
console.log('CARDS (' + cards.length + '):', cards.join(' | '));

// 2. select a new source (GitHub stars) then run the proof on coinbase (fast, stable)
await p.evaluate(() => [...document.querySelectorAll('#cards .demo')].find(a => /Bitcoin/.test(a.textContent)).click());
console.log('source URL shown:', await p.$eval('#srcurl', e => e.textContent).catch(() => '(none)'));

console.log('clicking Generate — real TLS-MPC proof + judge, up to 150s…');
await p.click('#prove');
await p.waitForSelector('#forgeval', { timeout: 150000 });   // appears only after proof + judge complete

const proven = await p.$eval('#result .step.done b', e => e.textContent).catch(() => '?');
const recovered = await p.$eval('#result .step.done small .mono', e => e.textContent).catch(() => '?');
const verdict = await p.$eval('#jverdict .t b', e => e.textContent).catch(() => '?');
console.log('PROVEN value :', proven);
console.log('RECOVERS     :', recovered);
console.log('JUDGE verdict:', verdict);

// 3. forgery test: change the proven value, recover again -> must NOT be the pinned attestor
await p.fill('#forgeval', '999999');
await p.click('#recover');
await p.waitForTimeout(400);
const forgeOut = (await p.$eval('#forgeout', e => e.textContent).catch(() => '?')).replace(/\s+/g, ' ').trim();
console.log('FORGE result :', forgeOut.slice(0, 220));

console.log('ERRORS:', errs.filter(e => !/8899|8788|favicon|net::ERR|Failed to fetch/.test(e)).join(' | ') || 'none');
await p.screenshot({ path: '/tmp/zktls_verify.png', fullPage: true });
await b.close();
