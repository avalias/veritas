import { chromium } from 'playwright';
const b = await chromium.launch(); const p = await b.newPage();
const errs = []; p.on('pageerror', e => errs.push(e.message));
p.on('console', m => { if (m.type() === 'error') errs.push('C:' + m.text()); });
await p.goto('http://127.0.0.1:8777/zktls.html', { waitUntil: 'domcontentloaded' });
await p.waitForTimeout(700);
console.log('CARDS:', (await p.$$eval('#cards .demo h3', e => e.map(x => x.textContent))).join(' | '));

async function run(label, cardMatch) {
  if (cardMatch) await p.evaluate(m => [...document.querySelectorAll('#cards .demo')].find(a => a.textContent.includes(m)).click(), cardMatch);
  await p.waitForTimeout(300);
  await p.click('#prove');
  await p.waitForSelector('#forgeval', { timeout: 220000 });
  const proven = await p.$eval('#result .step.done b', e => e.textContent).catch(() => '?');
  const rec = await p.$eval('#result .step.done small .mono', e => e.textContent).catch(() => '?');
  const verdict = await p.$eval('#jverdict .t b', e => e.textContent).catch(() => '?');
  console.log(`${label}: proven=${proven} | recovers=${rec.slice(0, 12)}… | judge=${verdict}`);
}
await run('BTC', null);
await p.fill('#forgeval', '999999'); await p.click('#recover'); await p.waitForTimeout(400);
console.log('  forge:', (await p.$eval('#forgeout', e => e.textContent).catch(() => '')).replace(/\s+/g, ' ').trim().slice(0, 150));
await run('Football', 'Football');
console.log('ERRORS:', errs.filter(e => !/8899|8788|favicon|net::ERR|Failed to fetch/.test(e)).join(' | ') || 'none');
await b.close();
