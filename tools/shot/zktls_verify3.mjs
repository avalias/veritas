import { chromium } from 'playwright';
const b = await chromium.launch(); const p = await b.newPage();
const errs = []; p.on('pageerror', e => errs.push(e.message));
p.on('console', m => { if (m.type() === 'error') errs.push('C:' + m.text()); });
await p.goto('http://127.0.0.1:8777/zktls.html', { waitUntil: 'domcontentloaded' });
await p.waitForTimeout(700);
const cards = await p.$$eval('#cards .demo', els => els.map(e => e.querySelector('.k').textContent + ' / ' + e.querySelector('h3').textContent));
console.log('CARDS:\n  ' + cards.join('\n  '));

async function run(label, match) {
  if (match) await p.evaluate(m => [...document.querySelectorAll('#cards .demo')].find(a => a.textContent.includes(m)).click(), match);
  await p.waitForTimeout(300);
  await p.click('#prove');
  await p.waitForSelector('#forgeval', { timeout: 240000 });
  const proven = await p.$eval('#result .step.done b', e => e.textContent).catch(() => '?');
  const rec = await p.$eval('#result .step.done small .mono', e => e.textContent).catch(() => '?');
  const verdict = await p.$eval('#jverdict .t b', e => e.textContent).catch(() => '?');
  console.log(`${label}: judge=${verdict} | recovers=${rec.slice(0,12)}… | proven=${proven.slice(0, 75)}`);
}
await run('launchA H3 (want YES)', null);
await p.fill('#forgeval', 'the rocket exploded on the pad'); await p.click('#recover'); await p.waitForTimeout(400);
console.log('  forge:', (await p.$eval('#forgeout', e => e.textContent).catch(() => '')).replace(/\s+/g, ' ').trim().slice(0, 130));
await run('launchB New Glenn (want NO)', 'New Glenn');
await run('BTC (want NO)', 'Bitcoin price');
console.log('ERRORS:', errs.filter(e => !/8899|8788|favicon|net::ERR|Failed to fetch/.test(e)).join(' | ') || 'none');
await b.close();
