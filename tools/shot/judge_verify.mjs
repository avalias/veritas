import { chromium } from 'playwright';
const b = await chromium.launch(); const p = await b.newPage();
const errs = []; p.on('pageerror', e => errs.push(e.message));
p.on('console', m => { if (m.type() === 'error') errs.push('C:' + m.text()); });
await p.goto('http://127.0.0.1:8777/judge.html', { waitUntil: 'domcontentloaded' });
await p.waitForTimeout(600);
const chips = await p.$$eval('#scenarios .demo .k', els => els.map(e => e.textContent));
console.log('SCENARIOS (' + chips.length + '):', chips.join(' | '));

async function ask(label) {
  await p.click('#ask');
  await p.waitForFunction(() => { const e = document.querySelector('#verdict .t b'); return e && e.textContent; }, { timeout: 90000 });
  await p.waitForTimeout(300);
  console.log(label, '| Q:', await p.$eval('#q', e => e.value), '=>', await p.$eval('#verdict .t b', e => e.textContent));
}
await ask('default(Rates, expect NO)');
await p.evaluate(() => [...document.querySelectorAll('#scenarios .demo')].find(a => /Football/.test(a.textContent)).click());
await ask('Football(expect NO)');
await p.evaluate(() => [...document.querySelectorAll('#scenarios .demo')].find(a => /Crypto/.test(a.textContent)).click());
await ask('Crypto(expect YES)');
console.log('ERRORS:', errs.filter(e => !/8899|favicon|net::ERR|Failed to fetch/.test(e)).join(' | ') || 'none');
await b.close();
