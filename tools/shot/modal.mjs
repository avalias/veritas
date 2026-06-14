import { chromium } from 'playwright';
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 760, height: 1200 } });
const errs=[]; p.on('pageerror',e=>errs.push(e.message));
await p.goto('http://127.0.0.1:8777/app.html');
await p.waitForSelector('.mcard', { timeout: 20000 });
await p.waitForTimeout(2500);
// trading market (first card)
await p.locator('.mcard').first().click();
await p.waitForTimeout(1200);
await p.screenshot({ path: '/tmp/modal_trade.png' });
await p.locator('#closeX').click(); await p.waitForTimeout(400);
// evidence market (last card)
await p.locator('.mcard').last().click();
await p.waitForTimeout(1200);
await p.screenshot({ path: '/tmp/modal_evidence.png' });
console.log('errors:', errs.join(' | ')||'none');
await b.close();
