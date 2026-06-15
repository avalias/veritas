// Quick load check for a demo page: console/page errors + a key selector's text.
import { chromium } from 'playwright';
const page_name = process.argv[2] || 'slashing.html';
const sel = process.argv[3] || '#n';
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 900, height: 1200 } });
const logs = [];
p.on('pageerror', e => logs.push('PAGEERR ' + e.message));
p.on('console', m => { if (m.type() === 'error') logs.push('CONSOLEERR ' + m.text()); });
await p.goto('http://127.0.0.1:8777/' + page_name, { waitUntil: 'networkidle' }).catch(e => logs.push('GOTO ' + e.message));
await p.waitForTimeout(2500);
const txt = await p.evaluate((s) => document.querySelector(s)?.textContent || '(missing)', sel);
const title = await p.title();
console.log('PAGE   :', page_name, '|', title);
console.log(sel, ':', txt);
console.log('ERRORS :', logs.filter(l => !/ERR_CONNECTION_REFUSED|8899/.test(l)).join(' || ') || 'none');
await p.screenshot({ path: '/tmp/' + page_name.replace('.html', '') + '.png', fullPage: true });
await b.close();
