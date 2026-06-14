import { chromium } from 'playwright';
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1100, height: 1400 } });
const errs = [];
p.on('console', m => { if (m.type()==='error') errs.push('CONSOLE: '+m.text()); });
p.on('pageerror', e => errs.push('PAGEERR: '+e.message));
await p.goto('http://127.0.0.1:8777/app.html', { waitUntil: 'networkidle', timeout: 30000 }).catch(e=>errs.push('GOTO: '+e.message));
await p.waitForTimeout(4000); // let RPC reads populate
await p.screenshot({ path: '/tmp/app_full.png', fullPage: true });
// also open a market modal
try { await p.click('.mcard'); await p.waitForTimeout(1500); await p.screenshot({ path: '/tmp/app_modal.png' }); } catch(e){ errs.push('MODAL: '+e.message); }
console.log('ERRORS:', errs.length ? errs.join('\n') : 'none');
const txt = await p.evaluate(()=>document.querySelector('#grid')?.innerText?.slice(0,200));
console.log('GRID TEXT:', txt);
await b.close();
