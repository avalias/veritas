import { chromium } from 'playwright';
const b=await chromium.launch();const p=await b.newPage();
const errs=[];p.on('pageerror',e=>errs.push(e.message));
await p.goto('http://127.0.0.1:8777/zktls.html',{waitUntil:'networkidle'});
await p.waitForTimeout(1000);
console.log('cards rendered:', await p.$$eval('.demo',els=>els.length));
await p.click('#prove');
// wait up to 110s for the real proof + judge
await p.waitForFunction(()=>document.getElementById('jverdict')?.textContent.length>5,{timeout:110000}).catch(()=>{});
await p.waitForTimeout(1000);
console.log('STATUS  :', (await p.$eval('#status',e=>e.textContent).catch(()=>'')).slice(0,60));
console.log('RESULT  :', (await p.$eval('#result',e=>e.textContent).catch(()=>'')).replace(/\s+/g,' ').trim().slice(0,170));
console.log('ERRORS  :', errs.filter(e=>!/8899|8788|CONNECTION|favicon/.test(e)).join('|')||'none');
await p.screenshot({path:'/tmp/zktls.png',fullPage:true});
await b.close();
