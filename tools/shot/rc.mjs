import { chromium } from 'playwright';
const b=await chromium.launch();const p=await b.newPage({viewport:{width:720,height:1000}});
const e=[];p.on('pageerror',x=>e.push(x.message));
await p.goto('http://127.0.0.1:8777/app.html');await p.waitForSelector('.mcard',{timeout:15000});await p.waitForTimeout(2500);
const n=await p.locator('.mcard').count();
// click the reclaim market (has 'prove it yourself' category)
const card=p.locator('.mcard',{hasText:'prove it yourself'});
await card.first().click();await p.waitForTimeout(1200);
await p.screenshot({path:'/tmp/reclaim.png'});
console.log('cards:',n,'| reclaim btn:',await p.locator('#reclaimBtn').count(),'| errors:',e.join('|')||'none');
await b.close();
