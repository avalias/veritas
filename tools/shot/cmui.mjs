import { chromium } from 'playwright';
const b=await chromium.launch();const p=await b.newPage({viewport:{width:700,height:900}});
const e=[];p.on('pageerror',x=>e.push(x.message));
await p.goto('http://127.0.0.1:8777/app.html');await p.waitForSelector('.mcard',{timeout:15000});await p.waitForTimeout(1500);
await p.click('#createBtn');await p.waitForTimeout(800);
await p.screenshot({path:'/tmp/create.png'});
console.log('errors:',e.join('|')||'none','| modal has question input:',await p.locator('#cmQ').count());
await b.close();
