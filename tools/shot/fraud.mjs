import { chromium } from 'playwright';
const b=await chromium.launch();const p=await b.newPage({viewport:{width:720,height:1000}});
const e=[];p.on('pageerror',x=>e.push(x.message));
await p.goto('http://127.0.0.1:8777/app.html');await p.waitForSelector('.mcard',{timeout:15000});await p.waitForTimeout(2000);
const bannerVisible=await p.locator('#fraudBanner').isVisible();
await p.click('#fraudBtn');await p.waitForTimeout(3000); // let bisection animate
await p.screenshot({path:'/tmp/fraud.png'});
console.log('banner visible:',bannerVisible,'| convict btn:',await p.locator('#convictBtn').count(),'| errors:',e.join('|')||'none');
await b.close();
