import { chromium } from 'playwright';
const b=await chromium.launch();const p=await b.newPage();
const errs=[];p.on('pageerror',e=>errs.push(e.message));p.on('console',m=>{if(m.type()==='error')errs.push('C:'+m.text());});
await p.goto('http://127.0.0.1:8777/evidence.html',{waitUntil:'networkidle'});
await p.waitForTimeout(2500);
console.log('REAL proof recovery:', (await p.$eval('#recovered',e=>e.textContent).catch(()=>'')).replace(/\n/g,' ').slice(0,120));
// tamper the claim
await p.fill('#claim','SpaceX Starship blew up and never reached orbit');
await p.click('#forge'); await p.waitForTimeout(600);
console.log('FORGED recovery   :', (await p.$eval('#recovered',e=>e.textContent).catch(()=>'')).replace(/\n/g,' ').slice(0,150));
console.log('opinion refusal works:', await p.evaluate(()=>{document.getElementById('vote').click(); return !!document.querySelector('#refused .panel.bad');}));
console.log('ERRORS:', errs.filter(e=>!/8899|CONNECTION|favicon/.test(e)).join('|')||'none');
await p.screenshot({path:'/tmp/evidence2.png',fullPage:true});
await b.close();
