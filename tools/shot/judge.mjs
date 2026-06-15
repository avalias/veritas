import { chromium } from 'playwright';
const b=await chromium.launch();const p=await b.newPage({viewport:{width:720,height:1100}});
const e=[];p.on('pageerror',x=>e.push(x.message));
await p.goto('http://127.0.0.1:8777/app.html');await p.waitForSelector('.mcard',{timeout:15000});await p.waitForTimeout(2500);
// open the evidence market (has 'evidence window')
await p.locator('.mcard',{hasText:'as reported by the wires'}).first().click();await p.waitForTimeout(1200);
await p.click('#runJudgeBtn');
// wait for streaming + verdict
await p.waitForFunction(()=>document.querySelector('#judgeVerdict')?.innerText?.includes('Verdict'),{timeout:60000}).catch(()=>{});
await p.waitForTimeout(800);
await p.screenshot({path:'/tmp/judge.png'});
const out=await p.evaluate(()=>document.getElementById('judgeOut')?.innerText||'');
const verdict=await p.evaluate(()=>document.getElementById('judgeVerdict')?.innerText||'');
console.log('QWEN OUTPUT:',JSON.stringify(out.slice(0,200)));
console.log('VERDICT:',verdict.replace(/\n/g,' '));
console.log('errors:',e.join('|')||'none');
await b.close();
