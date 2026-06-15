import { chromium } from 'playwright';
const b=await chromium.launch();const p=await b.newPage({viewport:{width:1280,height:1400}});
const logs=[];p.on('pageerror',e=>logs.push('PAGEERR '+e.message));p.on('console',m=>{if(m.type()==='error')logs.push('CONSOLEERR '+m.text());});
await p.goto('http://127.0.0.1:8777/app.html');
await p.waitForSelector('.mcard',{timeout:20000});
await p.waitForTimeout(2500);
// the two live cards present?
const cards=await p.evaluate(()=>[...document.querySelectorAll('.mcard.live')].map(el=>({
  cat:el.querySelector('.cat')?.innerText, q:el.querySelector('.q')?.innerText, ph:el.querySelector('.phase')?.innerText})));
console.log('LIVE CARDS:', JSON.stringify(cards,null,1));
await p.screenshot({path:'/tmp/grid.png'});
// open the ⚡ live card (first live card)
await p.locator('.mcard.live').first().click(); await p.waitForTimeout(1500);
const banner=await p.evaluate(()=>{const b=document.getElementById('lbanner');return b?b.innerText.replace(/\n/g,' | '):'(no banner)';});
console.log('⚡ BANNER:', banner);
const hasBuy=await p.evaluate(()=>!!document.getElementById('buyBtn'));
console.log('⚡ shows buy block:', hasBuy);
await p.screenshot({path:'/tmp/live_modal.png'});
// run the AI judge on the ⚡ card
await p.click('#runJudgeBtn');
await p.waitForFunction(()=>{const v=document.getElementById('judgeVerdict');return v&&v.innerText.includes('Verdict');},{timeout:60000}).catch(()=>{});
const judgeOut=await p.evaluate(()=>document.getElementById('judgeOut')?.innerText||'');
const verdict=await p.evaluate(()=>document.getElementById('judgeVerdict')?.innerText||'');
console.log('QWEN STREAM:', JSON.stringify(judgeOut));
console.log('VERDICT:', verdict.replace(/\n/g,' '));
await p.screenshot({path:'/tmp/live_judge.png'});
// close, open the ⚖️ READY card (second live card) — should show Resolve
await p.evaluate(()=>document.getElementById('closeX')?.click()); await p.waitForTimeout(800);
await p.locator('.mcard.live').nth(1).click(); await p.waitForTimeout(1500);
const ready=await p.evaluate(()=>({banner:document.getElementById('lbanner')?.innerText.replace(/\n/g,' | '),resolveBtn:!!document.getElementById('resolveEv')}));
console.log('⚖️ READY:', JSON.stringify(ready));
await p.screenshot({path:'/tmp/ready_modal.png'});
console.log('ERRORS:', logs.join(' || ')||'none');
await b.close();
