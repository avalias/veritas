import { chromium } from 'playwright';
const b=await chromium.launch();const p=await b.newPage();
await p.goto('http://127.0.0.1:8777/judge.html',{waitUntil:'networkidle'});
await p.waitForTimeout(1200);
async function ask(text){
  await p.fill('#custom', text);
  await p.click('#ask');
  await p.waitForFunction(()=>!document.getElementById('ask').disabled && document.getElementById('out').textContent.length>3,{timeout:40000}).catch(()=>{});
  await p.waitForTimeout(800);
  const out=await p.$eval('#out',e=>e.textContent).catch(()=>'');
  const v=await p.evaluate(()=>document.body.innerText.match(/Verdict:\s*\w+/)?.[0]||'');
  return {out:out.trim(), v};
}
const a=await ask('Reuters: the Starship launch was scrubbed and the rocket never left the ground');
console.log('INPUT A (scrubbed):', JSON.stringify(a.out.slice(0,110)), '|', a.v);
const c=await ask('BBC: SpaceX Starship reached orbit on its first attempt');
console.log('INPUT B (orbit):   ', JSON.stringify(c.out.slice(0,110)), '|', c.v);
console.log('DIFFERENT OUTPUTS:', a.out!==c.out ? 'YES — live model' : 'NO — suspicious');
await b.close();
