import { chromium } from 'playwright';
const b=await chromium.launch();const p=await b.newPage();
await p.goto('http://127.0.0.1:8777/app.html');await p.waitForTimeout(1500);
for(const v of ['4.3.0','3.2.1','latest']){
  const r=await p.evaluate(async (v)=>{try{const m=await import(`https://esm.sh/@reclaimprotocol/js-sdk@${v}`);return Object.keys(m).filter(k=>/Reclaim|Proof/i.test(k)).join(',')||Object.keys(m).slice(0,6).join(',');}catch(e){return 'ERR '+e.message.slice(0,70);}},v);
  console.log(v,'->',r.slice(0,160));
}
await b.close();
