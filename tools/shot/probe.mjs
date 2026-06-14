import { chromium } from 'playwright';
const b = await chromium.launch();
const p = await b.newPage();
await p.goto('http://127.0.0.1:8777/app.html');
for (const v of ['2.17.0','1.36.0']) {
  const r = await p.evaluate(async (v)=>{
    try { const m = await import(`https://esm.sh/@mysten/sui@${v}/client`); return Object.keys(m).join(','); }
    catch(e){ return 'ERR '+e.message.slice(0,90); }
  }, v);
  console.log(v, '->', r.slice(0,300));
}
const wc = await p.evaluate(async ()=>{ try{ const m=await import('https://esm.sh/@mysten/wallet-standard@0.20.3'); return Object.keys(m).join(','); }catch(e){return 'ERR '+e.message.slice(0,80);} });
console.log('wallet-standard ->', wc.slice(0,200));
await b.close();
