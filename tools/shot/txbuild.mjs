import { chromium } from 'playwright';
const b=await chromium.launch();const p=await b.newPage();
await p.goto('http://127.0.0.1:8777/app.html');
await p.waitForSelector('.mcard',{timeout:15000});
const r=await p.evaluate(async ()=>{
  const { SuiClient } = await import('https://esm.sh/@mysten/sui@1.36.0/client');
  const { Transaction } = await import('https://esm.sh/@mysten/sui@1.36.0/transactions');
  const client=new SuiClient({url:'https://fullnode.devnet.sui.io:443'});
  const CFG=await (await fetch('./markets.json')).json();
  const PKG=CFG.package, CLOCK='0x6', sender='0xb01503bef9a3acaab095a9269d21a5a8def0069478d5d4f8c5fbc6b0a4a650c9';
  const hex=(h)=>{h=h.replace(/^0x/,'');const a=[];for(let i=0;i<h.length;i+=2)a.push(parseInt(h.substr(i,2),16));return a;};
  const out={};
  // 1. buy_yes builds?
  try{const tx=new Transaction();tx.setSender(sender);const[c]=tx.splitCoins(tx.gas,[50000000]);
    tx.moveCall({target:`${PKG}::market::buy_yes`,arguments:[tx.object(CFG.markets[0].id),c,tx.object(CLOCK)]});
    await tx.build({client});out.buy='OK';}catch(e){out.buy='ERR '+e.message.slice(0,80);}
  // 2. submit_web_proof builds?
  try{const p=CFG.evidence_market.proof;const tx=new Transaction();tx.setSender(sender);
    tx.moveCall({target:`${PKG}::market::submit_web_proof`,arguments:[tx.object(CFG.evidence_market.id),
      tx.pure.u64(p.attestor_idx),tx.pure.u8(p.claim),tx.pure.vector('u8',hex(p.provider)),tx.pure.vector('u8',hex(p.parameters)),
      tx.pure.vector('u8',hex(p.context)),tx.pure.vector('u8',hex(p.owner)),tx.pure.u64(p.timestamp_s),tx.pure.u64(p.epoch),
      tx.pure.vector('u8',hex(p.signature)),tx.object(CLOCK)]});
    await tx.build({client});out.submit='OK';}catch(e){out.submit='ERR '+e.message.slice(0,80);}
  // 3. resolve builds?
  try{const tx=new Transaction();tx.setSender(sender);tx.moveCall({target:`${PKG}::market::resolve`,arguments:[tx.object(CFG.evidence_market.id),tx.object(CLOCK)]});await tx.build({client});out.resolve='OK';}catch(e){out.resolve='ERR '+e.message.slice(0,80);}
  return out;
});
console.log(JSON.stringify(r));
await b.close();
