import { chromium } from 'playwright';
const b=await chromium.launch();const p=await b.newPage();
await p.goto('http://127.0.0.1:8777/app.html');await p.waitForSelector('.mcard',{timeout:15000});
const r=await p.evaluate(async ()=>{
  const { SuiClient } = await import('https://esm.sh/@mysten/sui@1.36.0/client');
  const { Transaction } = await import('https://esm.sh/@mysten/sui@1.36.0/transactions');
  const { bcs } = await import('https://esm.sh/@mysten/sui@1.36.0/bcs');
  const client=new SuiClient({url:'https://fullnode.devnet.sui.io:443'});
  const CFG=await (await fetch('./markets.json')).json(); const PKG=CFG.package, sender='0xb01503bef9a3acaab095a9269d21a5a8def0069478d5d4f8c5fbc6b0a4a650c9';
  const hex=(h)=>{h=h.replace(/^0x/,'');const a=[];for(let i=0;i<h.length;i+=2)a.push(parseInt(h.substr(i,2),16));return a;};
  const str=(s)=>Array.from(new TextEncoder().encode(s));
  const att=CFG.attestors.map(hex);
  const out={};
  // try tx.pure.vector('vector<u8>', ...)
  try{const tx=new Transaction();tx.setSender(sender);const[c]=tx.splitCoins(tx.gas,[500000000]);
    tx.moveCall({target:`${PKG}::market::create_market_entry`,arguments:[
      tx.pure.vector('u8',str('Will X happen by 2027?')),tx.pure.vector('u8',hex('0x'+'a7'.repeat(32))),tx.pure.u8(12),
      tx.pure.vector('vector<u8>',att),tx.pure.vector('u8',[3,3,3]),tx.pure.vector('u64',[0,1,2]),
      tx.pure.u64(2),tx.pure.u8(0),tx.pure.u64(Date.now()+2592000000),tx.pure.u64(604800000),tx.pure.u64(100),c,tx.object('0x6')]});
    await tx.build({client});out.method1='OK';}catch(e){out.method1='ERR '+e.message.slice(0,90);}
  return out;
});
console.log(JSON.stringify(r));await b.close();
