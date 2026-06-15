import { chromium } from 'playwright';
const b=await chromium.launch();const p=await b.newPage();
await p.goto('http://127.0.0.1:8777/app.html');await p.waitForSelector('.mcard',{timeout:15000});
const r=await p.evaluate(async ()=>{
  const { SuiClient } = await import('https://esm.sh/@mysten/sui@1.36.0/client');
  const { Transaction } = await import('https://esm.sh/@mysten/sui@1.36.0/transactions');
  const client=new SuiClient({url:'https://fullnode.devnet.sui.io:443'});
  const d=await (await fetch('./dispute.json')).json(); const v=d.verify;
  const hex=(h)=>{h=(h||'0x').replace(/^0x/,'');const a=[];for(let i=0;i<h.length;i+=2)a.push(parseInt(h.substr(i,2),16));return a;};
  try{const tx=new Transaction();tx.setSender(d.challenger);
    tx.moveCall({target:`${d.package}::dispute::verify_step`,arguments:[
      tx.object(d.fact),tx.pure.vector('u8',hex(v.regs)),tx.pure.vector('u8',hex(v.mem_root)),tx.pure.vector('u8',hex(v.instr)),
      tx.pure.vector('vector<u8>',v.instr_sibs.map(hex)),
      tx.pure.vector('u8',hex(v.page_a)),tx.pure.vector('vector<u8>',v.sibs_a.map(hex)),
      tx.pure.vector('u8',hex(v.page_b)),tx.pure.vector('vector<u8>',v.sibs_b.map(hex)),
      tx.pure.vector('u8',hex(v.page_w)),tx.pure.vector('vector<u8>',v.sibs_w.map(hex))]});
    await tx.build({client});return 'BUILD OK';}catch(e){return 'ERR '+e.message.slice(0,120);}
});
console.log('verify_step',r);await b.close();
