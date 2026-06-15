// Full guided lifecycle in the real dApp with a real wallet: connect → buy ⚡ →
// add a real source's zkTLS proof → AI judge → resolve → redeem → convict fraud.
// Run against a FAST-staged ⚡ (T_TRADE=30000 T_WIN=30000 python3 demo/judge_setup.py).
import { chromium } from 'playwright';
const b=await chromium.launch();const p=await b.newPage({viewport:{width:1320,height:1500}});
const logs=[];p.on('pageerror',e=>logs.push('PAGEERR '+e.message));
const wait=ms=>p.waitForTimeout(ms);
await p.goto('http://127.0.0.1:8777/app.html');
await p.waitForSelector('.mcard.live',{timeout:20000});
const kp=await p.evaluate(async ()=>{const { Ed25519Keypair } = await import('https://esm.sh/@mysten/sui@1.36.0/keypairs/ed25519');const k=new Ed25519Keypair();window.__sk=k.getSecretKey();return { address:k.getPublicKey().toSuiAddress(), sk:window.__sk };});
console.log('addr', kp.address);
await fetch('https://faucet.devnet.sui.io/v2/gas',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({FixedAmountRequest:{recipient:kp.address}})}).then(r=>r.text());
await wait(9000);
await p.evaluate(async (sk)=>{const { Ed25519Keypair } = await import('https://esm.sh/@mysten/sui@1.36.0/keypairs/ed25519');const { SuiClient } = await import('https://esm.sh/@mysten/sui@1.36.0/client');const { registerWallet } = await import('https://esm.sh/@mysten/wallet-standard@0.20.3');const key=Ed25519Keypair.fromSecretKey(sk);const addr=key.getPublicKey().toSuiAddress();const client=new SuiClient({url:'https://fullnode.devnet.sui.io:443'});const account={address:addr,publicKey:key.getPublicKey().toRawBytes(),chains:['sui:devnet'],features:['sui:signAndExecuteTransaction']};const wallet={version:'1.0.0',name:'TestWallet',icon:'data:image/svg+xml;base64,PHN2Zz48L3N2Zz4=',chains:['sui:devnet'],accounts:[account],features:{'standard:connect':{version:'1.0.0',connect:async()=>({accounts:[account]})},'standard:events':{version:'1.0.0',on:()=>()=>{}},'sui:signAndExecuteTransaction':{version:'2.0.0',signAndExecuteTransaction:async({transaction,account})=>{transaction.setSenderIfNotSet(account.address);const bytes=await transaction.build({client});const {signature}=await key.signTransaction(bytes);const res=await client.executeTransactionBlock({transactionBlock:bytes,signature,options:{showEffects:true}});return {digest:res.digest};}}}};registerWallet(wallet);}, kp.sk);

const tourN=async()=>p.evaluate(()=>Object.keys(window.tour?.done||{}).length);
const openLive=async()=>{await p.evaluate(()=>{const lm=CFG.live_market;openMarket(lm.id,'live');});await wait(1200);};
const liveId=await p.evaluate(()=>CFG.live_market.id);

// 1 connect
await p.click('#walletBtn'); await wait(1500); console.log('after connect, tour done =', await tourN());
// 2 buy ⚡
await openLive();
if(await p.evaluate(()=>!!document.getElementById('buyBtn'))){ await p.click('#buyBtn'); await wait(11000); console.log('BUY:', await p.evaluate(()=>document.getElementById('toast')?.innerText||'')); }
else console.log('⚡ not in trading — buy skipped');
// 3 add evidence (poll until evidence phase, then click first Add)
let added=false;
for(let i=0;i<26 && !added;i++){ await openLive();
  const addBtn=await p.$('[data-add]');
  if(addBtn){ await addBtn.click(); await wait(11000); console.log('ADD EVIDENCE:', await p.evaluate(()=>document.getElementById('toast')?.innerText||'')); added=true; }
  else { await p.evaluate(()=>document.getElementById('closeX')?.click()); await wait(5000); }
}
// 4 AI judge
await openLive();
if(await p.$('#runJudgeBtn')){ await p.click('#runJudgeBtn'); await p.waitForFunction(()=>document.getElementById('judgeVerdict')?.innerText.includes('Verdict'),{timeout:60000}).catch(()=>{}); console.log('JUDGE:', await p.evaluate(()=>document.getElementById('judgeVerdict')?.innerText.replace(/\n/g,' ')||'')); }
// 5 resolve (poll until resolvable)
let resolved=false;
for(let i=0;i<26 && !resolved;i++){ await openLive();
  if(await p.$('#resolveEv')){ await p.click('#resolveEv'); await wait(13000); console.log('RESOLVE:', await p.evaluate(()=>document.getElementById('toast')?.innerText||'')); resolved=true; }
  else { await p.evaluate(()=>document.getElementById('closeX')?.click()); await wait(5000); }
}
// 6 redeem
await openLive();
if(await p.$('#redeemBtn')){ await p.click('#redeemBtn'); await wait(12000); console.log('REDEEM:', await p.evaluate(()=>document.getElementById('toast')?.innerText||'')); }
else console.log('no redeem button (outcome/position?)');
// 7 fraud
await p.evaluate(()=>document.getElementById('closeX')?.click()); await wait(800);
await p.evaluate(()=>openFraud()); await wait(1500);
if(await p.$('#convictBtn')){ await p.click('#convictBtn'); await wait(13000); console.log('CONVICT:', await p.evaluate(()=>document.getElementById('toast')?.innerText||'')); }

const done=await p.evaluate(()=>window.tour?.done||{});
console.log('TOUR DONE:', JSON.stringify(done), '=', Object.keys(done).length+'/7');
console.log('ERRORS:', logs.join(' || ')||'none');
await b.close();
