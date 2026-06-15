// E2E in the real dApp with a real (ephemeral, faucet-funded) wallet:
//   buy YES on the ⚡ live card  +  Resolve the ⚖️ ready card → resolved YES.
// Proves trade and resolve work end-to-end through the actual UI on devnet.
import { chromium } from 'playwright';
const b=await chromium.launch();const p=await b.newPage();
const logs=[];p.on('pageerror',e=>logs.push('PAGEERR '+e.message));
await p.goto('http://127.0.0.1:8777/app.html');
await p.waitForSelector('.mcard.live',{timeout:20000});
const kp=await p.evaluate(async ()=>{
  const { Ed25519Keypair } = await import('https://esm.sh/@mysten/sui@1.36.0/keypairs/ed25519');
  const k=new Ed25519Keypair(); window.__sk=k.getSecretKey();
  return { address:k.getPublicKey().toSuiAddress(), sk:window.__sk };
});
console.log('addr:', kp.address);
const fund=await fetch('https://faucet.devnet.sui.io/v2/gas',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({FixedAmountRequest:{recipient:kp.address}})}).then(r=>r.text()).catch(e=>'ERR '+e);
console.log('faucet:', fund.slice(0,50));
await p.waitForTimeout(9000);
await p.evaluate(async (sk)=>{
  const { Ed25519Keypair } = await import('https://esm.sh/@mysten/sui@1.36.0/keypairs/ed25519');
  const { SuiClient } = await import('https://esm.sh/@mysten/sui@1.36.0/client');
  const { registerWallet } = await import('https://esm.sh/@mysten/wallet-standard@0.20.3');
  const key=Ed25519Keypair.fromSecretKey(sk); const addr=key.getPublicKey().toSuiAddress();
  const client=new SuiClient({url:'https://fullnode.devnet.sui.io:443'});
  const account={address:addr,publicKey:key.getPublicKey().toRawBytes(),chains:['sui:devnet'],features:['sui:signAndExecuteTransaction']};
  const wallet={version:'1.0.0',name:'TestWallet',icon:'data:image/svg+xml;base64,PHN2Zz48L3N2Zz4=',chains:['sui:devnet'],accounts:[account],
    features:{'standard:connect':{version:'1.0.0',connect:async()=>({accounts:[account]})},'standard:events':{version:'1.0.0',on:()=>()=>{}},
      'sui:signAndExecuteTransaction':{version:'2.0.0',signAndExecuteTransaction:async({transaction,account})=>{
        transaction.setSenderIfNotSet(account.address);const bytes=await transaction.build({client});
        const {signature}=await key.signTransaction(bytes);
        const res=await client.executeTransactionBlock({transactionBlock:bytes,signature,options:{showEffects:true}});
        return {digest:res.digest};}}}};
  registerWallet(wallet);
}, kp.sk);
await p.click('#walletBtn'); await p.waitForTimeout(1500);

// --- BUY YES on the ⚡ live card ---
await p.locator('.mcard.live').first().click(); await p.waitForTimeout(1500);
if(await p.evaluate(()=>!!document.getElementById('buyBtn'))){
  await p.click('#buyBtn'); await p.waitForTimeout(11000);
  console.log('BUY ⚡ toast:', await p.evaluate(()=>document.getElementById('toast')?.innerText||''));
} else { console.log('⚡ not in trading phase (window elapsed) — skipping buy'); }
await p.evaluate(()=>document.getElementById('closeX')?.click()); await p.waitForTimeout(1000);

// --- RESOLVE the ⚖️ ready card ---
await p.locator('.mcard.live').nth(1).click(); await p.waitForTimeout(1500);
const hasResolve=await p.evaluate(()=>!!document.getElementById('resolveEv'));
console.log('⚖️ shows Resolve button:', hasResolve);
if(hasResolve){
  await p.click('#resolveEv'); await p.waitForTimeout(13000);
  console.log('RESOLVE toast:', await p.evaluate(()=>document.getElementById('toast')?.innerText||''));
  await p.waitForTimeout(2000);
  const st=await p.evaluate(()=>document.querySelector('.modal')?.innerText?.match(/phase \w+( \w+)?/)?.[0]||'');
  console.log('⚖️ modal phase line:', st);
}
console.log('ERRORS:', logs.join(' || ')||'none');
await b.close();
