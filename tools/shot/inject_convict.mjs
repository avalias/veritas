import { chromium } from 'playwright';
const b=await chromium.launch();const p=await b.newPage();
const logs=[];p.on('pageerror',e=>logs.push('PE '+e.message));
await p.goto('http://127.0.0.1:8777/app.html');await p.waitForSelector('.mcard',{timeout:20000});
const kp=await p.evaluate(async ()=>{const { Ed25519Keypair }=await import('https://esm.sh/@mysten/sui@1.36.0/keypairs/ed25519');const k=new Ed25519Keypair();return {address:k.getPublicKey().toSuiAddress(),sk:k.getSecretKey()};});
const fund=await fetch('https://faucet.devnet.sui.io/v2/gas',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({FixedAmountRequest:{recipient:kp.address}})}).then(r=>r.text()).catch(e=>'ERR');
console.log('convictor:',kp.address.slice(0,12),'funded:',fund.includes('Success'));
await p.waitForTimeout(8000);
await p.evaluate(async (sk)=>{
  const { Ed25519Keypair }=await import('https://esm.sh/@mysten/sui@1.36.0/keypairs/ed25519');
  const { SuiClient }=await import('https://esm.sh/@mysten/sui@1.36.0/client');
  const { registerWallet }=await import('https://esm.sh/@mysten/wallet-standard@0.20.3');
  const key=Ed25519Keypair.fromSecretKey(sk);const addr=key.getPublicKey().toSuiAddress();
  const client=new SuiClient({url:'https://fullnode.devnet.sui.io:443'});
  const account={address:addr,publicKey:key.getPublicKey().toRawBytes(),chains:['sui:devnet'],features:['sui:signAndExecuteTransaction']};
  registerWallet({version:'1.0.0',name:'TestWallet',icon:'data:image/svg+xml;base64,PHN2Zz48L3N2Zz4=',chains:['sui:devnet'],accounts:[account],features:{
    'standard:connect':{version:'1.0.0',connect:async()=>({accounts:[account]})},'standard:events':{version:'1.0.0',on:()=>()=>{}},
    'sui:signAndExecuteTransaction':{version:'2.0.0',signAndExecuteTransaction:async({transaction,account})=>{transaction.setSenderIfNotSet(account.address);const bytes=await transaction.build({client});const {signature}=await key.signTransaction(bytes);const res=await client.executeTransactionBlock({transactionBlock:bytes,signature,options:{showEffects:true}});return {digest:res.digest};}}}});
},kp.sk);
await p.click('#walletBtn');await p.waitForTimeout(1200);
await p.click('#fraudBtn');await p.waitForTimeout(1500);
await p.click('#convictBtn');await p.waitForTimeout(11000);
const toast=await p.evaluate(()=>document.getElementById('toast')?.innerText||'');
const out=await p.evaluate(()=>document.getElementById('convictOut')?.innerText||'');
console.log('TOAST:',toast);console.log('OUT:',out.slice(0,90));console.log('errors:',logs.join('|')||'none');
await b.close();
