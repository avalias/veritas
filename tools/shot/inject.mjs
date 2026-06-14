import { chromium } from 'playwright';
const b=await chromium.launch();const p=await b.newPage();
const logs=[];p.on('pageerror',e=>logs.push('PAGEERR '+e.message));
await p.goto('http://127.0.0.1:8777/app.html');
await p.waitForSelector('.mcard',{timeout:20000});
// 1. generate an ephemeral keypair in-page
const kp=await p.evaluate(async ()=>{
  const { Ed25519Keypair } = await import('https://esm.sh/@mysten/sui@1.36.0/keypairs/ed25519');
  const k=new Ed25519Keypair();
  window.__sk=k.getSecretKey(); // bech32 suiprivkey
  return { address:k.getPublicKey().toSuiAddress(), sk:window.__sk };
});
console.log('ephemeral addr:', kp.address);
// 2. fund it from devnet faucet (node side)
const fund=await fetch('https://faucet.devnet.sui.io/v2/gas',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({FixedAmountRequest:{recipient:kp.address}})}).then(r=>r.text()).catch(e=>'ERR '+e);
console.log('faucet:', fund.slice(0,60));
await p.waitForTimeout(8000); // wait for coins
// 3. register a wallet-standard wallet backed by the keypair + drive Connect→Buy
const result=await p.evaluate(async (sk)=>{
  const { Ed25519Keypair } = await import('https://esm.sh/@mysten/sui@1.36.0/keypairs/ed25519');
  const { SuiClient } = await import('https://esm.sh/@mysten/sui@1.36.0/client');
  const { registerWallet } = await import('https://esm.sh/@mysten/wallet-standard@0.20.3');
  const key=Ed25519Keypair.fromSecretKey(sk);
  const addr=key.getPublicKey().toSuiAddress();
  const client=new SuiClient({url:'https://fullnode.devnet.sui.io:443'});
  const account={address:addr,publicKey:key.getPublicKey().toRawBytes(),chains:['sui:devnet'],features:['sui:signAndExecuteTransaction']};
  const wallet={version:'1.0.0',name:'TestWallet',icon:'data:image/svg+xml;base64,PHN2Zz48L3N2Zz4=',chains:['sui:devnet'],accounts:[account],
    features:{
      'standard:connect':{version:'1.0.0',connect:async()=>({accounts:[account]})},
      'standard:events':{version:'1.0.0',on:()=>()=>{}},
      'sui:signAndExecuteTransaction':{version:'2.0.0',signAndExecuteTransaction:async({transaction,account,chain})=>{
        transaction.setSenderIfNotSet(account.address);
        const bytes=await transaction.build({client});
        const {signature}=await key.signTransaction(bytes);
        const res=await client.executeTransactionBlock({transactionBlock:bytes,signature,options:{showEffects:true}});
        return {digest:res.digest};
      }}
    }};
  registerWallet(wallet);
  return {ok:true,addr};
}, kp.sk);
console.log('registered wallet for', result.addr);
// click Connect, then a market, then Buy
await p.click('#walletBtn'); await p.waitForTimeout(1500);
await p.locator('.mcard').first().click(); await p.waitForTimeout(1500);
const mid=await p.evaluate(()=>document.querySelector('.modal')?.outerHTML?.match(/0x[0-9a-f]{64}/)?.[0]);
await p.click('#buyBtn'); await p.waitForTimeout(9000);
const toast=await p.evaluate(()=>document.getElementById('toast')?.innerText||'');
console.log('TOAST AFTER BUY:', toast);
console.log('errors:', logs.join('|')||'none');
await b.close();
