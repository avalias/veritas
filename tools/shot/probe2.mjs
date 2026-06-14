import { chromium } from 'playwright';
const b = await chromium.launch(); const p = await b.newPage();
await p.goto('http://127.0.0.1:8777/app.html');
console.log('tx@1.36:', (await p.evaluate(async()=>{try{const m=await import('https://esm.sh/@mysten/sui@1.36.0/transactions');return Object.keys(m).filter(k=>/Transaction/.test(k)).join(',');}catch(e){return 'ERR '+e.message.slice(0,60);}})));
console.log('getWallets@0.20.3:', (await p.evaluate(async()=>{try{const m=await import('https://esm.sh/@mysten/wallet-standard@0.20.3');return ['getWallets','registerWallet'].filter(k=>k in m).join(',')||'NONE; has:'+Object.keys(m).filter(k=>/allet/i.test(k)).join(',');}catch(e){return 'ERR';}})));
await b.close();
