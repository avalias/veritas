import { chromium } from 'playwright';
const b=await chromium.launch();const p=await b.newPage({viewport:{width:390,height:1500}});
const e=[];p.on('pageerror',x=>e.push(x.message));
await p.goto('http://127.0.0.1:8777/app.html');await p.waitForSelector('.mcard',{timeout:15000});await p.waitForTimeout(2500);
await p.screenshot({path:'/tmp/mobile.png',fullPage:true});
// check nav doesn't overflow
const navOverflow=await p.evaluate(()=>{const n=document.querySelector('nav .wrap');return n.scrollWidth>n.clientWidth;});
console.log('nav overflow:',navOverflow,'errors:',e.join('|')||'none');
await b.close();
