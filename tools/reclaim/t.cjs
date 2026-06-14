const nc=require('node:crypto');
try{ const core=require('@reclaimprotocol/attestor-core'); console.log('CJS load OK, has createClaim:', typeof core.createClaimOnAttestor);}catch(e){console.log('CJS err:', e.message.slice(0,140));}
