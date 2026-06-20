// veritas.js: the small shared layer every demo page uses.
//
// It loads the deployed addresses, connects a Sui wallet, signs one
// transaction, and reads on-chain state. Each demo page imports what it needs
// and adds only the one thing it is trying to show. Keep it boring; the
// interesting code lives in the demo pages.

import { SuiClient } from 'https://esm.sh/@mysten/sui@1.36.0/client';
import { Transaction } from 'https://esm.sh/@mysten/sui@1.36.0/transactions';
import { getWallets } from 'https://esm.sh/@mysten/wallet-standard@0.20.3';
export { Transaction };

// config.json carries the network + the two deployed package ids.
export const CFG = await (await fetch('./config.json')).json();
export const client = new SuiClient({ url: CFG.rpc });
export const NET = CFG.network;                 // "testnet" / "devnet" / "mainnet"
export const PKG = CFG.package;                 // veritas market package
export const OPML = CFG.opml_package || PKG;    // opml verifier package (fraud proofs)
export const CLOCK = '0x6';
export const CHAIN = `sui:${NET}`;

export const scanTx = (d) => `https://suiscan.xyz/${NET}/tx/${d}`;
export const scanObj = (o) => `https://suiscan.xyz/${NET}/object/${o}`;

export const hexToBytes = (h) => {
  h = h.replace(/^0x/, '');
  const a = new Uint8Array(h.length / 2);
  for (let i = 0; i < a.length; i++) a[i] = parseInt(h.substr(i * 2, 2), 16);
  return a;
};
export const bytesToStr = (arr) => { try { return new TextDecoder().decode(new Uint8Array(arr)); } catch { return ''; } };

// ---- a tiny toast, the only shared UI ----
export function toast(msg, ok = true) {
  let t = document.getElementById('toast');
  if (!t) { t = document.createElement('div'); t.id = 'toast'; t.className = 'toast'; document.body.appendChild(t); }
  t.innerHTML = msg;
  t.style.display = 'block';
  t.style.borderColor = ok ? 'var(--ok)' : 'var(--bad)';
  clearTimeout(t._h);
  t._h = setTimeout(() => { t.style.display = 'none'; }, 8000);
}

// ---- wallet ----
let wallet = null, account = null;
export const getAccount = () => account;

function findWallet() {
  return getWallets().get().find(w =>
    w.features['standard:connect'] &&
    (w.features['sui:signAndExecuteTransaction'] || w.features['sui:signAndExecuteTransactionBlock']));
}

export async function connect() {
  wallet = findWallet();
  if (!wallet) {
    toast(`No Sui wallet found. Install <a href="https://slush.app" target="_blank">Slush</a> and set it to ${NET}.`, false);
    return null;
  }
  try {
    const r = await wallet.features['standard:connect'].connect();
    account = (r.accounts && r.accounts[0]) || wallet.accounts[0];
    document.dispatchEvent(new CustomEvent('wallet', { detail: account }));
    toast(`Connected ${wallet.name}. Make sure it is on <b>${NET}</b>.`);
    return account;
  } catch (e) { toast('Connect failed: ' + e.message, false); return null; }
}

// Sign and run one transaction. Returns the result (with .digest) or null.
// A success toast links straight to the transaction on suiscan.
export async function sign(tx, label) {
  if (!account && !(await connect())) return null;
  tx.setSenderIfNotSet(account.address);
  try {
    let res;
    if (wallet.features['sui:signAndExecuteTransaction']) {
      res = await wallet.features['sui:signAndExecuteTransaction'].signAndExecuteTransaction({ transaction: tx, account, chain: CHAIN });
    } else {
      await tx.build({ client });
      res = await wallet.features['sui:signAndExecuteTransactionBlock'].signAndExecuteTransactionBlock({ transactionBlock: tx, account, chain: CHAIN });
    }
    const d = res.digest || res.effectsDigest;
    toast(`${label} ✓ <a href="${scanTx(d)}" target="_blank">view tx</a>`);
    return res;
  } catch (e) { toast(`${label} failed: ` + (e.message || e), false); return null; }
}

// ---- reads ----
export async function loadMarket(id) {
  const o = await client.getObject({ id, options: { showContent: true } });
  const f = o.data.content.fields;
  const ry = Number(f.reserve_yes), rn = Number(f.reserve_no);
  const ra = Number(f.resolve_after_ms), win = Number(f.evidence_window_ms), now = Date.now(), stored = Number(f.phase);
  // effective phase from the clock: 0 trading, 1 evidence, 2 resolved, 3 window closed
  const phase = stored === 2 ? 2 : now < ra ? 0 : now < ra + win ? 1 : 3;
  return {
    id, f, ry, rn, stored, phase, outcome: Number(f.outcome),
    priceYes: Math.round(rn / (ry + rn) * 100),
    question: bytesToStr(f.question),
    evidence: f.evidence || [],
    collateral: Number(f.collateral),
  };
}

export async function myPosition(marketId) {
  if (!account) return null;
  try {
    const tx = new Transaction();
    tx.moveCall({ target: `${PKG}::market::position_of`, arguments: [tx.object(marketId), tx.pure.address(account.address)] });
    const r = await client.devInspectTransactionBlock({ transactionBlock: tx, sender: account.address });
    const rv = r.results?.[0]?.returnValues; if (!rv) return null;
    const u64 = (b) => { let n = 0n; for (let i = b.length - 1; i >= 0; i--) n = (n << 8n) | BigInt(b[i]); return Number(n); };
    return { yes: u64(rv[0][0]), no: u64(rv[1][0]), paid: u64(rv[2][0]) };
  } catch { return null; }
}

// ---- the market transactions, in one place so every demo calls the same code ----
export const tx = {
  buy(id, side, sui) {
    const t = new Transaction();
    const [coin] = t.splitCoins(t.gas, [Math.floor(sui * 1e9)]);
    t.moveCall({ target: `${PKG}::market::${side === 'yes' ? 'buy_yes' : 'buy_no'}`, arguments: [t.object(id), coin, t.object(CLOCK)] });
    return t;
  },
  // submit a Reclaim-format zkTLS proof; verified on-chain by native ecrecover
  submitProof(id, p) {
    const t = new Transaction();
    t.moveCall({
      target: `${PKG}::market::submit_web_proof`, arguments: [
        t.object(id), t.pure.u64(p.attestor_idx), t.pure.u8(p.claim),
        t.pure.vector('u8', hexToBytes(p.provider)), t.pure.vector('u8', hexToBytes(p.parameters)),
        t.pure.vector('u8', hexToBytes(p.context)), t.pure.vector('u8', hexToBytes(p.owner)),
        t.pure.u64(p.timestamp_s), t.pure.u64(p.epoch), t.pure.vector('u8', hexToBytes(p.signature)), t.object(CLOCK)],
    });
    return t;
  },
  resolve(id) {
    const t = new Transaction();
    t.moveCall({ target: `${PKG}::market::resolve`, arguments: [t.object(id), t.object(CLOCK)] });
    return t;
  },
  redeem(id) {
    const t = new Transaction();
    t.moveCall({ target: `${PKG}::market::redeem_to_sender`, arguments: [t.object(id)] });
    return t;
  },
  // re-run the single disputed micro-op on-chain. v is dispute.json's `verify` object.
  verifyStep(pkg, fact, v) {
    const vec = (a) => t.pure.vector('vector<u8>', a.map(h => Array.from(hexToBytes(h))));
    const b = (h) => t.pure.vector('u8', Array.from(hexToBytes(h)));
    const t = new Transaction();
    t.moveCall({
      target: `${pkg}::dispute::verify_step`, arguments: [
        t.object(fact), b(v.regs), b(v.mem_root), b(v.instr), vec(v.instr_sibs),
        b(v.page_a), vec(v.sibs_a), b(v.page_b), vec(v.sibs_b), b(v.page_w), vec(v.sibs_w)],
    });
    return t;
  },
};

// Stream the real Qwen judge reading a piece of evidence. The resolver sends the
// exact prompt first (onPrompt), then each token (onToken), and resolves to the
// verdict "YES" / "NO" / "UNKNOWN". It runs the model off-chain; the chain only
// re-runs one micro-op if the verdict is disputed. Returns null if the resolver
// is unreachable.
export function askJudge(question, evidence, { onPrompt, onToken } = {}) {
  // resolver URL: ?resolver= override; on a Tailscale-funnel host use the same-origin
  // /ai path on :443 (works behind firewalls that block odd ports); otherwise config / local :8899.
  const url = (new URLSearchParams(location.search).get('resolver')
    || (location.hostname.endsWith('.ts.net') ? location.origin + '/ai' : (CFG.resolver_url || 'http://127.0.0.1:8899')));
  return new Promise((resolve) => {
    let es;
    try { es = new EventSource(`${url}/judge?q=${encodeURIComponent(question)}&e=${encodeURIComponent(evidence)}`); }
    catch { resolve(null); return; }
    es.onmessage = (ev) => {
      try {
        const d = JSON.parse(ev.data);
        if (d.prompt && onPrompt) onPrompt(d.prompt);
        if (d.t && onToken) onToken(d.t);
        if (d.done) { es.close(); resolve(d.verdict || 'UNKNOWN'); }
      } catch { /* ignore keep-alives */ }
    };
    es.onerror = () => { es.close(); resolve(null); };
  });
}
