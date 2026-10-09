/*
 * Service worker of the GeoScr extension: the only part that talks to the local locator server
 * (default http://localhost:8080; WSL2 forwards localhost to Windows). Requests come from the content
 * script of www.geoguessr.com pages: predict (views + map settings), health, the bundled hud.css and
 * the auto-analyse switch. Clue-card images of the result are replaced by data: URLs from the
 * server's offline cache (/hud/img), so the page never loads an image from GeoGuessr. Nothing here
 * talks to GeoGuessr.
 */
'use strict';

const DEFAULTS = { server: 'http://localhost:8080', auto: false, replay: false };
const GG_IMAGE = /^https:\/\/www\.geoguessr\.com\//;

async function settings() {
  const s = await chrome.storage.local.get(Object.keys(DEFAULTS));
  return Object.assign({}, DEFAULTS, s);
}

// The server must be this machine or the local network (e.g. the WSL address); never a public host.
function localServer(url) {
  let u;
  try { u = new URL(url); } catch (e) { return null; }
  if (!/^https?:$/.test(u.protocol) || u.username || u.password) return null;
  const h = u.hostname;
  const ok = h === 'localhost' || h === '[::1]' || /^127\./.test(h) || /^10\./.test(h) || /^192\.168\./.test(h) ||
    /^172\.(1[6-9]|2\d|3[01])\./.test(h);
  return ok ? u.origin : null;
}

async function call(url, init, ms) {
  const ac = new AbortController();
  const t = setTimeout(() => ac.abort(), ms);
  try {
    return await fetch(url, Object.assign({ cache: 'no-store', credentials: 'omit', signal: ac.signal }, init));
  } catch (e) {
    const why = e && e.name === 'AbortError' ? 'немає відповіді' : 'недоступний';
    throw new Error(`Сервер локатора ${why} (${new URL(url).origin}). Запустіть у WSL: python3 web/server.py`);
  } finally {
    clearTimeout(t);
  }
}

async function json(resp) {
  let data = null;
  try { data = await resp.json(); } catch (e) { throw new Error(`Сервер відповів ${resp.status} без JSON`); }
  if (!resp.ok || data.error) throw new Error(data.error || `HTTP ${resp.status}`);
  return data;
}

// Only the fields the server takes; never anything else from the page.
function predictBody(p) {
  const views = Array.isArray(p && p.views) ? p.views : [];
  if (!views.length || views.length > 40) throw new Error('немає кадрів');
  const body = { views: views.map(v => {
    const img = String(v.image_b64 || '');
    if (!/^data:image\/(jpeg|png);base64,/.test(img)) throw new Error('кадр не є зображенням');
    return { image_b64: img, yaw: +v.yaw, pitch: +v.pitch, hfov: +v.hfov, vfov: +v.vfov };
  }) };
  const m = p.map;
  if (m && typeof m === 'object') {
    body.map = {};
    for (const k of ['id', 'slug', 'name', 'bounds', 'maxErrorDistance']) if (m[k] !== undefined && m[k] !== null) body.map[k] = m[k];
  }
  return body;
}

function base64(buf) {
  const b = new Uint8Array(buf);
  let s = '';
  for (let i = 0; i < b.length; i += 0x8000) s += String.fromCharCode.apply(null, b.subarray(i, i + 0x8000));
  return btoa(s);
}

// GeoGuessr clue images -> data: URLs from the server's cache (null when not cached).
async function inlineImages(result, srv) {
  const cards = [];
  for (const h of (result && result.hints) || []) for (const k of (h && h.geoguessr) || []) if (k && k.image_url) cards.push(k);
  await Promise.all(cards.map(async k => {
    const url = String(k.image_url);
    k.image_url = null;
    if (!GG_IMAGE.test(url)) return;
    try {
      const r = await call(`${srv}/hud/img?u=${encodeURIComponent(url)}`, {}, 5000);
      const type = (r.headers.get('content-type') || '').split(';')[0];
      if (r.ok && /^image\/(jpeg|png|webp)$/.test(type)) k.image_url = `data:${type};base64,${base64(await r.arrayBuffer())}`;
    } catch (e) { /* no image */ }
  }));
  return result;
}

async function handle(msg, sender) {
  const from = (sender && (sender.url || (sender.tab && sender.tab.url))) || '';
  if (sender.id !== chrome.runtime.id || !/^https:\/\/www\.geoguessr\.com\//.test(from)) throw new Error('чужий запит');
  const s = await settings();
  const srv = localServer(s.server);
  switch (msg && msg.type) {
    case 'hud-css': {
      const r = await fetch(chrome.runtime.getURL('hud/hud.css'));
      return r.text();
    }
    case 'set': {
      const p = msg.payload || {};
      if (typeof p.auto === 'boolean') await chrome.storage.local.set({ auto: p.auto });
      return { auto: (await settings()).auto };
    }
    case 'health':
    case 'predict':
      if (!srv) throw new Error(`Адреса сервера ${s.server} не локальна: вкажіть http://localhost:8080 у налаштуваннях розширення`);
      if (msg.type === 'health') return json(await call(srv + '/api/health', {}, 5000));
      {
        const body = JSON.stringify(predictBody(msg.payload));
        const result = await json(await call(srv + '/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body }, 120000));
        return inlineImages(result, srv);
      }
    default:
      throw new Error('невідомий запит');
  }
}

chrome.runtime.onMessage.addListener((msg, sender, reply) => {
  handle(msg, sender).then(payload => reply({ ok: true, payload }), e => reply({ ok: false, error: String(e && e.message || e) }));
  return true;
});

chrome.action.onClicked.addListener(() => chrome.runtime.openOptionsPage());
