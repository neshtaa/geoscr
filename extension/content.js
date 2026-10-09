/*
 * Content script (isolated world) of the GeoScr extension on www.geoguessr.com. It bridges the
 * page-world part (page.js: hooks, capture, round controller, HUD) and the service worker (local
 * locator server, bundled hud.css): settings, predict, health, the auto-analyse switch.
 *
 * The bridge is a private MessagePort: the only window message is the one that hands it to page.js,
 * whose listener takes it before any page script runs. Requests are answered on single-player pages
 * only (and on replay pages of finished games in the replay test mode). Nothing is drawn here. The
 * replay switch is mirrored into localStorage only while it is on, so page.js knows it at
 * document_start of the next load.
 */
(() => {
  'use strict';
  const DEFAULTS = { server: 'http://localhost:8080', auto: false, replay: false };
  const ALLOWED = [/^\/challenge\/[A-Za-z0-9]+$/, /^\/game\/[A-Za-z0-9]+$/];
  const REPLAY = /^\/game\/[A-Za-z0-9]+\/replay$/;
  const TYPES = ['predict', 'health', 'set', 'hud-css'];
  const MIRROR = '__geoscr_cfg';
  let settings = null;

  const channel = new MessageChannel();
  const port = channel.port1;
  window.postMessage({ __geoscr: 'port' }, location.origin, [channel.port2]);

  const allowedHere = () => ALLOWED.some(re => re.test(location.pathname)) ||
    (!!(settings && settings.replay) && REPLAY.test(location.pathname));
  const forPage = s => ({ auto: !!s.auto, replay: !!s.replay });
  const alive = () => { try { return !!chrome.runtime.id; } catch (e) { return false; } };

  const bg = msg => new Promise(resolve => {
    if (!alive()) return resolve({ ok: false, error: 'Розширення оновлено: перезавантажте сторінку (F5).' });
    try {
      chrome.runtime.sendMessage(msg, r => {
        const err = chrome.runtime.lastError;
        resolve(err ? { ok: false, error: err.message } : r || { ok: false, error: 'немає відповіді' });
      });
    } catch (e) {
      resolve({ ok: false, error: 'Розширення оновлено: перезавантажте сторінку (F5).' });
    }
  });

  function mirror(s) {
    try {
      if (s.replay) {
        const v = JSON.stringify({ replay: true });
        if (localStorage.getItem(MIRROR) !== v) localStorage.setItem(MIRROR, v);
      } else if (localStorage.getItem(MIRROR) !== null) localStorage.removeItem(MIRROR);
    } catch (e) { /* storage blocked */ }
  }

  let helloSeen = false;
  function apply(s) {
    settings = Object.assign({}, DEFAULTS, s);
    mirror(settings);
    if (helloSeen) port.postMessage({ type: 'settings', payload: forPage(settings) });
  }

  chrome.storage.local.get(DEFAULTS, s => apply(s));
  chrome.storage.onChanged.addListener((ch, area) => {
    if (area !== 'local' || !settings) return;
    const s = Object.assign({}, settings);
    for (const k of Object.keys(ch)) if (k in DEFAULTS) s[k] = ch[k].newValue;
    apply(s);
  });

  port.onmessage = async e => {
    const d = e.data;
    if (!d || typeof d.type !== 'string') return;
    if (d.type === 'hello') {
      helloSeen = true;
      if (settings) port.postMessage({ type: 'settings', payload: forPage(settings) });
      return;
    }
    if (!TYPES.includes(d.type) || typeof d.id !== 'string') return;
    const r = allowedHere() ? await bg({ type: d.type, payload: d.payload })
      : { ok: false, error: 'на цій сторінці локатор не працює' };
    port.postMessage({ type: 'reply', id: d.id, ok: !!r.ok, payload: r.ok ? r.payload : null, error: r.ok ? null : r.error });
  };
})();
