'use strict';
const DEFAULTS = { server: 'http://localhost:8080', auto: false, replay: false };
const $ = id => document.getElementById(id);

// This machine or the local network (e.g. the WSL address) only; background.js checks it again.
function origin(url) {
  const u = new URL(url);
  if (u.protocol !== 'http:' || u.username || u.password) throw new Error('потрібна адреса http://');
  const h = u.hostname;
  if (!(h === 'localhost' || h === '[::1]' || /^127\./.test(h) || /^10\./.test(h) || /^192\.168\./.test(h) ||
    /^172\.(1[6-9]|2\d|3[01])\./.test(h))) throw new Error('сервер має бути на цьому комп’ютері (localhost) або в локальній мережі');
  return u.origin;
}

async function ensurePermission(o) {
  if (/^http:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/.test(o)) return true;
  return chrome.permissions.request({ origins: [o + '/*'] });
}

function show(text, ok) {
  const el = $('health');
  el.textContent = text;
  el.className = ok === undefined ? 'muted' : ok ? 'ok' : 'bad';
}

async function save() {
  let o;
  try { o = origin($('server').value.trim() || DEFAULTS.server); } catch (e) { show(e.message, false); return null; }
  if (!(await ensurePermission(o))) { show('Без дозволу на ' + o + ' розширення не зможе звертатися до сервера', false); return null; }
  await chrome.storage.local.set({ server: o, auto: $('auto').checked, replay: $('replay').checked });
  $('server').value = o;
  show('Збережено');
  return o;
}

async function test() {
  const o = await save();
  if (!o) return;
  show('Перевіряю…');
  try {
    const j = await (await fetch(o + '/api/health', { cache: 'no-store', credentials: 'omit' })).json();
    show(j.ok ? `Сервер працює: ${j.countries} країн` : 'Сервер відповів з помилкою', !!j.ok);
  } catch (e) {
    show('Сервер недоступний: запустіть у WSL python3 web/server.py', false);
  }
}

chrome.storage.local.get(DEFAULTS, s => {
  $('server').value = s.server;
  $('auto').checked = !!s.auto;
  $('replay').checked = !!s.replay;
});
$('save').addEventListener('click', save);
$('test').addEventListener('click', test);
$('auto').addEventListener('change', save);
$('replay').addEventListener('change', save);
