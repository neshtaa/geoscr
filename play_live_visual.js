/*
 * Live GeoGuessr helper: opens a visible browser, captures the round from 12 camera
 * directions, asks the local pure-math locator (python3 web/server.py) and shows the
 * country probabilities and GeoGuessr / Plonk It hints in an on-screen HUD.
 *
 *   node play_live_visual.js                      watch mode: open geoguessr.com, play yourself, get hints
 *   node play_live_visual.js --game <token>       open a running standard game
 *   node play_live_visual.js --challenge <token>  start a challenge (needs a plan that allows it)
 *   add --submit to also submit the locator's guess through the game API
 *
 * The page's Street View object is only used to POINT the camera (setPov / setZoom).
 * The position or panorama id is never read: the locator sees exactly what a player sees.
 * Cookie: data/session_cookie.txt (_ncfa value) or env GEOGUESSR_COOKIE.
 */
const puppeteer = require('puppeteer');
const fs = require('fs');

const SERVER = process.env.LOCATOR_URL || 'http://localhost:8080';
const PITCHES = [-55, 0, 55];
const YAWS = [0, 90, 180, 270];
const HFOV = 120;                       // horizontal field of view of each capture (deg)
const ZOOM = Math.log2(180 / HFOV);     // Maps JS Street View: fov = 180 / 2^zoom

function arg(name) {
  const i = process.argv.indexOf(name);
  return i >= 0 ? process.argv[i + 1] : null;
}
const SUBMIT = process.argv.includes('--submit');
const sleep = ms => new Promise(r => setTimeout(r, ms));

function getCookie() {
  const p = 'data/session_cookie.txt';
  if (fs.existsSync(p)) return fs.readFileSync(p, 'utf8').trim();
  return process.env.GEOGUESSR_COOKIE || '';
}

async function apiRequest(url, method = 'GET', data = null, cookie = '') {
  const headers = { 'Origin': 'https://www.geoguessr.com', 'Referer': 'https://www.geoguessr.com/' };
  if (cookie) headers['Cookie'] = `_ncfa=${cookie}`;
  const opts = { method, headers };
  if (data) { headers['Content-Type'] = 'application/json'; opts.body = JSON.stringify(data); }
  const resp = await fetch(url, opts);
  if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${(await resp.text()).slice(0, 150)}`);
  return await resp.json();
}

// Runs in the page before any script: remember the StreetViewPanorama instance.
function hookStreetView() {
  const tryHook = () => {
    const g = window.google;
    if (!g || !g.maps || !g.maps.StreetViewPanorama) return false;
    const P = g.maps.StreetViewPanorama;
    if (P.__geoscrHooked) return true;
    const proto = P.prototype;
    for (const m of ['setPov', 'setZoom', 'setVisible', 'setOptions', 'getPov', 'setPano', 'setPosition']) {
      const orig = proto[m];
      if (typeof orig !== 'function') continue;
      proto[m] = function (...a) { window.__geoscrPano = this; return orig.apply(this, a); };
    }
    const Hooked = function (...a) { const inst = new P(...a); window.__geoscrPano = inst; return inst; };
    Hooked.prototype = proto;
    Object.setPrototypeOf(Hooked, P);
    Hooked.__geoscrHooked = true;
    P.__geoscrHooked = true;
    try { g.maps.StreetViewPanorama = Hooked; } catch (e) {}
    return true;
  };
  const iv = setInterval(() => { try { if (tryHook()) clearInterval(iv); } catch (e) {} }, 10);
}

async function panoReady(page) {
  return page.evaluate(() => !!(window.__geoscrPano && typeof window.__geoscrPano.setPov === 'function'));
}

// Hide every element that is not the panorama (UI, map, compass, Google controls).
async function isolatePanorama(page, on) {
  return page.evaluate((on) => {
    const pano = document.querySelector("[data-qa='panorama']") ||
      (document.querySelector("[data-qa='panorama-canvas'], canvas.widget-scene-canvas") || {}).parentElement;
    if (!pano) return null;
    if (on) {
      window.__geoscrHidden = [];
      for (const el of document.body.querySelectorAll('*')) {
        if (el.contains(pano)) continue;
        const inside = pano.contains(el);
        const gm = inside && el.matches('.gmnoprint, .gm-style-cc, .gm-iv-address, .gm-compass, a[href*="maps.google"], [class*="compass"], [class*="controls"], [class*="tooltip"]');
        if ((!inside || gm) && el.style.visibility !== 'hidden') {
          window.__geoscrHidden.push([el, el.style.visibility]);
          el.style.visibility = 'hidden';
        }
      }
    } else {
      for (const [el, v] of (window.__geoscrHidden || [])) el.style.visibility = v;
      window.__geoscrHidden = [];
    }
    const r = pano.getBoundingClientRect();
    return { x: r.left, y: r.top, width: r.width, height: r.height };
  }, on);
}

async function captureViews(page) {
  const box = await isolatePanorama(page, true);
  if (!box) throw new Error('panorama element not found');
  const views = [];
  try {
    for (const pitch of PITCHES) {
      for (const yaw of YAWS) {
        await page.evaluate((h, p, z) => {
          window.__geoscrPano.setZoom(z);
          window.__geoscrPano.setPov({ heading: h, pitch: p });
        }, yaw, pitch, ZOOM);
        await sleep(650); // let the tiles of this direction load
        const buf = await page.screenshot({ type: 'jpeg', quality: 90, clip: box });
        views.push({ image_b64: buf.toString('base64'), yaw, pitch, hfov: HFOV });
      }
    }
    await page.evaluate(() => window.__geoscrPano.setPov({ heading: 0, pitch: 0 }));
  } finally {
    await isolatePanorama(page, false);
  }
  return views;
}

async function predict(views) {
  const resp = await fetch(`${SERVER}/api/predict`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ views })
  });
  const data = await resp.json();
  if (data.error) throw new Error(data.error);
  return data;
}

async function showHud(page, state) {
  try {
    await page.evaluate((s) => {
      let hud = document.getElementById('geoscr-hud');
      if (!hud) {
        hud = document.createElement('div');
        hud.id = 'geoscr-hud';
        hud.style.cssText = 'position:fixed;top:16px;right:16px;width:380px;max-height:86vh;overflow-y:auto;' +
          'background:rgba(15,23,42,.94);border:2px solid #10b981;border-radius:12px;padding:14px;color:#f8fafc;' +
          'font:13px/1.4 system-ui,sans-serif;z-index:99999999;box-shadow:0 12px 32px rgba(0,0,0,.6)';
        document.body.appendChild(hud);
      }
      const esc = t => String(t == null ? '' : t).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
      let h = `<div style="display:flex;justify-content:space-between;font-weight:800;color:#10b981;margin-bottom:8px">
        <span>Локатор (без ШІ)</span><span>${esc(s.round || '')}</span></div>
        <div style="color:#94a3b8;margin-bottom:8px">${esc(s.status || '')}</div>`;
      const r = s.result;
      if (r) {
        for (const c of r.countries) {
          const w = Math.max(2, Math.round(c.probability * 100));
          h += `<div style="display:flex;align-items:center;gap:6px;margin:3px 0"><span style="width:150px">${esc(c.name)}</span>
            <div style="height:7px;width:${w}%;max-width:150px;background:#10b981;border-radius:4px"></div>
            <span>${(c.probability * 100).toFixed(1)}%</span></div>`;
        }
        h += `<div style="color:#94a3b8;margin:6px 0">Здогадка ${r.guess.lat}, ${r.guess.lng} · ~${r.guess.expected_score} балів</div>`;
        if (r.observations.length) {
          h += `<div style="margin:8px 0 4px;color:#94a3b8;font-weight:600">ЩО ВИДНО</div>` +
            r.observations.slice(0, 8).map(o => `<span style="display:inline-block;border:1px solid #0284c7;color:#38bdf8;border-radius:6px;padding:2px 7px;margin:2px;font-size:11px">${esc(o.text)}</span>`).join('');
        }
        for (const c of r.hints) {
          h += `<div style="margin-top:10px;font-weight:700">${esc(c.country)} — ${(c.probability * 100).toFixed(1)}%</div>`;
          if (c.regions && c.regions.length) h += `<div style="color:#f59e0b;font-size:11px">Регіон: ${c.regions.map(x => `${esc(x.name)} ${Math.round(x.probability * 100)}%`).join(', ')}</div>`;
          for (const k of c.geoguessr.slice(0, 2)) h += `<div style="border-left:3px solid #10b981;padding:4px 8px;margin-top:4px;background:#1e293b;border-radius:4px"><b>${esc(k.title)}</b><div style="color:#cbd5e1;font-size:11px">${esc(k.text)}</div></div>`;
          for (const k of c.plonkit.slice(0, 1)) h += `<div style="border-left:3px solid #f59e0b;padding:4px 8px;margin-top:4px;background:#1e293b;border-radius:4px;color:#cbd5e1;font-size:11px">Plonk It: ${esc(k.text)}</div>`;
        }
      }
      if (s.score) h += `<div style="margin-top:10px;padding:8px;background:#064e3b;border-radius:8px">Раунд: +${esc(s.score.points)} балів, ${esc(s.score.distance)} км · всього ${esc(s.score.total)}</div>`;
      hud.innerHTML = h;
    }, state);
  } catch (e) { /* page navigated */ }
}

async function analyzeRound(page, label) {
  await showHud(page, { round: label, status: 'Знімаю 12 напрямків…' });
  for (let i = 0; i < 50 && !(await panoReady(page)); i++) await sleep(200);
  if (!(await panoReady(page))) throw new Error('Street View object not captured (hook failed)');
  const views = await captureViews(page);
  await showHud(page, { round: label, status: 'Аналіз…' });
  const result = await predict(views);
  const top = result.countries[0];
  console.log(`[${label}] ${result.countries.map(c => `${c.code} ${(c.probability * 100).toFixed(0)}%`).join(', ')}  -> ${result.guess.lat}, ${result.guess.lng}`);
  console.log('   видно: ' + result.observations.map(o => o.text).join('; '));
  if (result.hints[0] && result.hints[0].regions.length)
    console.log('   регіон: ' + result.hints[0].regions.map(x => `${x.name} ${Math.round(x.probability * 100)}%`).join(', '));
  await showHud(page, { round: label, status: `Найімовірніше: ${top.name}`, result });
  return result;
}

async function roundLabel(page) {
  return page.evaluate(() => {
    const el = document.querySelector("[data-qa='round-number']");
    return el ? el.innerText.replace(/\s+/g, ' ').trim() : null;
  });
}

async function playGame(page, cookie, gameToken) {
  await page.goto(`https://www.geoguessr.com/game/${gameToken}`, { waitUntil: 'networkidle2', timeout: 45000 });
  let total = 0;
  for (;;) {
    const game = await apiRequest(`https://www.geoguessr.com/api/v3/games/${gameToken}`, 'GET', null, cookie);
    const done = (game.player && game.player.guesses) ? game.player.guesses.length : 0;
    if (game.state === 'finished' || done >= game.roundCount) break;
    const label = `Раунд ${done + 1}/${game.roundCount}`;
    await page.waitForSelector("[data-qa='panorama'], [data-qa='panorama-canvas']", { timeout: 20000 });
    await sleep(1500);
    const result = await analyzeRound(page, label);
    if (!SUBMIT) {
      console.log('   (--submit не задано: зробіть хід самостійно)');
      await waitForGuess(cookie, gameToken, done);
      await sleep(3000);
      continue;
    }
    const g = await apiRequest(`https://www.geoguessr.com/api/v3/games/${gameToken}`, 'POST',
      { token: gameToken, lat: result.guess.lat, lng: result.guess.lng, timedOut: false }, cookie);
    const last = g.player.guesses[g.player.guesses.length - 1];
    const pts = parseInt(last.roundScoreInPoints || 0);
    total += pts;
    const dist = Math.round(parseFloat(last.distanceInMeters || 0) / 100) / 10;
    console.log(`   результат: ${pts} балів, ${dist} км`);
    await page.goto(`https://www.geoguessr.com/game/${gameToken}`, { waitUntil: 'networkidle2', timeout: 35000 });
    await showHud(page, { round: label, status: 'Результат', result, score: { points: pts, distance: dist, total } });
    await sleep(4000);
  }
  console.log(`Гру завершено. Всього: ${total}`);
}

async function waitForGuess(cookie, gameToken, done) {
  for (;;) {
    await sleep(2000);
    const g = await apiRequest(`https://www.geoguessr.com/api/v3/games/${gameToken}`, 'GET', null, cookie).catch(() => null);
    if (g && ((g.player && g.player.guesses ? g.player.guesses.length : 0) > done || g.state === 'finished')) return;
  }
}

async function watch(page) {
  console.log('Режим спостереження: грайте у вікні браузера, підказки з’являтимуться на кожному раунді.');
  await page.goto('https://www.geoguessr.com/', { waitUntil: 'networkidle2', timeout: 45000 });
  let last = null;
  for (;;) {
    await sleep(1500);
    if (page.isClosed()) return;
    let label;
    try { label = await roundLabel(page); } catch (e) { continue; }
    const hasPano = await page.$("[data-qa='panorama'], [data-qa='panorama-canvas']").catch(() => null);
    if (!label || !hasPano || label === last) continue;
    last = label;
    await sleep(1500);
    try { await analyzeRound(page, label); } catch (e) { console.error('[!] ' + e.message); }
  }
}

async function main() {
  const cookie = getCookie();
  try { await fetch(`${SERVER}/api/health`); } catch (e) {
    console.error(`[!] Сервер локатора недоступний (${SERVER}). Запустіть: python3 web/server.py`);
    process.exit(1);
  }
  const browser = await puppeteer.launch({ headless: false, defaultViewport: null,
    args: ['--no-sandbox', '--window-size=1400,900'] });
  const page = (await browser.pages())[0] || await browser.newPage();
  await page.evaluateOnNewDocument(hookStreetView);
  if (cookie) await page.setCookie({ name: '_ncfa', value: cookie, domain: '.geoguessr.com' });
  try {
    if (arg('--game')) await playGame(page, cookie, arg('--game'));
    else if (arg('--challenge')) {
      const g = await apiRequest(`https://www.geoguessr.com/api/v3/challenges/${arg('--challenge')}`, 'POST', {}, cookie);
      await playGame(page, cookie, g.token);
    } else await watch(page);
  } catch (e) {
    console.error('[!] ' + (e.stack || e));
  } finally {
    if (arg('--game') || arg('--challenge')) { await sleep(8000); await browser.close(); }
  }
}

main();
