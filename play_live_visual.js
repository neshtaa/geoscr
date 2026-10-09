/*
 * Live GeoGuessr helper: points the round's own Street View camera, captures what a player can
 * see, asks the local pure-math locator (python3 web/server.py) and shows countries, regions,
 * observations and GeoGuessr / Plonk It hints in the shared HUD (web/hud/hud.js). The page-world
 * part (hooks, capture, round controller) is extension/page.js, the same file the Chrome extension
 * injects, so watch mode behaves like the extension: Alt+G / the HUD button analyse the current
 * panorama, optional auto-analyse at the start of each round (off by default, as in the extension).
 *
 *   node play_live_visual.js                       watch: play yourself, hints on /challenge/<t> and /game/<t>
 *     ... --auto                                   also analyse at the start of each round (Alt+A toggles it)
 *   node play_live_visual.js --challenge <token>   open /challenge/<token> (you press Play; the page joins)
 *   node play_live_visual.js --game <token>        open /game/<token>
 *     ... --submit                                 also submit the locator's guess (single-player pages only; implies --auto)
 *   node play_live_visual.js --replay <finishedGameToken> --round N    safe test on a finished game
 *     ... --extension                              run the capture through the unpacked extension (extension/)
 *     ... --moves K                                then simulate a player who walks K steps of the round's walk
 *                                                  (cached by tools/moving_calib.py) with auto-analyse on: each
 *                                                  move is re-captured once the player stops, fused by the
 *                                                  server; HUD screenshots hud_move<k>.png, points per step
 *     ... --interrupt hud|pano                     during the first move's sweep click the HUD / drag the panorama
 *
 * Moving games: when the round's panorama changes inside the round (the player moved) and auto-analyse
 * is on, the page's controller re-captures it once the player has been idle for 1.5 s (no pano change,
 * camera turn or mouse button) and the server fuses all panoramas of the round (engine/fusion.py; the
 * round key is the game token + round number). Alt+G does the same at any time.
 *
 * Options: --capture page|screenshot  page (default): the canvas is read in the page, as in the
 *          extension; screenshot: puppeteer screenshots of the isolated canvas (FOV registration probe
 *          for the 2D renderer), --hfov <deg> widest horizontal FOV wanted (default 125; Google caps the
 *          vertical FOV at 90; only used when the game allows zooming), --webgl (software WebGL, so
 *          the FOV is read from Google's projection matrix), --headless, --keep-open; replay only:
 *          --fov-table (FOV for a range of zooms, screenshot capture), --player-zoom <z> (start at
 *          zoom z and keep it, as in a no-zoom game).
 *
 * Fair play: the round's location is never read. The script never requests the game, clue or
 * location endpoints and never queries the panorama id or position. Map, time limit, round number,
 * game type and forbid flags come from GET /api/v3/challenges/<t>, from the page's own responses
 * (reduced to settings by safeGameMeta() inside the page, so no location reaches this process) and
 * from the DOM. Only single-player /challenge/<t> and /game/<t> pages (game type standard or
 * challenge) are captured. The pass-through hooks that must exist before the game builds its
 * panorama are injected into every page, but do nothing outside those pages; multiplayer pages are
 * never captured or automated and show nothing. The HUD gets card images only as data: URLs from
 * the local server's cache (GET /hud/img), never from www.geoguessr.com.
 * --replay takes the finished round from data/calibration/history_rounds.json and makes no API call.
 * Every capture is saved to scratch/live/<game>_r<round>/ (views, views.json, meta, response).
 * Cookie: data/session_cookie.txt (_ncfa value) or env GEOGUESSR_COOKIE.
 */
'use strict';
require('dns').setDefaultResultOrder('ipv4first');
const fs = require('fs');
const path = require('path');
const { execFile } = require('child_process');
const P = require('./extension/page.js');

const ROOT = __dirname;
const SITE = 'https://www.geoguessr.com';
const SERVER = process.env.LOCATOR_URL || 'http://localhost:8080';
const OUT_ROOT = path.join(ROOT, 'scratch', 'live');
const FOV_CACHE = path.join(OUT_ROOT, 'fov_cache.json');
const HISTORY = path.join(ROOT, 'data', 'calibration', 'history_rounds.json');
const EXTENSION_DIR = path.join(ROOT, 'extension');
const PAGE_SRC = fs.readFileSync(path.join(EXTENSION_DIR, 'page.js'), 'utf8');
const HUD_JS = path.join(ROOT, 'web', 'hud', 'hud.js');
const HUD_CSS = path.join(ROOT, 'web', 'hud', 'hud.css');
const GG_IMAGE = /^https:\/\/www\.geoguessr\.com\//;
const RAD = Math.PI / 180;
const PROBE_YAW = 20;          // largest yaw step between the two frames of the photometric FOV probe
const PROBE_OK = 0.3;          // probe accepted when its best MAD is below 0.3 x the median MAD
const SETTLE_MS = 1000;        // a new panorama is captured no earlier than this after its pano change
const TILE_RE = /streetviewpixels|\/cbk\?|GeoPhotoService|photometa|ggpht\.com|googleusercontent\.com/;
const { ALLOWED_PATHS, REPLAY_PATH, isAllowedPath, mergeMeta, serverMap, captureVerdict, gameFlags, challengeMeta,
  hfovFromVfov, vfovFromHfov, hfovForZoom, zoomForHfov, formulaFov, planGrid, predictBody } = P;
const MAX_VFOV = P.MAX_VFOV;

const sleep = ms => new Promise(r => setTimeout(r, ms));

// ------------------------------------------------------------------ fair-play checks (node side)

// Why --submit must not post on this page (null when it may).
function submitProblem(pathname, meta) {
  const v = captureVerdict(pathname, meta);
  if (!v.ok) return v.refuse || v.wait;
  const g = (meta && meta.game) || {};
  if (!P.SINGLE_PLAYER_TYPES.includes(g.type)) return 'game type not known yet (press Play so the page loads the game)';
  if (g.mode !== 'standard') return `game mode ${g.mode || 'unknown'} is not supported`;
  if (g.state !== 'started') return `game state is ${g.state || 'unknown'}`;
  if (!g.token) return 'game token not known yet';
  return null;
}

// The screenshot must be the whole canvas, undistorted: the per-view FOV describes the full canvas.
function clipProblem(iso, canvas) {
  const c = iso.canvasBox, k = iso.clip;
  const r = b => `${Math.round(b.width)}x${Math.round(b.height)}@${Math.round(b.x)},${Math.round(b.y)}`;
  if (Math.abs(k.x - c.x) > 1 || Math.abs(k.y - c.y) > 1 || Math.abs(k.x + k.width - c.x - c.width) > 1 ||
    Math.abs(k.y + k.height - c.y - c.height) > 1)
    return `panorama canvas is cut off (canvas ${r(c)}, visible ${r(k)}): enlarge the window or scroll it into view`;
  if (Math.abs((canvas.width / canvas.height) / (c.width / c.height) - 1) > 0.01)
    return `panorama canvas is stretched (${canvas.width}x${canvas.height} shown as ${r(c)})`;
  return null;
}

// --replay: the finished round from the local game history (no request to the game API).
function replayRound(rows, token, n) {
  const own = (rows || []).filter(r => r && r.game === token && r.kind === 'standard');
  if (!own.length) {
    throw new Error(`--replay: ${token} is not a finished game in data/calibration/history_rounds.json ` +
      '(refresh it with tools/build_dataset.py history)');
  }
  const r = own.find(x => x.round === n);
  if (!r || !r.pano_id) throw new Error(`--replay: no round ${n} of ${token} in the history`);
  return { view: { pano: r.pano_id, heading: r.gg_heading || 0, pitch: 0, zoom: 0 }, truth: { lat: r.lat, lng: r.lng },
    map: r.map ? { name: r.map } : null };
}

// ------------------------------------------------------------------ injected scripts

// extension/page.js with its config (only the switches; the allowed paths are fixed in page.js).
function pageScript(cfg = {}) {
  const c = { replay: !!cfg.replay, preserve: cfg.preserve !== false, submit: !!cfg.submit };
  return `window.__geoscrConfig = ${JSON.stringify(c)};\n${PAGE_SRC}`;
}
const PAGE_SCRIPT = pageScript({});

function hudScript() {
  return `window.__geoscrHudCss = ${JSON.stringify(fs.readFileSync(HUD_CSS, 'utf8'))};\n${fs.readFileSync(HUD_JS, 'utf8')}`;
}

// The round controller of page.js talks to this process (window.__geoscrNode).
const attachScript = settings => `(function () { if (window.__geoscr) window.__geoscr.attach(null, ${JSON.stringify(settings)}); })();`;

// --webgl: Maps JS draws Street View with a 2D canvas when WebGL is software (SwiftShader); hiding
// the renderer name on the allowed pages makes it use WebGL, whose projection matrix gives the exact FOV.
function maskSoftwareGl(pathSources) {
  const paths = pathSources.map(s => new RegExp(s));
  for (const C of [window.WebGLRenderingContext, window.WebGL2RenderingContext]) {
    if (!C) continue;
    const gp = C.prototype.getParameter;
    C.prototype.getParameter = function (p) {
      const v = gp.apply(this, arguments);
      return (p === 0x9245 || p === 0x9246) && typeof v === 'string' && /swiftshader/i.test(v) &&
        paths.some(re => re.test(location.pathname)) ? 'ANGLE (Generic GPU)' : v;
    };
  }
}

// ------------------------------------------------------------------ node side

function getCookie() {
  const p = path.join(ROOT, 'data', 'session_cookie.txt');
  if (fs.existsSync(p)) return fs.readFileSync(p, 'utf8').trim();
  return process.env.GEOGUESSR_COOKIE || '';
}

let lastRequestAt = 0, rateLimited = false;
// The script's own GeoGuessr requests: >= 1.3 s apart, stop at the first 429.
async function requestSlot() {
  if (rateLimited) throw new Error('GeoGuessr returned 429 earlier: no more requests');
  const wait = lastRequestAt + 1300 - Date.now();
  if (wait > 0) await sleep(wait);
  lastRequestAt = Date.now();
}

async function siteGet(url) {
  await requestSlot();
  const headers = { Origin: SITE, Referer: SITE + '/', Accept: 'application/json', Cookie: `_ncfa=${getCookie()}` };
  let resp;
  for (let attempt = 0; ; attempt++) {
    try { resp = await fetch(url, { headers }); break; } catch (e) {
      if (attempt >= 1) throw e;
      await sleep(1500);
    }
  }
  if (resp.status === 429) { rateLimited = true; throw new Error('HTTP 429 from GeoGuessr: stopping requests'); }
  if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${(await resp.text()).slice(0, 150)}`);
  return resp.json();
}

function loadFovCache() {
  try { return JSON.parse(fs.readFileSync(FOV_CACHE, 'utf8')); } catch (e) { return {}; }
}

function saveFovCache(cache) {
  fs.mkdirSync(OUT_ROOT, { recursive: true });
  fs.writeFileSync(FOV_CACHE, JSON.stringify(cache, null, 1));
}

function trackTiles(page) {
  const pending = new Set();
  let last = Date.now();
  page.on('request', r => { if (TILE_RE.test(r.url())) { pending.add(r); last = Date.now(); } });
  const done = r => { if (pending.delete(r)) last = Date.now(); };
  page.on('requestfinished', done);
  page.on('requestfailed', done);
  // wait until the tiles of the new direction are in and drawn
  return async (minMs = 300, idleMs = 250, maxMs = 3000) => {
    const t0 = Date.now();
    await sleep(minMs);
    while (Date.now() - t0 < maxMs && (pending.size || Date.now() - last < idleMs)) await sleep(40);
    await page.evaluate(() => new Promise(r => {
      requestAnimationFrame(() => requestAnimationFrame(r));
      setTimeout(r, 500);
    }));
  };
}

const pageState = page => page.evaluate(() => (window.__geoscr ? window.__geoscr.state() : null));
const isolate = (page, on, p) => page.evaluate((on, p) => (window.__geoscr ? window.__geoscr.isolate(on, p) : null), on, p);

// HUD state through the page's round controller.
async function hudShow(page, patch) {
  await page.evaluate(s => {
    const S = window.__geoscr;
    if (!S || !S.ctl) return;
    if (s.status && s.status.level === 'busy' && S.ctl.view.status && S.ctl.view.status.level === 'busy')
      s.status.since = S.ctl.view.status.since;
    else if (s.status && s.status.level === 'busy') s.status.since = Date.now();
    S.ctl.show(s);
  }, patch).catch(() => null);
}
const hudStatus = (page, text, level = 'busy', extra = {}) => hudShow(page, { status: Object.assign({ text, level }, extra) });

// Every camera action is bound to the page the capture started on.
function assertPath(ctx) {
  let p = null;
  try { p = new URL(ctx.page.url()).pathname; } catch (e) { /* closed */ }
  if (p !== ctx.capturePath) throw new Error(`page changed during the capture (${p})`);
}

async function setView(ctx, v) {
  const ok = await ctx.page.evaluate((v, p) => window.__geoscr.setView(v, p), v, ctx.capturePath);
  if (!ok) throw new Error('page changed during the capture (or the panorama is gone)');
}

async function readMeta(ctx, p) {
  const m = await ctx.page.evaluate(p => (window.__geoscr ? window.__geoscr.metaFor(p) : null), p).catch(() => null);
  ctx.metaCache[p] = mergeMeta(ctx.metaCache[p] || {}, m);
  return ctx.metaCache[p];
}

function matrixFov(st, f) {
  const aspect = st.canvas.width / st.canvas.height;
  return { hfov: hfovFromVfov(f.vfov, aspect), vfov: f.vfov, hfovMatrix: f.hfov, method: 'webgl-matrix' };
}

// ------------------------------------------------------------------ screenshot capture (--capture screenshot)

// Point the camera, wait for the tiles, read back POV / zoom / projection and take the frame.
// With the WebGL renderer a fresh projection matrix also proves that the new view was drawn.
async function grabAt(ctx, clip, view) {
  const before = await pageState(ctx.page);
  const drawn = s => (s.fov ? s.fov.seq > before.seq : s.draws > before.draws);
  await setView(ctx, view);
  await ctx.waitTiles();
  let st = await pageState(ctx.page);
  const tracked = st.fov || st.draws > 0;
  for (let i = 0; tracked && !drawn(st) && i < 20; i++) {
    if (i === 10) await setView(ctx, { heading: view.heading + 0.01, pitch: view.pitch });
    await sleep(100);
    st = await pageState(ctx.page);
  }
  if (tracked && !drawn(st)) console.log(`   [!] no redraw seen for heading ${view.heading}, pitch ${view.pitch}`);
  const fresh = st.fov && st.fov.seq > before.seq ? st.fov : null;
  assertPath(ctx);
  const buf = await ctx.page.screenshot({ type: 'jpeg', quality: 92, clip, captureBeyondViewport: false });
  return { buf, st, fresh };
}

function runProbe(a, poseA, b, poseB) {
  const args = [path.join(ROOT, 'tools', 'rebuild_views.py'), '--probe',
    `${a}:${poseA.heading}:${poseA.pitch}`, `${b}:${poseB.heading}:${poseB.pitch}`];
  return new Promise((resolve, reject) => {
    execFile('python3', args, { timeout: 60000 }, (err, stdout, stderr) => {
      if (err) return reject(new Error('FOV probe failed: ' + String(stderr || err.message).slice(-200)));
      try { resolve(JSON.parse(stdout.trim().split('\n').pop())); } catch (e) { reject(e); }
    });
  });
}

const fovKey = (canvas, zoom) => `2d|${canvas.width}x${canvas.height}|z${Number(zoom).toFixed(3)}`;

// Measured FOV at `zoom`: Google's projection matrix (WebGL renderer); with the 2D renderer two
// frames a few degrees apart are registered (tools/rebuild_views.py --probe), cached per canvas size.
// setZoom false: the zoom is the player's (no-zoom games) and is only read, never set.
async function measureFov(ctx, zoom, base, clip, dir, { forceProbe = false, setZoom = true } = {}) {
  const z = setZoom ? zoom : null;
  const a = await grabAt(ctx, clip, { zoom: z, heading: base.heading + 1, pitch: 0 });
  const st = a.st;
  const aspect = st.canvas.width / st.canvas.height;
  const matrix = a.fresh ? matrixFov(st, a.fresh) : null;
  const key = fovKey(st.canvas, st.zoom);
  let probe = !forceProbe && !matrix && !st.fov ? ctx.fovCache[key] : null;
  if ((forceProbe || !matrix) && !probe) {
    const step = Math.min(PROBE_YAW, Math.max(3, formulaFov(st.zoom, aspect).hfov / 5));
    const b = await grabAt(ctx, clip, { zoom: z, heading: base.heading + 1 + step, pitch: 0 });
    fs.mkdirSync(dir, { recursive: true });
    const fa = path.join(dir, `fovprobe_z${st.zoom.toFixed(2)}_a.jpg`), fb = path.join(dir, `fovprobe_z${st.zoom.toFixed(2)}_b.jpg`);
    fs.writeFileSync(fa, a.buf);
    fs.writeFileSync(fb, b.buf);
    const r = await runProbe(fa, a.st.pov, fb, b.st.pov).catch(e => { console.log('   [!] ' + e.message); return null; });
    if (r) probe = { hfov: r.hfov, vfov: vfovFromHfov(r.hfov, aspect), madRatio: r.mad_ratio, method: 'registration' };
    if (r && r.mad_ratio < PROBE_OK && !matrix && !st.fov) { ctx.fovCache[key] = probe; saveFovCache(ctx.fovCache); }
  }
  const out = { zoom: st.zoom, matrix, probe };
  if (matrix) return Object.assign(out, matrix);
  if (probe && probe.madRatio < PROBE_OK) return Object.assign(out, { hfov: probe.hfov, vfov: probe.vfov, method: 'registration' });
  return Object.assign(out, formulaFov(st.zoom, aspect));
}

// Zoom whose measured horizontal FOV is close to the target (cached per canvas size). Google
// caps the vertical FOV at 90 deg, so the widest view depends on the canvas aspect; at the cap
// (zoom 0) two rows of views cover -85..85. fixedZoom: keep the player's zoom (no-zoom games).
async function chooseZoom(ctx, base, clip, dir, canvas, fixedZoom) {
  if (fixedZoom !== null) {
    return Object.assign(await measureFov(ctx, fixedZoom, base, clip, dir, { setZoom: false }), { zoom: fixedZoom });
  }
  const key = `${canvas.width}x${canvas.height}`;
  const prev = ctx.zoomChoice[key];
  if (prev && prev.method !== 'webgl-matrix') return prev;
  const widest = hfovFromVfov(MAX_VFOV, canvas.width / canvas.height);
  const target = Math.min(ctx.opts.hfov, widest);
  let z = prev ? prev.zoom : target >= widest - 0.01 ? 0 : Math.max(0, zoomForHfov(target)), best = null, last = null;
  for (let i = 0; i < 4; i++) {
    const f = Object.assign(await measureFov(ctx, z, base, clip, dir), { zoom: Math.round(z * 1000) / 1000 });
    if (!best || Math.abs(f.hfov - target) < Math.abs(best.hfov - target)) best = f;
    if (Math.abs(f.hfov - target) < 2 || f.method === 'formula' || (last && Math.abs(f.hfov - last.hfov) < 0.3)) break;
    last = f;
    const nz = Math.min(3, Math.max(0, z + Math.log2(Math.tan(f.hfov * RAD / 2) / Math.tan(target * RAD / 2))));
    if (Math.abs(nz - z) < 0.01) break;
    z = nz;
  }
  ctx.zoomChoice[key] = best;
  return best;
}

async function isolateChecked(ctx, st) {
  const iso = await isolate(ctx.page, true, ctx.capturePath);
  const bad = !iso ? 'panorama not visible' : iso.clip.width < 50 || iso.clip.height < 50 ? 'panorama too small'
    : clipProblem(iso, st.canvas);
  if (bad) {
    await isolate(ctx.page, false, ctx.capturePath);
    throw new Error(bad);
  }
  return iso;
}

async function captureSweep(ctx, st, dir, zoomLocked) {
  const { page } = ctx;
  const fixedZoom = zoomLocked ? st.zoom : null;
  const base = { heading: st.pov.heading, pitch: st.pov.pitch, zoom: fixedZoom === null ? st.zoom : null };
  const iso = await isolateChecked(ctx, st);
  const views = [];
  let choice, grid;
  try {
    choice = await chooseZoom(ctx, base, iso.clip, dir, st.canvas, fixedZoom);
    grid = planGrid(choice.hfov, choice.vfov, base.heading);
    console.log(`   FOV: zoom ${choice.zoom}${zoomLocked ? ' (player\'s zoom)' : ''} -> hfov ${choice.hfov.toFixed(2)}, ` +
      `vfov ${choice.vfov.toFixed(2)} (${choice.method}); ${grid.length} views`);
    for (const g of grid) {
      const r = await grabAt(ctx, iso.clip, { heading: g.yaw, pitch: g.pitch, zoom: fixedZoom === null ? choice.zoom : null });
      const f = r.fresh ? matrixFov(r.st, r.fresh) : choice;
      views.push({ buf: r.buf, yaw: r.st.pov.heading, pitch: r.st.pov.pitch, hfov: f.hfov, vfov: f.vfov,
        zoom: r.st.zoom, fov_method: r.fresh ? 'webgl-matrix' : choice.method });
    }
  } finally {
    await setView(ctx, base).catch(() => null);
    await isolate(page, false, ctx.capturePath).catch(() => null);
  }
  return { views, mode: 'sweep', fov: choice, grid, iso, canvas: st.canvas, zoomLocked };
}

// No-rotate rounds: only the current frame, with its getPov heading, once the canvas is not blurred.
async function captureSingle(ctx) {
  const { page } = ctx;
  let st = await pageState(page);
  for (let t0 = Date.now(); st.blur && Date.now() - t0 < 15000;) {
    await sleep(200);
    st = await pageState(page);
  }
  if (st.blur) throw new Error('panorama still blurred (mouse button held?)');
  const iso = await isolateChecked(ctx, st);
  try {
    const aspect = st.canvas.width / st.canvas.height;
    const cached = ctx.fovCache[fovKey(st.canvas, st.zoom)];
    const f = st.fov ? matrixFov(st, st.fov) : cached || formulaFov(st.zoom, aspect);
    st = await pageState(page);
    assertPath(ctx);
    const buf = await page.screenshot({ type: 'jpeg', quality: 92, clip: iso.clip, captureBeyondViewport: false });
    const views = [{ buf, yaw: st.pov.heading, pitch: st.pov.pitch, hfov: f.hfov, vfov: f.vfov, zoom: st.zoom,
      fov_method: f.method }];
    console.log(`   FOV: zoom ${st.zoom} -> hfov ${f.hfov.toFixed(2)}, vfov ${f.vfov.toFixed(2)} (${f.method}); single frame`);
    return { views, mode: 'single', fov: f, grid: null, iso, canvas: st.canvas, zoomLocked: true };
  } finally {
    await isolate(page, false, ctx.capturePath).catch(() => null);
  }
}

// views.json + view_XX.jpg + meta.json in scratch/live/<game>_r<round>/ (input of tools/rebuild_views.py).
function saveCapture(dir, cap, info, meta) {
  fs.mkdirSync(dir, { recursive: true });
  const views = cap.views.map((v, i) => {
    v.file = `view_${String(i).padStart(2, '0')}.jpg`;
    fs.writeFileSync(path.join(dir, v.file), v.buf);
    return { file: v.file, yaw: v.yaw, pitch: v.pitch, hfov: v.hfov, vfov: v.vfov, zoom: v.zoom, fov_method: v.fov_method,
      read: v.read };
  });
  const json = {
    game: info.key, round: info.round, mode: cap.mode, capture: cap.source || 'screenshot', captured_at: new Date().toISOString(),
    zoom_locked: cap.zoomLocked, fov: cap.fov, grid: cap.grid, canvas: cap.canvas, views,
  };
  if (cap.iso) {
    const dpr = cap.iso.dpr;
    Object.assign(json, { canvas_box: cap.iso.canvasBox, clip: cap.iso.clip, clip_to: cap.iso.clipTo, dpr,
      image: [Math.round(cap.iso.clip.width * dpr), Math.round(cap.iso.clip.height * dpr)] });
  } else if (cap.views[0]) json.image = [cap.views[0].width, cap.views[0].height];
  fs.writeFileSync(path.join(dir, 'views.json'), JSON.stringify(json, null, 1));
  fs.writeFileSync(path.join(dir, 'meta.json'), JSON.stringify({ map: serverMap(meta), game: meta.game || null,
    challenge: meta.challenge || null }, null, 1));
}

// A capture made in the page (views with JPEG data URLs) in the same layout as a screenshot capture.
function pageCapture(payload) {
  const c = payload.capture || {};
  const views = (payload.views || []).map(v => Object.assign({}, v, {
    buf: Buffer.from(String(v.image_b64).replace(/^data:image\/\w+;base64,/, ''), 'base64') }));
  return { views, mode: c.mode || 'sweep', fov: c.fov || null, grid: c.grid || null, canvas: c.canvas || null,
    zoomLocked: !!c.zoomLocked, source: 'page' };
}

async function predict(views, map, fusion = null) {
  const body = predictBody(views.map(v => ({ image_b64: v.buf ? v.buf.toString('base64') : v.image_b64, yaw: v.yaw,
    pitch: v.pitch, hfov: v.hfov, vfov: v.vfov })), map, fusion);
  const resp = await fetch(`${SERVER}/api/predict`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  });
  const data = await resp.json();
  if (data.error) throw new Error(data.error);
  return data;
}

// The result as the HUD gets it: GeoGuessr clue images become data: URLs from the server's offline
// cache (tools/fetch_hud_images.py) or are dropped; the page never loads them from GeoGuessr.
async function inlineImages(result) {
  const out = JSON.parse(JSON.stringify(result));
  const cards = [];
  for (const h of out.hints || []) for (const k of (h && h.geoguessr) || []) if (k && k.image_url) cards.push(k);
  await Promise.all(cards.map(async k => {
    const url = String(k.image_url);
    k.image_url = null;
    if (!GG_IMAGE.test(url)) return;
    try {
      const r = await fetch(`${SERVER}/hud/img?u=${encodeURIComponent(url)}`);
      const type = (r.headers.get('content-type') || '').split(';')[0];
      if (r.ok && /^image\/(jpeg|png|webp)$/.test(type)) k.image_url = `data:${type};base64,${Buffer.from(await r.arrayBuffer()).toString('base64')}`;
    } catch (e) { /* no image */ }
  }));
  return out;
}

function logResult(info, result, tCap, map, dir) {
  console.log(`[${info.label}] ${result.countries.map(x => `${x.code} ${(x.probability * 100).toFixed(0)}%`).join(', ')}` +
    `  -> ${result.guess.lat}, ${result.guess.lng}  (capture ${(tCap / 1000).toFixed(1)} s, map ${map ? map.name || map.id : '?'})`);
  const f = result.fusion;
  if (f && (f.n > 1 || f.replaced)) {
    console.log(`   рух: панорам ${f.n}${f.replaced ? ' (та сама панорама, замінено)' : ''}, ` +
      `${f.changed_top ? `відповідь змінилась (${f.prev_top} -> ${f.top})` : 'відповідь та сама'}; окремо: ` +
      (f.captures || []).map((c, i) => `${i + 1}: ${(c.countries[0] || {}).code}`).join(', '));
  }
  console.log('   видно: ' + result.observations.map(o => o.text).join('; '));
  if (result.hints[0] && result.hints[0].regions.length)
    console.log('   регіон: ' + result.hints[0].regions.map(x => `${x.name} ${Math.round(x.probability * 100)}%`).join(', '));
  console.log(`   збережено: ${path.relative(ROOT, dir)}`);
}

// GeoGuessr reuses one StreetViewPanorama for all rounds and getStatus() stays 'OK' through setPano,
// so a new round is ready only after a pano change (since the last capture on this page), a fresh
// draw, SETTLE_MS and idle tiles.
async function waitPanorama(ctx, since, timeoutMs = 20000) {
  const t0 = Date.now();
  let moved = !since;
  for (;;) {
    assertPath(ctx);
    const st = await pageState(ctx.page);
    if (st && st.renderer === 'tiles') throw new Error('unsupported renderer: GeoGuessr tiles panorama (Baidu / GG_ round)');
    if (st && !moved && (st.id !== since.id || st.panoChanges !== since.panoChanges)) moved = true;
    if (!moved && Date.now() - t0 > 12000) {
      moved = true;
      console.log('   [!] no panorama change seen since the last capture; capturing what is shown');
    }
    if (moved && st && st.renderer === 'google' && st.status === 'OK' && st.pov && !st.blur && st.canvas &&
      st.drawnSincePano && (st.sincePano === null || st.sincePano >= SETTLE_MS)) {
      await ctx.waitTiles(0, 400, 4000);
      return pageState(ctx.page);
    }
    if (Date.now() - t0 > timeoutMs) {
      if (!st || st.renderer === 'none') throw new Error('no panorama on the page');
      if (st.renderer === 'google-unhooked') throw new Error('Street View object not captured (hook failed)');
      if (st.blur) throw new Error('panorama still blurred (mouse button held?)');
      throw new Error(`panorama not loaded (status ${st.status})`);
    }
    await sleep(200);
  }
}

// --capture screenshot: the whole round in this process (the page's controller waits for the result).
async function analyseRound(ctx, info) {
  const { page } = ctx;
  ctx.capturePath = info.path;
  const dir = path.join(OUT_ROOT, info.dir || `${info.key}_r${info.round}`);
  await hudStatus(page, 'Чекаю панораму…');
  const st = await waitPanorama(ctx, info.since);
  const flags = gameFlags(info.meta, st.nmpz);
  let noRotate = flags.noRotate;
  if (noRotate === null) {
    noRotate = await page.evaluate(p => window.__geoscr.rotateLocked(p), info.path);
    if (noRotate === null) throw new Error('page changed during the capture');
    console.log(`   forbidRotating невідомий; перевірка камери: ${noRotate ? 'без обертання' : 'обертання дозволене'}`);
  }
  const zoomLocked = info.freeZoom ? false : flags.zoomLocked;
  await hudStatus(page, noRotate ? 'Без обертання: знімаю поточний кадр…' : 'Знімаю панораму…');
  const t0 = Date.now();
  const cap = noRotate ? await captureSingle(ctx) : await captureSweep(ctx, st, dir, zoomLocked);
  const map = serverMap(info.meta);
  saveCapture(dir, cap, info, info.meta);
  const tCap = Date.now() - t0;
  await hudStatus(page, 'Аналіз…');
  const t1 = Date.now();
  const result = await predict(cap.views, map, info.fusion || null);
  fs.writeFileSync(path.join(dir, 'response.json'), JSON.stringify(result, null, 1));
  logResult(info, result, tCap, map, dir);
  return { result, dir, noRotate: !!noRotate, timing: { capture_ms: tCap, server_ms: result.timing_ms ? result.timing_ms.total : null,
    request_ms: Date.now() - t1, views: cap.views.length, mode: cap.mode, fov: cap.fov } };
}

// ------------------------------------------------------------------ bridge to the page's round controller

// Requests of extension/page.js (window.__geoscrNode): meta, set, health, predict, capture, log.
async function bridge(ctx, type, payload) {
  payload = payload || {};
  if (type === 'health') {
    const r = await fetch(`${SERVER}/api/health`).then(x => x.json()).catch(() => null);
    if (!r || !r.ok) throw new Error(`Сервер локатора недоступний (${SERVER}). Запустіть: python3 web/server.py`);
    return r;
  }
  if (type === 'meta') {
    const p = String(payload.path || '');
    return mergeMeta(ctx.metaCache[p] || {}, ctx.extraMeta[p] || null);
  }
  if (type === 'set') {
    if (typeof payload.auto === 'boolean') {
      ctx.settings.auto = payload.auto;
      console.log(`Автоаналіз на початку раунду: ${payload.auto ? 'увімкнено' : 'вимкнено'}`);
    }
    return ctx.settings;
  }
  if (type === 'log') { console.log('   [page] ' + String(payload.text || '').slice(0, 300)); return null; }
  const job = payload.job || payload;
  const p = String(job.path || '');
  const meta = mergeMeta(mergeMeta(ctx.metaCache[p] || {}, job.meta || null), ctx.extraMeta[p] || null);
  const info = { key: String(job.key || 'page').replace(/[^A-Za-z0-9]/g, '') || 'page', round: +job.round || 0,
    label: job.label || '', path: p, meta, trigger: job.trigger, fusion: payload.fusion || job.fusion || null };
  // a move of the player: its own capture directory <game>_r<round>_m<k>
  const rk = `${info.key}_r${info.round}`;
  if (job.trigger === 'move') ctx.moves[rk] = (ctx.moves[rk] || 0) + 1;
  const sub = job.trigger === 'move' ? `${rk}_m${ctx.moves[rk]}` : rk;
  let out;
  if (type === 'predict') {
    const dir = path.join(OUT_ROOT, sub);
    const cap = pageCapture(payload);
    saveCapture(dir, cap, info, meta);
    const result = await predict(cap.views, payload.map || serverMap(meta), info.fusion);
    fs.writeFileSync(path.join(dir, 'response.json'), JSON.stringify(result, null, 1));
    const c = payload.capture || {};
    console.log(`   знімання в сторінці: ${cap.views.length} кадр., ${cap.views[0] ? cap.views[0].read : '?'}, ` +
      `FOV ${cap.fov ? `${cap.fov.hfov.toFixed(2)} x ${cap.fov.vfov.toFixed(2)} (${cap.fov.method})` : '?'}`);
    logResult(info, result, c.ms || 0, payload.map, dir);
    out = await inlineImages(result);
  } else if (type === 'capture') {
    ctx.capturePath = p;
    const st = await pageState(ctx.page);
    info.since = job.since && st ? { id: st.id, panoChanges: job.since.panoChanges } : null;
    info.freeZoom = ctx.freeZoom;
    info.dir = sub;
    const r = await analyseRound(ctx, info);
    out = { result: await inlineImages(r.result), timing: r.timing, noRotate: r.noRotate };
  } else throw new Error(`unknown request ${type}`);
  if (ctx.opts.submit && job.trigger === 'auto') {
    const result = out.result || out;
    setTimeout(() => submitAndContinue(ctx, result, info).catch(async e => {
      console.error('[!] ' + e.message);
      await hudStatus(ctx.page, e.message, 'error');
    }), 300);
  }
  return out;
}

// ------------------------------------------------------------------ modes

async function submitGuess(ctx, guess, p) {
  const meta = await readMeta(ctx, p);
  const problem = submitProblem(p, meta);
  if (problem) throw new Error('--submit: ' + problem);
  const token = meta.game.token;
  await requestSlot();
  const r = await ctx.page.evaluate((url, body, p) => window.__geoscr.post(url, body, p),
    `${SITE}/api/v3/games/${token}`, { token, lat: guess.lat, lng: guess.lng, timedOut: false, stepsCount: 0 }, p);
  if (r.status === 429) { rateLimited = true; throw new Error('HTTP 429 from GeoGuessr: stopping requests'); }
  if (r.status !== 200) throw new Error(`--submit: HTTP ${r.status} ${r.error || ''}`);
  ctx.metaCache[p] = mergeMeta(ctx.metaCache[p], r.meta);
  return r.meta && r.meta.game ? r.meta.game.lastGuess : null;
}

async function submitAndContinue(ctx, result, info) {
  const { page } = ctx;
  ctx.capturePath = info.path;
  const last = await submitGuess(ctx, result.guess, info.path);
  const pts = last ? last.points : 0;
  const km = last ? Math.round(last.distanceMeters / 100) / 10 : null;
  ctx.total += pts || 0;
  console.log(`   результат: ${pts} балів, ${km} км (всього ${ctx.total})`);
  await page.reload({ waitUntil: 'domcontentloaded', timeout: 45000 });
  await page.waitForSelector("[data-qa='close-round-result']", { timeout: 20000 });
  await hudShow(page, { mode: 'game', round: info.label, result, status: { text: 'Результат', level: 'ok' },
    score: { points: pts, distance: km, total: ctx.total } });
  await sleep(4000);
  if (/^\/game\//.test(info.path))
    console.log('   (якщо з’явиться вікно квитка, закрийте його, не натискайте Play: це витрачає квиток)');
  assertPath(ctx);
  // the next round is captured only after its own pano change (after this result screen)
  await page.evaluate(() => {
    const S = window.__geoscr;
    if (S && S.ctl && S.ctl.lastKey) { S.ctl.resultKey = S.ctl.lastKey; S.ctl.resultMark = S.panoChanges; }
  });
  await page.click("[data-qa='close-round-result']");
}

async function fovTable(ctx, dir) {
  const { page } = ctx;
  const st = await waitPanorama(ctx, null);
  const base = { heading: st.pov.heading, pitch: st.pov.pitch, zoom: st.zoom };
  const iso = await isolateChecked(ctx, st);
  const rows = [];
  try {
    for (const z of [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3]) {
      const f = await measureFov(ctx, z, base, iso.clip, path.join(dir, 'fov_table'), { forceProbe: true });
      const row = { zoom: z, zoom_readback: f.zoom, formula_tan: hfovForZoom(z), formula_180: Math.min(180 / Math.pow(2, z), 170),
        matrix: f.matrix, probe: f.probe };
      rows.push(row);
      console.log(`   zoom ${z.toFixed(2)}: matrix ${f.matrix ? `hfov ${f.matrix.hfov.toFixed(2)} vfov ${f.matrix.vfov.toFixed(2)}` : '-'}` +
        `  registration ${f.probe ? `hfov ${f.probe.hfov.toFixed(2)} (mad ratio ${f.probe.madRatio})` : '-'}` +
        `  | 2atan(2^(1-z)) ${row.formula_tan.toFixed(2)}  180/2^z ${row.formula_180.toFixed(2)}`);
    }
  } finally {
    await setView(ctx, base).catch(() => null);
    await isolate(page, false, ctx.capturePath).catch(() => null);
  }
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, 'fov_table.json'), JSON.stringify({ canvas: st.canvas, rows }, null, 1));
}

// The unpacked extension's settings (server URL, replay test mode) through its service worker.
async function configureExtension(browser, settings) {
  const target = await browser.waitForTarget(t => t.type() === 'service_worker' && /^chrome-extension:\/\//.test(t.url()),
    { timeout: 20000 });
  const worker = await target.worker();
  await worker.evaluate(s => chrome.storage.local.set(s), settings);
  return target.url().split('/')[2];
}

// Waits for the page controller to finish the analysis it is running (or has finished).
async function controllerResult(page, timeoutMs = 120000) {
  const t0 = Date.now();
  for (;;) {
    const v = await page.evaluate(() => {
      const S = window.__geoscr;
      if (!S || !S.ctl) return null;
      return { busy: !!S.ctl.busy, level: S.ctl.view.status && S.ctl.view.status.level, text: S.ctl.view.status && S.ctl.view.status.text,
        result: S.ctl.view.result || null, timing: S.ctl.view.timing || null };
    }).catch(() => null);
    if (v && !v.busy && (v.level === 'ok' || v.level === 'error') && (v.result || v.level === 'error')) return v;
    if (Date.now() - t0 > timeoutMs) throw new Error('no result from the page controller' + (v ? `: ${v.text}` : ''));
    await sleep(300);
  }
}

async function runReplay(ctx, fin) {
  const { page, opts } = ctx;
  const n = fin.round;
  const p = `/game/${opts.replay}/replay`;
  ctx.capturePath = p;
  ctx.extraMeta[p] = { map: fin.map };
  if (opts.extension) {
    // the extension's content script mirrors the replay test switch into localStorage on a GeoGuessr page;
    // page.js reads it at document_start of the next load (as after "reload the page" in its options)
    await page.goto(`${SITE}/`, { waitUntil: 'domcontentloaded', timeout: 60000 });
    await page.waitForFunction(() => localStorage.getItem('__geoscr_cfg') !== null, { timeout: 20000 });
  }
  await page.goto(`${SITE}${p}?round=${n}&step=0`, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForSelector('canvas.widget-scene-canvas', { timeout: 30000 });
  await sleep(3000);
  assertPath(ctx);
  const st = await pageState(page);
  if (!st || st.status !== 'OK') {
    console.log('   повтор не записано: показую панораму завершеного раунду в панорамі сторінки');
    if (!(await page.evaluate((v, p) => window.__geoscr.loadPano(v, p), fin.view, p))) throw new Error('no panorama to load the round into');
  }
  const info = { key: opts.replay, round: n, label: `Повтор ${n}`, path: p, meta: { map: fin.map }, since: null };
  const dir = path.join(OUT_ROOT, `${info.key}_r${info.round}`);
  if (opts.fovTable) await fovTable(ctx, dir);
  let result;
  if (opts.extension) {
    // as the user would: Alt+G in the page, the extension captures and asks the server
    await page.waitForFunction(() => window.__geoscr && window.__geoscr.ctl && window.__geoscr.ctl.attached, { timeout: 20000 });
    await page.waitForFunction(() => window.GeoscrHud && window.__geoscr.ctl.hud, { timeout: 20000 }).catch(() => null);
    const t0 = Date.now();
    await page.keyboard.down('Alt');
    await page.keyboard.press('KeyG');
    await page.keyboard.up('Alt');
    await sleep(500);
    const v = await controllerResult(page);
    if (!v.result) throw new Error('extension: ' + v.text);
    result = v.result;
    const last = await page.evaluate(() => {
      const c = window.__geoscr.ctl.last;
      return c && c.capture ? { views: c.capture.views, capture: { mode: c.capture.mode, fov: c.capture.fov, grid: c.capture.grid,
        canvas: c.capture.canvas, zoomLocked: c.capture.zoomLocked, ms: c.capture.ms } } : null;
    });
    if (last) {
      const cap = pageCapture(last);
      saveCapture(dir, cap, info, info.meta);
      fs.writeFileSync(path.join(dir, 'response.json'), JSON.stringify(result, null, 1));
      console.log(`   розширення: ${cap.views.length} кадр., читання ${cap.views[0] && cap.views[0].read}, ` +
        `FOV ${cap.fov ? `${cap.fov.hfov.toFixed(2)} x ${cap.fov.vfov.toFixed(2)} (${cap.fov.method})` : '?'}`);
      logResult(info, result, last.capture.ms || 0, serverMap(info.meta), dir);
    }
    console.log(`   розширення: аналіз за ${((Date.now() - t0) / 1000).toFixed(1)} с (timing ${JSON.stringify(v.timing)})`);
  } else {
    result = await page.evaluate(() => window.__geoscr.ctl.analyse('replay'));
    if (!result) {
      const s = await page.evaluate(() => window.__geoscr.ctl.view.status);
      throw new Error((s && s.text) || 'no result');
    }
  }
  const resp = await fetch(`${SERVER}/api/evaluate_round`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ lat: fin.truth.lat, lng: fin.truth.lng, guess_lat: result.guess.lat, guess_lng: result.guess.lng,
      pred_code: result.countries[0].code, map: serverMap(info.meta) }),
  }).then(r => r.json()).catch(e => ({ error: e.message }));
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, 'evaluation.json'), JSON.stringify(resp, null, 1));
  console.log(`   перевірка (завершений раунд): справжня країна ${resp.true_code}, топ-1 ${resp.pred_code}, ` +
    `${resp.distance_km} км, ${resp.points} балів`);
  if (opts.moves) await runMoves(ctx, fin, info, dir, { result, evaluation: resp });
}

// Neighbouring panoramas of a finished round cached by tools/moving_calib.py (local, no API call).
function cachedNeighbours(pano) {
  return new Promise((resolve, reject) => {
    execFile('python3', [path.join(ROOT, 'tools', 'moving_calib.py'), 'neighbours', pano], { timeout: 60000 }, (err, stdout, stderr) => {
      if (err) return reject(new Error(String(stderr || err.message).trim().split('\n').pop()));
      try { resolve(JSON.parse(stdout.trim().split('\n').pop())); } catch (e) { reject(e); }
    });
  });
}

const fusionView = page => page.evaluate(() => {
  const C = window.__geoscr && window.__geoscr.ctl;
  if (!C) return null;
  const f = C.view.fusion;
  return { busy: !!C.busy, level: C.view.status && C.view.status.level, text: C.view.status && C.view.status.text,
    seq: f ? f.seq : 0, fusion: f || null, result: C.view.result || null, panoChanges: window.__geoscr.panoChanges };
}).catch(() => null);

// Waits until the controller has fused a capture newer than `seq` (or failed).
async function waitFusion(page, seq, timeoutMs = 180000) {
  const t0 = Date.now();
  let started = false;
  for (;;) {
    const v = await fusionView(page);
    if (v && v.busy) started = true;
    if (v && !v.busy && v.seq > seq && v.level === 'ok') return v;
    if (v && !v.busy && started && v.level === 'error') throw new Error('рух: ' + v.text);
    if (Date.now() - t0 > timeoutMs) throw new Error('рух: немає нового результату' + (v ? ` (${v.text})` : ''));
    await sleep(300);
  }
}

// --replay --moves K: a player who walks K steps of the finished round's cached walk (tools/moving_calib.py:
// Street View's own links from the round's panorama; the first step, then about 45 / 100 / 170 m). Auto-analyse
// is switched on in the page, so its controller notices each pano change, waits until the player is idle,
// re-captures and gets the server's fused answer. Every step: the fused result, the captures' own answers,
// a HUD screenshot (hud_move<k>.png) and the score of the fused guess against the finished round; the
// summary lists the points of every step from the start (step 0: the round's panorama alone), worse or not.
// --interrupt hud|pano: during the first move's sweep the "player" presses like a real one would: a click on the
// HUD, or a drag of the panorama. Records the player's camera before the sweep, the camera the sweep had
// turned to, the camera right after the press (the page puts the player's one back inside the press event)
// and after the release (a drag stays the player's).
async function interruptSweep(page, where) {
  const pov = () => page.evaluate(() => {
    const S = window.__geoscr, p = S.pano();
    const v = p.getPov();
    return { heading: Math.round(v.heading * 100) / 100, pitch: Math.round(v.pitch * 100) / 100, zoom: p.getZoom() };
  }).catch(() => null);
  const t0 = Date.now();
  let sw = null;
  while (!sw && Date.now() - t0 < 90000) {
    sw = await page.evaluate(() => {
      const s = window.__geoscr.sweep;
      return s && s.touched ? { base: s.base } : null;
    }).catch(() => null);
    if (!sw) await sleep(100);
  }
  if (!sw) return { error: 'no sweep seen' };
  await sleep(1500);   // the middle of the sweep
  const during = await pov();
  const sel = where === 'hud' ? '#geoscr-hud-host' : 'canvas.widget-scene-canvas';
  const box = await page.$eval(sel, el => { const r = el.getBoundingClientRect(); return { x: r.left, y: r.top, w: r.width, h: r.height }; });
  const x = where === 'hud' ? box.x + 40 : box.x + box.w / 2, y = where === 'hud' ? box.y + 14 : box.y + box.h / 2;
  await page.mouse.move(x, y);
  await page.mouse.down();
  const atPress = await pov();
  const stop = await page.evaluate(() => { const s = window.__geoscr.sweep; return s ? s.stop : 'ended'; }).catch(() => null);
  if (where === 'pano') await page.mouse.move(x + 80, y + 10, { steps: 8 });
  await page.mouse.up();
  await sleep(400);
  const afterRelease = await pov();
  const base = { heading: Math.round(sw.base.heading * 100) / 100, pitch: Math.round(sw.base.pitch * 100) / 100, zoom: sw.base.zoom };
  const restored = !!atPress && Math.abs(((atPress.heading - base.heading) % 360 + 540) % 360 - 180) < 0.05 &&
    Math.abs(atPress.pitch - base.pitch) < 0.05;
  return { where, base, during, atPress, stop, afterRelease, restored_at_press: restored };
}

async function runMoves(ctx, fin, info, dir, start) {
  const { page, opts } = ctx;
  const p = info.path;
  const nb = (await cachedNeighbours(fin.view.pano)).slice(0, opts.moves);
  if (!nb.length) throw new Error('--moves: no cached neighbouring panoramas for this round');
  await page.evaluate(() => { const C = window.__geoscr.ctl; C.settings.auto = true; C.show({ auto: true }); });
  const hud = async name => {
    const el = await page.$('#geoscr-hud-host');
    if (el) await el.screenshot({ path: path.join(dir, name) }).catch(e => console.log('   [!] HUD screenshot: ' + e.message));
  };
  await hud('hud_move0.png');
  const steps = [];
  for (const [i, n] of nb.entries()) {
    const before = await fusionView(page);
    const heading = Math.round((fin.view.heading + 50 * (i + 1)) % 360);
    console.log(`   крок ${i + 1}: гравець переходить на панораму ${n.dist_m} м від старту (${n.steps || '?'} кл. стрілок)`);
    const ok = await page.evaluate((v, p) => window.__geoscr.loadPano(v, p), { pano: n.pano_id, heading, pitch: 0, zoom: 0 }, p);
    if (!ok) throw new Error('--moves: the replay page has no panorama to move in');
    const t0 = Date.now();
    const probe = opts.interrupt && i === 0 ? interruptSweep(page, opts.interrupt) : null;
    const v = await waitFusion(page, before ? before.seq : 0);
    const interrupted = probe ? await probe : null;
    if (interrupted) {
      console.log(`   крок 1: «гравець» натискає (${interrupted.where}) під час огляду: ${JSON.stringify(interrupted)}`);
      await page.screenshot({ path: path.join(dir, 'interrupt_after.png') }).catch(() => null);
    }
    const f = v.fusion, r = v.result;
    const ev = await fetch(`${SERVER}/api/evaluate_round`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ lat: fin.truth.lat, lng: fin.truth.lng, guess_lat: r.guess.lat, guess_lng: r.guess.lng,
        pred_code: r.countries[0].code, map: serverMap(info.meta) }),
    }).then(x => x.json()).catch(e => ({ error: e.message }));
    const step = { step: i + 1, dist_m: n.dist_m, arrow_steps: n.steps, role: n.role, seconds: Math.round((Date.now() - t0) / 100) / 10,
      panoChanges: v.panoChanges,
      fused: { n: f.n, w: f.w, changed_top: f.changed_top, prev_top: f.prev_top, replaced: f.replaced,
        countries: r.countries.slice(0, 3), guess: r.guess, regions: (r.regions || []).slice(0, 2) },
      captures: f.captures, evaluation: ev, status: v.text, interrupted };
    steps.push(step);
    console.log(`   крок ${i + 1}: за ${step.seconds} с; ${v.text}`);
    console.log(`      злито ${f.n}: ${r.countries.slice(0, 3).map(x => `${x.code} ${(x.probability * 100).toFixed(0)}%`).join(', ')}` +
      ` -> ${r.guess.lat}, ${r.guess.lng}; окремо ${f.captures.map((c, k) => `${k + 1}:${(c.countries[0] || {}).code}`).join(' ')}` +
      `; перевірка: ${ev.distance_km} км, ${ev.points} балів`);
    await hud(`hud_move${i + 1}.png`);
  }
  const ev0 = (start && start.evaluation) || {};
  const r0 = start && start.result;
  const step0 = { step: 0, dist_m: 0, fused: r0 ? { n: 1, countries: r0.countries.slice(0, 3), guess: r0.guess } : null, evaluation: ev0 };
  fs.writeFileSync(path.join(dir, 'moves.json'), JSON.stringify([step0].concat(steps), null, 1));
  const pts = [step0].concat(steps).map(s => (s.evaluation && typeof s.evaluation.points === 'number' ? s.evaluation.points : null));
  console.log(`   рух: бали по кроках (0 = лише стартова панорама): ${pts.map((x, k) => `${k}: ${x === null ? '?' : x}`).join(', ')}` +
    (pts[0] !== null && pts[pts.length - 1] !== null ? `; останній крок ${pts[pts.length - 1] >= pts[0] ? '+' : ''}${pts[pts.length - 1] - pts[0]} до старту` : ''));
  console.log(`   рух: ${steps.length} кроків, збережено ${path.relative(ROOT, path.join(dir, 'moves.json'))} і hud_move*.png`);
}

function parseArgs(argv) {
  const val = (name, def = null) => { const i = argv.indexOf(name); return i >= 0 && i + 1 < argv.length ? argv[i + 1] : def; };
  const has = name => argv.includes(name);
  return {
    challenge: val('--challenge'), game: val('--game'), replay: val('--replay'), round: val('--round', '1'),
    submit: has('--submit'), headless: has('--headless'), webgl: has('--webgl'), fovTable: has('--fov-table'),
    keepOpen: has('--keep-open'), hfov: Math.min(125, Math.max(60, parseFloat(val('--hfov', '125')))),
    playerZoom: val('--player-zoom') === null ? null : Math.min(3, Math.max(0, parseFloat(val('--player-zoom')) || 0)),
    auto: has('--auto') || has('--submit'), extension: has('--extension'),
    capture: val('--capture', 'page') === 'screenshot' ? 'screenshot' : 'page',
    moves: Math.max(0, Math.min(4, parseInt(val('--moves', '0'), 10) || 0)),
    interrupt: ['hud', 'pano'].includes(val('--interrupt')) ? val('--interrupt') : null,
  };
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  for (const k of ['challenge', 'game', 'replay']) {
    if (opts[k] && !/^[A-Za-z0-9]+$/.test(opts[k])) { console.error(`[!] bad --${k} token`); process.exit(1); }
  }
  if (opts.submit && opts.replay) { console.error('[!] --submit cannot be used with --replay'); process.exit(1); }
  if (opts.extension && !opts.replay) { console.error('[!] --extension is a test of the extension on --replay'); process.exit(1); }
  let fin = null;
  if (opts.replay) {
    try {
      const rows = JSON.parse(fs.readFileSync(HISTORY, 'utf8'));
      fin = Object.assign(replayRound(rows, opts.replay, parseInt(opts.round, 10)), { round: parseInt(opts.round, 10) });
      if (opts.playerZoom !== null) fin.view.zoom = opts.playerZoom;
    } catch (e) { console.error('[!] ' + e.message); process.exit(1); }
  }
  try { await fetch(`${SERVER}/api/health`); } catch (e) {
    console.error(`[!] Сервер локатора недоступний (${SERVER}). Запустіть: python3 web/server.py`);
    process.exit(1);
  }
  const puppeteer = require('puppeteer');
  const args = ['--no-sandbox', '--window-size=1400,900'];
  if (opts.webgl || opts.headless) args.push('--use-angle=swiftshader', '--enable-unsafe-swiftshader');
  if (opts.extension) args.push(`--disable-extensions-except=${EXTENSION_DIR}`, `--load-extension=${EXTENSION_DIR}`);
  const browser = await puppeteer.launch({ headless: opts.headless, args,
    ignoreDefaultArgs: opts.extension ? ['--disable-extensions'] : [],
    defaultViewport: opts.headless ? { width: 1400, height: 800 } : null });
  browser.on('disconnected', () => process.exit(process.exitCode || 0));
  const page = (await browser.pages())[0] || await browser.newPage();
  const paths = opts.replay ? ALLOWED_PATHS.concat([REPLAY_PATH]) : ALLOWED_PATHS;
  const ctx = { page, opts, metaCache: {}, extraMeta: {}, fovCache: loadFovCache(), zoomChoice: {}, total: 0, moves: {},
    capturePath: null, waitTiles: trackTiles(page), settings: { auto: opts.auto && !opts.replay },
    freeZoom: !!opts.replay && opts.playerZoom === null };
  if (opts.replay) {
    await page.setRequestInterception(true);
    page.on('request', req => {
      let host = '';
      try { host = new URL(req.url()).hostname; } catch (e) { /* data: urls */ }
      if (/(^|\.)geoguessr\.com$/.test(host) && !['GET', 'HEAD', 'OPTIONS'].includes(req.method())) {
        console.log(`   [replay] заблоковано ${req.method()} ${req.url().slice(0, 100)}`);
        return req.abort();
      }
      return req.continue();
    });
  }
  if (opts.extension) {
    const id = await configureExtension(browser, { server: SERVER, replay: true, auto: false });
    console.log(`Розширення ${id} завантажено; сервер ${SERVER}, режим перевірки на повторах увімкнено`);
  } else {
    await page.exposeFunction('__geoscrNode', (type, payload) => bridge(ctx, type, payload));
    await page.evaluateOnNewDocument(pageScript({ replay: !!opts.replay, submit: opts.submit }));
    await page.evaluateOnNewDocument(hudScript());
    await page.evaluateOnNewDocument(attachScript({ node: true, auto: ctx.settings.auto, capture: opts.capture,
      keepZoom: opts.playerZoom !== null }));
  }
  if (opts.webgl) await page.evaluateOnNewDocument(`(${maskSoftwareGl})(${JSON.stringify(paths.map(re => re.source))});`);
  const cookie = getCookie();
  if (cookie) await page.setCookie({ name: '_ncfa', value: cookie, domain: '.geoguessr.com' });
  let failed = false;
  try {
    if (opts.replay) await runReplay(ctx, fin);
    else {
      let start = SITE + '/';
      if (opts.challenge) {
        const base = challengeMeta(await siteGet(`${SITE}/api/v3/challenges/${opts.challenge}`)) || {};
        ctx.metaCache[`/challenge/${opts.challenge}`] = base;
        const c = base.challenge || {};
        console.log(`Челендж: ${(base.map || {}).name || c.mapSlug}, ${c.roundCount} раундів, ${c.timeLimit || '∞'} с,` +
          ` forbid move/zoom/rotate ${!!c.forbidMoving}/${!!c.forbidZooming}/${!!c.forbidRotating}. Натисніть Play на сторінці.`);
        start = `${SITE}/challenge/${opts.challenge}`;
      } else if (opts.game) start = `${SITE}/game/${opts.game}`;
      await page.goto(start, { waitUntil: 'domcontentloaded', timeout: 60000 });
      console.log('Режим спостереження: грайте у вікні браузера. Alt+G або кнопка «Аналіз» у HUD — аналіз панорами; ' +
        `автоаналіз на початку раунду ${ctx.settings.auto ? 'увімкнено' : 'вимкнено'} (Alt+A).`);
      await new Promise(() => null);   // the page's controller does the rest; exits with the browser
    }
  } catch (e) {
    failed = true;
    console.error('[!] ' + (e.stack || e));
  } finally {
    if (opts.replay && !opts.keepOpen) await browser.close();
  }
  if (failed) process.exitCode = 1;
}

module.exports = Object.assign({}, P, {
  submitProblem, clipProblem, replayRound, pageScript, hudScript, attachScript, PAGE_SCRIPT, trackTiles, waitPanorama,
  pageCapture, bridge, inlineImages,
});

if (require.main === module) main();
