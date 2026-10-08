/*
 * Live GeoGuessr helper: points the round's own Street View camera, captures what a player can
 * see, asks the local pure-math locator (python3 web/server.py) and shows countries, regions,
 * observations and GeoGuessr / Plonk It hints in an on-screen HUD.
 *
 *   node play_live_visual.js                       watch: play yourself, hints on /challenge/<t> and /game/<t>
 *   node play_live_visual.js --challenge <token>   open /challenge/<token> (you press Play; the page joins)
 *   node play_live_visual.js --game <token>        open /game/<token>
 *     ... --submit                                 also submit the locator's guess (single-player pages only)
 *   node play_live_visual.js --replay <finishedGameToken> --round N    safe test on a finished game
 *
 * Options: --hfov <deg> widest horizontal FOV wanted (default 125; Google caps the vertical FOV at 90;
 *          only used when the game allows zooming), --webgl (software WebGL, so the FOV is read from
 *          Google's projection matrix), --headless, --keep-open; replay only: --fov-table (FOV for a
 *          range of zooms), --player-zoom <z> (start at zoom z and keep it, as in a no-zoom game).
 *
 * Fair play: the round's location is never read. The script never requests the game, clue or
 * location endpoints and never queries the panorama id or position. Map, time limit, round number,
 * game type and forbid flags come from GET /api/v3/challenges/<t>, from the page's own responses
 * (reduced to settings by safeGameMeta() inside the page, so no location reaches this process) and
 * from the DOM. Only single-player /challenge/<t> and /game/<t> pages (game type standard or
 * challenge) are captured. The pass-through hooks that must exist before the game builds its
 * panorama are injected into every page, but do nothing outside those pages; multiplayer pages are
 * never captured or automated and only get a "not supported" notice.
 * --replay takes the finished round from data/calibration/history_rounds.json and makes no API call.
 * Every capture is saved to scratch/live/<game>_r<round>/ (views, views.json, meta, response).
 * Cookie: data/session_cookie.txt (_ncfa value) or env GEOGUESSR_COOKIE.
 */
'use strict';
require('dns').setDefaultResultOrder('ipv4first');
const fs = require('fs');
const path = require('path');
const { execFile } = require('child_process');

const ROOT = __dirname;
const SITE = 'https://www.geoguessr.com';
const SERVER = process.env.LOCATOR_URL || 'http://localhost:8080';
const OUT_ROOT = path.join(ROOT, 'scratch', 'live');
const FOV_CACHE = path.join(OUT_ROOT, 'fov_cache.json');
const HISTORY = path.join(ROOT, 'data', 'calibration', 'history_rounds.json');
const RAD = Math.PI / 180;
const OVERLAP = 0.10;          // minimum overlap of neighbouring views
const PITCH_LIMIT = 85;        // the sweep covers -85..+85 deg
const PROBE_YAW = 20;          // largest yaw step between the two frames of the photometric FOV probe
const PROBE_OK = 0.3;          // probe accepted when its best MAD is below 0.3 x the median MAD
const MAX_VFOV = 90;           // Google clamps the vertical FOV (measured in both renderers)
const SETTLE_MS = 1000;        // a new panorama is captured no earlier than this after its pano change
const ALLOWED_PATHS = [/^\/challenge\/[A-Za-z0-9]+$/, /^\/game\/[A-Za-z0-9]+$/];
const REPLAY_PATH = /^\/game\/[A-Za-z0-9]+\/replay$/;
const SINGLE_PLAYER_TYPES = ['standard', 'challenge'];
const TILE_RE = /streetviewpixels|\/cbk\?|GeoPhotoService|photometa|ggpht\.com|googleusercontent\.com/;
const NOT_SUPPORTED = 'Тут не підтримується (мультиплеєр або інша сторінка). Підказки працюють лише в одиночних /challenge/… та /game/….';

const sleep = ms => new Promise(r => setTimeout(r, ms));

// ------------------------------------------------------------------ fair-play metadata whitelist
// These functions are also injected into the page (keep them self-contained). They read only the
// whitelisted keys; nothing below touches rounds, coordinates, pano ids or streak codes.

function isAllowedPath(pathname) {
  return ALLOWED_PATHS.some(re => re.test(pathname));
}

function safeBounds(b) {
  if (!b || !b.min || !b.max) return null;
  const n = x => (typeof x === 'number' && isFinite(x) ? x : null);
  const out = { min: { lat: n(b.min.lat), lng: n(b.min.lng) }, max: { lat: n(b.max.lat), lng: n(b.max.lng) } };
  return [out.min.lat, out.min.lng, out.max.lat, out.max.lng].every(v => v !== null) ? out : null;
}

function safeMapMeta(m) {
  if (!m || typeof m !== 'object') return null;
  const s = x => (typeof x === 'string' ? x : null);
  return { id: s(m.id), slug: s(m.slug), name: s(m.name), bounds: safeBounds(m.bounds),
    maxErrorDistance: typeof m.maxErrorDistance === 'number' ? m.maxErrorDistance : null };
}

// Copies only game settings and progress.
function safeGameMeta(g) {
  if (!g || typeof g !== 'object' || typeof g.token !== 'string') return null;
  const out = {};
  for (const k of ['token', 'type', 'mode', 'state', 'round', 'roundCount', 'timeLimit',
    'forbidMoving', 'forbidZooming', 'forbidRotating', 'mapName', 'guessMapType']) {
    const v = g[k];
    if (v !== undefined && (v === null || typeof v !== 'object')) out[k] = v;
  }
  if (typeof g.map === 'string') out.mapId = g.map;
  else if (g.map && typeof g.map === 'object') out.map = safeMapMeta(g.map);
  const b = safeBounds(g.bounds);
  if (b) out.bounds = b;
  const guesses = g.player && Array.isArray(g.player.guesses) ? g.player.guesses : null;
  if (guesses) {
    out.guessCount = guesses.length;
    const last = guesses[guesses.length - 1];
    if (last) out.lastGuess = { points: Number(last.roundScoreInPoints), distanceMeters: Number(last.distanceInMeters),
      timedOut: !!last.timedOut };
  }
  return out;
}

// GET /api/v3/challenges/<token>: challenge settings and the map.
function challengeMeta(json) {
  if (!json || typeof json !== 'object') return null;
  const c = json.challenge && typeof json.challenge === 'object' ? json.challenge : {};
  const challenge = {};
  for (const k of ['token', 'mapSlug', 'roundCount', 'timeLimit', 'forbidMoving', 'forbidZooming',
    'forbidRotating', 'gameMode']) {
    if (c[k] !== undefined && (c[k] === null || typeof c[k] !== 'object')) challenge[k] = c[k];
  }
  return { challenge, map: safeMapMeta(json.map) };
}

// Any JSON the page receives (game object, challenge, Next.js page data) -> whitelisted metadata.
function extractMeta(json) {
  if (!json || typeof json !== 'object') return null;
  const pp = (json.props && json.props.pageProps) || json.pageProps || null;
  const candidates = [json, json.game, pp && pp.initialGame, pp && pp.gameSnapshot, pp && pp.game];
  const dq = pp && pp.dehydratedState && Array.isArray(pp.dehydratedState.queries) ? pp.dehydratedState.queries : [];
  for (const q of dq) {
    if (q && Array.isArray(q.queryKey) && q.queryKey[0] === 'classic-game' && q.state) candidates.push(q.state.data);
  }
  let game = null;
  for (const g of candidates) {
    if (g && typeof g === 'object' && typeof g.token === 'string' && 'roundCount' in g) {
      game = safeGameMeta(g);
      break;
    }
  }
  const mapObj = (pp && pp.map && typeof pp.map === 'object' && pp.map) ||
    (json.map && typeof json.map === 'object' && json.map) || null;
  const ch = (pp && pp.challenge) || json.challenge;
  const challenge = ch && typeof ch === 'object' && 'roundCount' in ch ? challengeMeta({ challenge: ch }).challenge : null;
  const map = safeMapMeta(mapObj);
  if (!game && !map && !challenge) return null;
  return { game, map, challenge };
}

function mergeMeta(meta, add) {
  if (!add) return meta;
  const out = Object.assign({}, meta);
  for (const k of ['game', 'map', 'challenge']) {
    if (!add[k]) continue;
    const prev = k === 'game' && out.game && out.game.token !== add.game.token ? {} : (out[k] || {});
    const merged = Object.assign({}, prev);
    for (const [kk, v] of Object.entries(add[k])) if (v !== null && v !== undefined) merged[kk] = v;
    out[k] = merged;
  }
  return out;
}

// The page a response of the site's own API (or Next.js page data) belongs to: /game/<t>, /challenge/<t>,
// map:<id> for the map details (GET /api/maps/<id>) or null.
function metaKey(url, origin) {
  let u;
  try { u = new URL(url, origin); } catch (e) { return null; }
  if (u.origin !== origin) return null;
  let m = /^\/api\/v3\/(games|challenges)\/([A-Za-z0-9]+)(\/game)?$/.exec(u.pathname);
  if (m) return `/${m[1] === 'games' ? 'game' : 'challenge'}/${m[2]}`;
  m = /^\/api\/maps\/([A-Za-z0-9-]+)$/.exec(u.pathname);
  if (m) return `map:${m[1]}`;
  m = /^\/_next\/data\/[^/]+\/(game|challenge)\/([A-Za-z0-9]+)(\/replay)?\.json$/.exec(u.pathname);
  return m ? `/${m[1]}/${m[2]}` : null;
}

// The "map" object of POST /api/predict.
function serverMap(meta) {
  const m = (meta && meta.map) || {}, g = (meta && meta.game) || {}, c = (meta && meta.challenge) || {};
  const gm = g.map || {};
  const gameSlug = g.mapId && !/^[0-9a-f]{24}$/.test(g.mapId) ? g.mapId : null;  // games carry an id or a slug
  const out = {
    id: m.id || gm.id || g.mapId || c.mapSlug || null,
    slug: m.slug || gm.slug || c.mapSlug || gameSlug,
    name: m.name || gm.name || g.mapName || null,
    bounds: m.bounds || gm.bounds || g.bounds || null,
    maxErrorDistance: m.maxErrorDistance || gm.maxErrorDistance || null,
  };
  return Object.values(out).some(v => v !== null) ? out : null;
}

function parseRound(text) {
  const m = /(\d+)\s*\/\s*(\d+)/.exec(text || '');
  return m ? { round: +m[1], of: +m[2] } : null;
}

// Whether a page may be captured: {ok}, {wait: why} (game data not seen yet) or {refuse: why}.
// /game/<t> also hosts Play-Along games, so there the game type must be known and single-player.
function captureVerdict(pathname, meta) {
  if (!isAllowedPath(pathname)) return { refuse: 'not a single-player /challenge/<t> or /game/<t> page' };
  const g = (meta && meta.game) || {};
  if (g.type && !SINGLE_PLAYER_TYPES.includes(g.type)) return { refuse: `game type "${g.type}" is not single-player` };
  if (pathname.startsWith('/game/')) {
    if (!g.type) return { wait: 'game type not known yet' };
    if (g.token !== pathname.split('/')[2]) return { wait: 'game data of this page not seen yet' };
  }
  return { ok: true };
}

// Why --submit must not post on this page (null when it may).
function submitProblem(pathname, meta) {
  const v = captureVerdict(pathname, meta);
  if (!v.ok) return v.refuse || v.wait;
  const g = (meta && meta.game) || {};
  if (!SINGLE_PLAYER_TYPES.includes(g.type)) return 'game type not known yet (press Play so the page loads the game)';
  if (g.mode !== 'standard') return `game mode ${g.mode || 'unknown'} is not supported`;
  if (g.state !== 'started') return `game state is ${g.state || 'unknown'}`;
  if (!g.token) return 'game token not known yet';
  return null;
}

// forbidRotating (null = unknown, probe the camera) and whether the zoom must stay as the player has it.
function gameFlags(meta, nmpz) {
  const g = (meta && meta.game) || {}, c = (meta && meta.challenge) || {};
  const flag = k => (typeof g[k] === 'boolean' ? g[k] : typeof c[k] === 'boolean' ? c[k] : null);
  const rotate = flag('forbidRotating');
  return { noRotate: rotate !== null ? rotate : nmpz ? true : null, zoomLocked: flag('forbidZooming') !== false };
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

// ------------------------------------------------------------------ camera geometry

// GeoGuessr's own test for Google's projection matrix (uniformMatrix4fv): row norms give the focal lengths.
function fovFromMatrix(r) {
  if (!r || r.length < 16 || Math.abs(r[14] + 0.6667) > 0.01 || Math.abs(r[15]) > 0.01) return null;
  const row = i => Math.sqrt(r[i] * r[i] + r[i + 4] * r[i + 4] + r[i + 8] * r[i + 8] + r[i + 12] * r[i + 12]);
  const w = row(3);
  if (!(w > 10)) return null;
  const vfov = 2 * Math.atan(w / row(1)) * 180 / Math.PI;
  const hfov = 2 * Math.atan(w / row(0)) * 180 / Math.PI;
  return vfov > 5 && vfov < 170 ? { vfov, hfov } : null;
}

const hfovFromVfov = (vfov, aspect) => 2 * Math.atan(Math.tan(vfov * RAD / 2) * aspect) / RAD;
const vfovFromHfov = (hfov, aspect) => 2 * Math.atan(Math.tan(hfov * RAD / 2) / aspect) / RAD;
// Google Street View (measured, WebGL and 2D): tan(hfov / 2) = 2^(1 - zoom) until the vertical FOV
// reaches 90 deg. Only a fallback and a first guess.
const hfovForZoom = z => 2 * Math.atan(Math.pow(2, 1 - z)) / RAD;
const zoomForHfov = h => 1 - Math.log2(Math.tan(h * RAD / 2));
function formulaFov(zoom, aspect) {
  const vfov = Math.min(MAX_VFOV, vfovFromHfov(hfovForZoom(zoom), aspect));
  return { hfov: hfovFromVfov(vfov, aspect), vfov, method: 'formula' };
}

function inFrustum(azDeg, elDeg, pitchDeg, tx, ty) {
  const a = azDeg * RAD, e = elDeg * RAD, p = pitchDeg * RAD;
  const dx = Math.sin(a) * Math.cos(e), dy = Math.cos(a) * Math.cos(e), dz = Math.sin(e);
  const zc = dy * Math.cos(p) + dz * Math.sin(p);
  if (zc <= 1e-6) return false;
  return Math.abs(dx / zc) <= tx && Math.abs((dz * Math.cos(p) - dy * Math.sin(p)) / zc) <= ty;
}

// Azimuth half-width (deg) that a camera pitched by `pitch` covers at elevation `el`.
function halfWidth(el, pitch, tx, ty) {
  if (!inFrustum(0, el, pitch, tx, ty)) return 0;
  let a = 0;
  while (a < 180 && inFrustum(a + 0.25, el, pitch, tx, ty)) a += 0.25;
  return a;
}

// Rows of pitches covering -85..85 and, per row, enough yaws that neighbours overlap by >= 10 %.
function planGrid(hfov, vfov, startYaw = 0) {
  const tx = Math.tan(hfov * RAD / 2), ty = Math.tan(vfov * RAD / 2);
  const span = 2 * PITCH_LIMIT - vfov;
  let rows = 2;
  while (rows < 9 && span / (rows - 1) > vfov * (1 - OVERLAP)) rows++;
  const pitches = [];
  for (let i = 0; i < rows; i++) pitches.push(-PITCH_LIMIT + vfov / 2 + i * Math.max(0, span) / (rows - 1));
  const grid = [];
  pitches.forEach((p, i) => {
    const lo = i === 0 ? -PITCH_LIMIT : (p + pitches[i - 1]) / 2;
    const hi = i === rows - 1 ? PITCH_LIMIT : (p + pitches[i + 1]) / 2;
    // the band edge nearest the horizon is the narrowest; near the poles the views overlap anyway
    const el = Math.abs(lo) < Math.abs(hi) ? lo : hi;
    const w = halfWidth(Math.max(-60, Math.min(60, el)), p, tx, ty);
    const n = w >= 180 ? 1 : Math.max(1, Math.ceil(360 / (2 * w * (1 - OVERLAP))));
    for (let k = 0; k < n; k++) {
      grid.push({ yaw: Math.round(((startYaw + k * 360 / n) % 360 + 360) % 360 * 100) / 100,
        pitch: Math.round(p * 100) / 100 });
    }
  });
  return grid;
}

// ------------------------------------------------------------------ page-side hooks

// Runs in every page before any script and must stay pass-through: outside the allowed paths
// nothing is remembered, recorded or read. On allowed pages it remembers the StreetViewPanorama
// (constructor + prototype hook; GeoGuessr builds it from the global google.maps), counts pano
// changes and draws, records Google's projection matrix and keeps the whitelisted metadata of the
// page's own game / challenge responses (the full bodies never leave the page).
function installPageHooks(pathSources) {
  if (window.__geoscr) return;
  const paths = pathSources.map(s => new RegExp(s));
  const allowed = () => paths.some(re => re.test(location.pathname));
  const S = window.__geoscr = { id: Math.random().toString(36).slice(2), panos: [], seq: 0, hidden: [], store: {},
    panoChanges: 0, panoAt: 0, mark: null };
  const panoChanged = () => {
    const c = S.canvas();
    S.panoChanges++;
    S.panoAt = performance.now();
    S.mark = { seq: S.seq, canvas: c, draws: c ? c.__geoscrDraws || 0 : 0 };
  };
  const remember = inst => {
    if (!inst || typeof inst !== 'object' || !allowed()) return;
    if (!S.panos.includes(inst)) {
      S.panos.push(inst);
      try { inst.addListener('pano_changed', panoChanged); } catch (e) { /* not an MVCObject */ }
    }
    inst.__geoscrUsed = performance.now();
  };
  const hookStreetView = () => {
    const g = window.google;
    if (!g || !g.maps || !g.maps.StreetViewPanorama) return false;
    const P = g.maps.StreetViewPanorama;
    if (P.__geoscrHooked) return true;
    const proto = P.prototype;
    for (const m of ['setPov', 'setZoom', 'setVisible', 'setOptions', 'getPov', 'getZoom', 'setPano', 'setPosition']) {
      const orig = proto[m];
      if (typeof orig !== 'function') continue;
      proto[m] = function () {
        remember(this);
        if (m === 'setPano' && allowed()) panoChanged();
        return orig.apply(this, arguments);
      };
    }
    const Hooked = function () {
      const inst = new P(...arguments);
      if (allowed()) {
        inst.__geoscrDiv = arguments[0];
        remember(inst);
      }
      return inst;
    };
    Hooked.prototype = proto;
    Object.setPrototypeOf(Hooked, P);
    Hooked.__geoscrHooked = true;
    P.__geoscrHooked = true;
    try { g.maps.StreetViewPanorama = Hooked; } catch (e) { /* read-only: the prototype hook still works */ }
    return true;
  };
  let tries = 0;
  const tick = () => {
    try { if (hookStreetView()) return; } catch (e) { /* retry */ }
    setTimeout(tick, ++tries < 4000 ? 5 : 50);
  };
  tick();

  for (const C of [window.WebGLRenderingContext, window.WebGL2RenderingContext]) {
    if (!C || typeof C.prototype.uniformMatrix4fv !== 'function') continue;
    const orig = C.prototype.uniformMatrix4fv;
    C.prototype.uniformMatrix4fv = function (loc, transpose, data) {
      try {
        const f = data && data.length === 16 && allowed() ? fovFromMatrix(data) : null;
        if (f && this.canvas) {
          f.seq = ++S.seq;
          this.canvas.__geoscrFov = f;
        }
      } catch (e) { /* never break the renderer */ }
      return orig.apply(this, arguments);
    };
  }

  // 2D renderer: count draws per canvas, so a capture can wait until the new view is painted
  const C2 = window.CanvasRenderingContext2D;
  for (const m of C2 ? ['drawImage', 'putImageData'] : []) {
    const orig = C2.prototype[m];
    if (typeof orig !== 'function') continue;
    C2.prototype[m] = function () {
      try {
        if (this.canvas && allowed()) this.canvas.__geoscrDraws = (this.canvas.__geoscrDraws || 0) + 1;
      } catch (e) { /* never break the page */ }
      return orig.apply(this, arguments);
    };
  }

  const note = (key, m) => {
    if (!key || !m) return;
    S.store[key] = mergeMeta(S.store[key] || {}, m);
    if (m.game && m.game.token) {
      const gk = '/game/' + m.game.token;
      S.store[gk] = mergeMeta(S.store[gk] || {}, { game: m.game, map: m.map });
    }
  };
  const origFetch = window.fetch;
  if (typeof origFetch === 'function') {
    window.fetch = function (input) {
      const p = origFetch.apply(window, arguments);
      try {
        const url = input && typeof input === 'object' && 'url' in input ? input.url : String(input);
        const key = allowed() ? metaKey(url, location.origin) : null;
        if (key) {
          p.then(r => {
            if (r && r.ok && /json/.test(r.headers.get('content-type') || '')) {
              r.clone().json().then(j => note(key, key.startsWith('map:') ? { map: safeMapMeta(j) } : extractMeta(j)), () => null);
            }
          }, () => null);
        }
      } catch (e) { /* never break the page */ }
      return p;
    };
  }
  const readNextData = () => {
    if (!allowed()) return;
    const el = document.getElementById('__NEXT_DATA__');
    if (!el) return;
    try {
      const j = JSON.parse(el.textContent);
      const m = /^\/(game|challenge)\/\[token\]/.exec(String(j.page || ''));
      const token = j.query && typeof j.query.token === 'string' ? j.query.token : null;
      if (m && token && /^[A-Za-z0-9]+$/.test(token)) note(`/${m[1]}/${token}`, extractMeta(j));
    } catch (e) { /* no page data */ }
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', readNextData);
  else readNextData();

  S.metaFor = p => {
    const m = /^\/(game|challenge)\/([A-Za-z0-9]+)/.exec(p || '');
    if (!m || p !== location.pathname || !allowed()) return null;
    let out = S.store[`/${m[1]}/${m[2]}`] || null;
    const gt = m[1] === 'challenge' && out && out.game && out.game.token;
    if (gt && S.store['/game/' + gt]) out = mergeMeta(out, S.store['/game/' + gt]);
    const mid = out && ((out.game && out.game.mapId) || (out.challenge && out.challenge.mapSlug));
    if (mid && S.store['map:' + mid]) out = mergeMeta(S.store['map:' + mid], out);
    return out;
  };
  S.here = p => location.pathname === p && allowed();
  S.canvas = () => {
    let best = null, area = 0;
    for (const c of document.querySelectorAll('canvas.widget-scene-canvas')) {
      const r = c.getBoundingClientRect();
      if (r.width * r.height > area) { best = c; area = r.width * r.height; }
    }
    return best;
  };
  S.tilesCanvas = () => {
    const c = document.querySelector('canvas.renderCanvas');
    return c && c.getBoundingClientRect().width > 0 ? c : null;
  };
  S.pano = () => {
    const canv = S.canvas();
    const live = S.panos.filter(p => { try { return p.getVisible() !== false; } catch (e) { return false; } });
    return live.find(p => canv && p.__geoscrDiv && p.__geoscrDiv.contains(canv)) ||
      live.sort((a, b) => (b.__geoscrUsed || 0) - (a.__geoscrUsed || 0))[0] || null;
  };
  S.state = () => {
    const canv = S.canvas(), p = S.pano();
    let pov = null, zoom = null, status = null;
    if (p) {
      try { const v = p.getPov(); pov = { heading: v.heading, pitch: v.pitch }; zoom = p.getZoom(); } catch (e) { /* not ready */ }
      try { status = p.getStatus ? p.getStatus() : null; } catch (e) { /* not ready */ }
    }
    const r = canv ? canv.getBoundingClientRect() : null;
    const rn = document.querySelector("[data-qa='round-number']");
    const draws = canv ? canv.__geoscrDraws || 0 : 0;
    const mk = S.mark;
    return {
      id: S.id, path: location.pathname,
      renderer: S.tilesCanvas() ? 'tiles' : p && canv ? 'google' : canv ? 'google-unhooked' : 'none',
      pov, zoom, status, seq: S.seq, draws,
      fov: canv && canv.__geoscrFov ? canv.__geoscrFov : null,
      panoChanges: S.panoChanges, sincePano: S.panoAt ? performance.now() - S.panoAt : null,
      drawnSincePano: !mk || S.seq > mk.seq || (canv === mk.canvas ? draws > mk.draws : draws > 0),
      blur: !!(canv && /blur/.test(canv.style.filter || '')),
      nmpz: !!document.querySelector("[data-qa='panorama'][class*='playingNmpz']"),
      canvas: canv ? { width: canv.width, height: canv.height, cssWidth: r.width, cssHeight: r.height } : null,
      roundText: rn ? rn.innerText.replace(/\s+/g, ' ').trim() : null,
      result: !!document.querySelector("[data-qa='standard-round-result'], [data-qa='close-round-result']"),
    };
  };
  S.setView = (v, p) => {
    const pano = S.here(p) ? S.pano() : null;
    if (!pano) return false;
    if (v.zoom !== undefined && v.zoom !== null) pano.setZoom(v.zoom);
    if (v.heading !== undefined && v.heading !== null) pano.setPov({ heading: v.heading, pitch: v.pitch || 0 });
    return true;
  };
  // forbidRotating unknown: GeoGuessr puts the camera back 25 ms after any change in no-rotate rounds
  S.rotateLocked = async p => {
    const pano = S.here(p) ? S.pano() : null;
    if (!pano) return null;
    const v0 = pano.getPov(), target = (v0.heading + 5) % 360;
    pano.setPov({ heading: target, pitch: v0.pitch });
    await new Promise(r => setTimeout(r, 150));
    const v1 = pano.getPov();
    const locked = Math.abs(((v1.heading - target) % 360 + 540) % 360 - 180) > 2;
    if (!locked) pano.setPov({ heading: v0.heading, pitch: v0.pitch });
    return locked;
  };
  S.loadPano = (v, p) => {   // replay of a finished round only
    const pano = S.here(p) ? S.pano() : null;
    if (!pano) return false;
    pano.setPano(v.pano);
    pano.setPov({ heading: v.heading, pitch: v.pitch });
    pano.setZoom(v.zoom);
    return true;
  };
  S.post = async (url, body, p) => {   // --submit: the guess, answered with whitelisted metadata only
    if (!S.here(p)) return { status: 0, error: 'page changed' };
    const r = await origFetch.call(window, url, { method: 'POST', credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    if (!r.ok) return { status: r.status, error: (await r.text()).slice(0, 150) };
    let meta = null;
    try { meta = extractMeta(await r.json()); } catch (e) { /* no JSON */ }
    note(metaKey(url, location.origin), meta);
    return { status: r.status, meta };
  };
  // Hide everything but the panorama canvas and its ancestors (HUD, game UI, Google logo and
  // attribution); returns the screenshot clip. isolate(false) restores the page.
  S.isolate = (on, p) => {
    for (const [el, v, prio] of S.hidden) {
      if (v) el.style.setProperty('visibility', v, prio); else el.style.removeProperty('visibility');
    }
    S.hidden = [];
    const canv = S.canvas();
    if (!on || !canv || !S.here(p)) return null;
    const keep = new Set();
    for (let e = canv; e; e = e.parentElement) keep.add(e);
    for (const el of document.body.querySelectorAll('*')) {
      if (keep.has(el)) continue;
      S.hidden.push([el, el.style.getPropertyValue('visibility'), el.style.getPropertyPriority('visibility')]);
      el.style.setProperty('visibility', 'hidden', 'important');
    }
    const box = el => { const r = el.getBoundingClientRect(); return { x: r.left, y: r.top, width: r.width, height: r.height }; };
    const cut = (a, b) => {
      const x = Math.max(a.x, b.x), y = Math.max(a.y, b.y);
      return { x, y, width: Math.max(0, Math.min(a.x + a.width, b.x + b.width) - x),
        height: Math.max(0, Math.min(a.y + a.height, b.y + b.height) - y) };
    };
    const cont = document.querySelector('#panorama-container') || document.querySelector("[data-qa='panorama']");
    const inCont = !!(cont && cont.contains(canv));
    const canvasBox = box(canv);
    let clip = inCont ? cut(canvasBox, box(cont)) : canvasBox;
    clip = cut(clip, { x: 0, y: 0, width: window.innerWidth, height: window.innerHeight });
    clip = { x: Math.ceil(clip.x), y: Math.ceil(clip.y), width: Math.floor(clip.width), height: Math.floor(clip.height) };
    return { clip, canvasBox, dpr: window.devicePixelRatio || 1,
      clipTo: inCont ? (cont.id ? '#' + cont.id : "[data-qa='panorama']") : 'canvas.widget-scene-canvas' };
  };
}

function pageScript(paths) {
  return `(() => { ${safeBounds} ${safeMapMeta} ${safeGameMeta} ${challengeMeta} ${extractMeta} ${mergeMeta} ${metaKey}
    ${fovFromMatrix} (${installPageHooks})(${JSON.stringify(paths.map(re => re.source))}); })();`;
}
const PAGE_SCRIPT = pageScript(ALLOWED_PATHS);

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
// The script's own GeoGuessr requests: >= 1.2 s apart, stop at the first 429.
async function requestSlot() {
  if (rateLimited) throw new Error('GeoGuessr returned 429 earlier: no more requests');
  const wait = lastRequestAt + 1200 - Date.now();
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

function saveCapture(dir, cap, info, meta) {
  fs.mkdirSync(dir, { recursive: true });
  const dpr = cap.iso.dpr;
  const views = cap.views.map((v, i) => {
    v.file = `view_${String(i).padStart(2, '0')}.jpg`;
    fs.writeFileSync(path.join(dir, v.file), v.buf);
    return { file: v.file, yaw: v.yaw, pitch: v.pitch, hfov: v.hfov, vfov: v.vfov, zoom: v.zoom, fov_method: v.fov_method };
  });
  const json = {
    game: info.key, round: info.round, mode: cap.mode, captured_at: new Date().toISOString(), zoom_locked: cap.zoomLocked,
    fov: cap.fov, grid: cap.grid, canvas: cap.canvas, canvas_box: cap.iso.canvasBox, clip: cap.iso.clip,
    clip_to: cap.iso.clipTo, dpr, image: [Math.round(cap.iso.clip.width * dpr), Math.round(cap.iso.clip.height * dpr)], views,
  };
  fs.writeFileSync(path.join(dir, 'views.json'), JSON.stringify(json, null, 1));
  fs.writeFileSync(path.join(dir, 'meta.json'), JSON.stringify({ map: serverMap(meta), game: meta.game || null,
    challenge: meta.challenge || null }, null, 1));
}

async function predict(views, map) {
  const body = { views: views.map(v => ({ image_b64: v.buf.toString('base64'), yaw: v.yaw, pitch: v.pitch,
    hfov: v.hfov, vfov: v.vfov })) };
  if (map) body.map = map;
  const resp = await fetch(`${SERVER}/api/predict`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  });
  const data = await resp.json();
  if (data.error) throw new Error(data.error);
  return data;
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

async function analyseRound(ctx, info) {
  const { page } = ctx;
  ctx.capturePath = info.path;
  const dir = path.join(OUT_ROOT, `${info.key}_r${info.round}`);
  const hud = s => showHud(page, Object.assign({ round: info.label }, s), info.path);
  await hud({ status: 'Чекаю панораму…' });
  const st = await waitPanorama(ctx, info.since);
  ctx.lastMark = { id: st.id, panoChanges: st.panoChanges };
  const flags = gameFlags(info.meta, st.nmpz);
  let noRotate = flags.noRotate;
  if (noRotate === null) {
    noRotate = await page.evaluate(p => window.__geoscr.rotateLocked(p), info.path);
    if (noRotate === null) throw new Error('page changed during the capture');
    console.log(`   forbidRotating невідомий; перевірка камери: ${noRotate ? 'без обертання' : 'обертання дозволене'}`);
  }
  const zoomLocked = info.freeZoom ? false : flags.zoomLocked;
  await hud({ status: noRotate ? 'Без обертання: знімаю поточний кадр…' : 'Знімаю панораму…' });
  const t0 = Date.now();
  const cap = noRotate ? await captureSingle(ctx) : await captureSweep(ctx, st, dir, zoomLocked);
  const map = serverMap(info.meta);
  saveCapture(dir, cap, info, info.meta);
  const tCap = Date.now() - t0;
  await hud({ status: 'Аналіз…' });
  const result = await predict(cap.views, map);
  fs.writeFileSync(path.join(dir, 'response.json'), JSON.stringify(result, null, 1));
  const top = result.countries[0];
  console.log(`[${info.label}] ${result.countries.map(x => `${x.code} ${(x.probability * 100).toFixed(0)}%`).join(', ')}` +
    `  -> ${result.guess.lat}, ${result.guess.lng}  (capture ${(tCap / 1000).toFixed(1)} s, map ${map ? map.name || map.id : '?'})`);
  console.log('   видно: ' + result.observations.map(o => o.text).join('; '));
  if (result.hints[0] && result.hints[0].regions.length)
    console.log('   регіон: ' + result.hints[0].regions.map(x => `${x.name} ${Math.round(x.probability * 100)}%`).join(', '));
  console.log(`   збережено: ${path.relative(ROOT, dir)}`);
  await hud({ status: `Найімовірніше: ${top.name}`, result });
  return { result, dir };
}

// ------------------------------------------------------------------ HUD

async function showHud(page, state, expectPath = null) {
  try {
    await page.evaluate((s, expectPath) => {
      if (expectPath && location.pathname !== expectPath) return;
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
        <div style="color:${s.error ? '#f87171' : '#94a3b8'};margin-bottom:8px">${esc(s.status || '')}</div>`;
      const r = s.result;
      if (r) {
        for (const c of r.countries) {
          const w = Math.max(2, Math.round(c.probability * 100));
          h += `<div style="display:flex;align-items:center;gap:6px;margin:3px 0"><span style="width:150px">${esc(c.name)}</span>
            <div style="height:7px;width:${w}%;max-width:150px;background:#10b981;border-radius:4px"></div>
            <span>${(c.probability * 100).toFixed(1)}%</span></div>`;
        }
        h += `<div style="color:#94a3b8;margin:6px 0">Здогадка ${esc(r.guess.lat)}, ${esc(r.guess.lng)} · ~${esc(r.guess.expected_score)} балів</div>`;
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
    }, state, expectPath);
  } catch (e) { /* page navigated */ }
}

async function hideHud(page) {
  await page.evaluate(() => { const h = document.getElementById('geoscr-hud'); if (h) h.remove(); }).catch(() => null);
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
  const last = await submitGuess(ctx, result.guess, info.path);
  const pts = last ? last.points : 0;
  const km = last ? Math.round(last.distanceMeters / 100) / 10 : null;
  ctx.total += pts || 0;
  console.log(`   результат: ${pts} балів, ${km} км (всього ${ctx.total})`);
  await page.reload({ waitUntil: 'domcontentloaded', timeout: 45000 });
  await page.waitForSelector("[data-qa='close-round-result']", { timeout: 20000 });
  await showHud(page, { round: info.label, status: 'Результат', result, score: { points: pts, distance: km, total: ctx.total } }, info.path);
  await sleep(4000);
  if (/^\/game\//.test(info.path))
    console.log('   (якщо з’явиться вікно квитка, закрийте його, не натискайте Play: це витрачає квиток)');
  assertPath(ctx);
  const st = await pageState(page);
  if (st) ctx.lastMark = { id: st.id, panoChanges: st.panoChanges };
  await page.click("[data-qa='close-round-result']");
}

async function watchLoop(ctx) {
  const { page, opts } = ctx;
  let lastKey = null, notice = null;
  console.log('Режим спостереження: грайте у вікні браузера, підказки з’являтимуться на кожному раунді.');
  for (;;) {
    await sleep(800);
    if (page.isClosed()) return;
    let url;
    try { url = new URL(page.url()); } catch (e) { continue; }
    if (url.hostname !== 'www.geoguessr.com') continue;
    const p = url.pathname;
    if (!isAllowedPath(p)) {
      if (notice !== p) { notice = p; await showHud(page, { status: NOT_SUPPORTED }); }
      continue;
    }
    const meta = await readMeta(ctx, p);
    const verdict = captureVerdict(p, meta);
    if (!verdict.ok) {
      const msg = verdict.refuse ? `Тут не підтримується: ${verdict.refuse}` : `Чекаю дані гри (${verdict.wait})…`;
      if (notice !== p + msg) { notice = p + msg; await showHud(page, { status: msg, error: !!verdict.refuse }, p); }
      continue;
    }
    if (notice) { notice = null; await hideHud(page); }
    let st;
    try { st = await pageState(page); } catch (e) { continue; }
    if (!st || st.path !== p || st.result || st.renderer === 'none') continue;
    const pr = parseRound(st.roundText);
    const round = pr ? pr.round : meta.game && meta.game.round;
    if (!round) continue;
    const key = `${p}#${round}`;
    if (key === lastKey) continue;
    lastKey = key;
    const of = pr ? pr.of : meta.game && meta.game.roundCount;
    const info = { key: p.split('/')[2], round, label: `Раунд ${round}${of ? '/' + of : ''}`, path: p, meta,
      since: ctx.lastMark && ctx.lastMark.id === st.id ? ctx.lastMark : null };
    try {
      const { result } = await analyseRound(ctx, info);
      if (opts.submit) await submitAndContinue(ctx, result, info);
    } catch (e) {
      console.error('[!] ' + e.message);
      await showHud(page, { round: info.label, status: e.message, error: true }, p);
    }
  }
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

async function runReplay(ctx, fin) {
  const { page, opts } = ctx;
  const n = fin.round;
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
  const p = `/game/${opts.replay}/replay`;
  ctx.capturePath = p;
  await page.goto(`${SITE}${p}?round=${n}&step=0`, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForSelector('canvas.widget-scene-canvas', { timeout: 30000 });
  await sleep(3000);
  assertPath(ctx);
  const st = await pageState(page);
  if (!st || st.status !== 'OK') {
    console.log('   повтор не записано: показую панораму завершеного раунду в панорамі сторінки');
    if (!(await page.evaluate((v, p) => window.__geoscr.loadPano(v, p), fin.view, p))) throw new Error('no panorama to load the round into');
  }
  const meta = mergeMeta({ map: fin.map }, await readMeta(ctx, p));
  const info = { key: opts.replay, round: n, label: `Повтор ${n}`, path: p, meta, since: null, freeZoom: opts.playerZoom === null };
  const dir = path.join(OUT_ROOT, `${info.key}_r${info.round}`);
  if (opts.fovTable) await fovTable(ctx, dir);
  const { result } = await analyseRound(ctx, info);
  const resp = await fetch(`${SERVER}/api/evaluate_round`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ lat: fin.truth.lat, lng: fin.truth.lng, guess_lat: result.guess.lat, guess_lng: result.guess.lng,
      pred_code: result.countries[0].code, map: serverMap(meta) }),
  }).then(r => r.json()).catch(e => ({ error: e.message }));
  fs.writeFileSync(path.join(dir, 'evaluation.json'), JSON.stringify(resp, null, 1));
  console.log(`   перевірка (завершений раунд): справжня країна ${resp.true_code}, топ-1 ${resp.pred_code}, ` +
    `${resp.distance_km} км, ${resp.points} балів`);
}

function parseArgs(argv) {
  const val = (name, def = null) => { const i = argv.indexOf(name); return i >= 0 && i + 1 < argv.length ? argv[i + 1] : def; };
  const has = name => argv.includes(name);
  return {
    challenge: val('--challenge'), game: val('--game'), replay: val('--replay'), round: val('--round', '1'),
    submit: has('--submit'), headless: has('--headless'), webgl: has('--webgl'), fovTable: has('--fov-table'),
    keepOpen: has('--keep-open'), hfov: Math.min(125, Math.max(60, parseFloat(val('--hfov', '125')))),
    playerZoom: val('--player-zoom') === null ? null : Math.min(3, Math.max(0, parseFloat(val('--player-zoom')) || 0)),
  };
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  for (const k of ['challenge', 'game', 'replay']) {
    if (opts[k] && !/^[A-Za-z0-9]+$/.test(opts[k])) { console.error(`[!] bad --${k} token`); process.exit(1); }
  }
  if (opts.submit && opts.replay) { console.error('[!] --submit cannot be used with --replay'); process.exit(1); }
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
  const browser = await puppeteer.launch({ headless: opts.headless, args,
    defaultViewport: opts.headless ? { width: 1400, height: 800 } : null });
  browser.on('disconnected', () => process.exit(process.exitCode || 0));
  const page = (await browser.pages())[0] || await browser.newPage();
  const paths = opts.replay ? ALLOWED_PATHS.concat([REPLAY_PATH]) : ALLOWED_PATHS;
  await page.evaluateOnNewDocument(pageScript(paths));
  if (opts.webgl) await page.evaluateOnNewDocument(`(${maskSoftwareGl})(${JSON.stringify(paths.map(re => re.source))});`);
  const cookie = getCookie();
  if (cookie) await page.setCookie({ name: '_ncfa', value: cookie, domain: '.geoguessr.com' });
  const ctx = { page, opts, metaCache: {}, fovCache: loadFovCache(), zoomChoice: {}, total: 0, lastMark: null,
    capturePath: null, waitTiles: trackTiles(page) };
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
      await watchLoop(ctx);
    }
  } catch (e) {
    failed = true;
    console.error('[!] ' + (e.stack || e));
  } finally {
    if (opts.replay && !opts.keepOpen) await browser.close();
  }
  if (failed) process.exitCode = 1;
}

module.exports = {
  isAllowedPath, safeBounds, safeMapMeta, safeGameMeta, challengeMeta, extractMeta, mergeMeta, metaKey, serverMap,
  parseRound, captureVerdict, submitProblem, gameFlags, clipProblem, replayRound, fovFromMatrix, hfovFromVfov,
  vfovFromHfov, hfovForZoom, zoomForHfov, formulaFov, inFrustum, planGrid, pageScript, PAGE_SCRIPT, ALLOWED_PATHS,
  REPLAY_PATH, trackTiles, waitPanorama,
};

if (require.main === module) main();
