/*
 * Page-world part of the live helper, shared by the Chrome extension (content script in the MAIN
 * world at document_start) and play_live_visual.js (evaluateOnNewDocument). It holds the fair-play
 * metadata whitelist, the StreetViewPanorama / WebGL / fetch hooks, the FOV measurement, the camera
 * sweep with canvas readback and the round controller that drives the HUD (web/hud/hud.js).
 * Under node (require) it only exports the pure helpers.
 *
 * Fair play: the round's location is never read. Nothing here requests the game, clue or location
 * endpoints or reads the panorama id / position; the page's own game responses are reduced to
 * settings by safeGameMeta() before they are stored. Capture only on single-player /challenge/<t>
 * and /game/<t> pages (game type standard or challenge), plus /game/<t>/replay of finished games
 * when the replay test mode is on. Outside those pages every hook is pass-through and nothing is
 * shown. No guess is ever submitted from here (S.post exists only when play_live_visual.js runs
 * with --submit).
 *
 * Footprint: the hooks must exist on every page (GeoGuessr is a single-page app), but nothing else
 * is visible to the page's own scripts: no global (window.__geoscr only for play_live_visual.js and
 * in the replay test mode), no window messages after the first one (the content script hands over a
 * private MessagePort, which the listener below takes before any page script can see it), no DOM
 * outside the allowed pages.
 *
 * Config (window.__geoscrConfig from play_live_visual.js, else localStorage "__geoscr_cfg", which
 * the extension writes only while the replay test mode is on):
 *   {replay: bool, preserve: bool (default true), submit: bool}
 * Transport to the locator: window.__geoscrNode (puppeteer) or the content script's MessagePort.
 */
(function (factory) {
  const api = factory();
  if (typeof module === 'object' && module && module.exports) module.exports = api;
  else api.boot();
})(function () {
  'use strict';

  const API = 2;                 // interface version shared with web/hud/hud.js
  const ALLOWED_PATHS = [/^\/challenge\/[A-Za-z0-9]+$/, /^\/game\/[A-Za-z0-9]+$/];
  const REPLAY_PATH = /^\/game\/[A-Za-z0-9]+\/replay$/;
  const SINGLE_PLAYER_TYPES = ['standard', 'challenge'];
  const TILE_RE = /streetviewpixels|\/cbk\?|GeoPhotoService|photometa|ggpht\.com|googleusercontent\.com/;
  const RAD = Math.PI / 180;
  const OVERLAP = 0.10;          // minimum overlap of neighbouring views
  const PITCH_LIMIT = 85;        // the sweep covers -85..+85 deg
  const MAX_VFOV = 90;           // Google clamps the vertical FOV (measured in both renderers)
  const SETTLE_MS = 1000;        // a new panorama is captured no earlier than this after its pano change
  const HFOV_WANTED = 125;
  const MAX_WIDTH = 1280;        // views are scaled down to this width (the sphere is 2048 px wide)
  const NOT_SUPPORTED = 'Тут не підтримується (мультиплеєр або інша сторінка). Підказки працюють лише в одиночних /challenge/… та /game/….';
  const PAGE_CHANGED = 'сторінка змінилася під час знімання';
  const PANO_CHANGED = 'панорама змінилася під час знімання';

  // ---------------------------------------------------------------- fair-play metadata whitelist
  // These read only the whitelisted keys; nothing touches rounds, coordinates, pano ids or streak codes.

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

  // forbidRotating (null = unknown, probe the camera) and whether the zoom must stay as the player has it.
  function gameFlags(meta, nmpz) {
    const g = (meta && meta.game) || {}, c = (meta && meta.challenge) || {};
    const flag = k => (typeof g[k] === 'boolean' ? g[k] : typeof c[k] === 'boolean' ? c[k] : null);
    const rotate = flag('forbidRotating');
    return { noRotate: rotate !== null ? rotate : nmpz ? true : null, zoomLocked: flag('forbidZooming') !== false };
  }

  // ---------------------------------------------------------------- camera geometry

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
  // reaches 90 deg.
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

  const matrixFov = (canvas, f) => ({ hfov: hfovFromVfov(f.vfov, canvas.width / canvas.height), vfov: f.vfov,
    hfovMatrix: f.hfov, method: 'webgl-matrix' });

  // Payload of a "predict" request as the locator server takes it (no extra keys pass through).
  function predictBody(views, map) {
    const body = { views: (views || []).map(v => ({ image_b64: String(v.image_b64), yaw: +v.yaw, pitch: +v.pitch,
      hfov: +v.hfov, vfov: +v.vfov })) };
    const m = map && typeof map === 'object' ? map : null;
    if (m) {
      body.map = {};
      for (const k of ['id', 'slug', 'name', 'bounds', 'maxErrorDistance']) if (m[k] !== undefined && m[k] !== null) body.map[k] = m[k];
    }
    return body;
  }

  // ---------------------------------------------------------------- page-side hooks

  // Runs in every page before any script and must stay pass-through: outside the allowed paths
  // nothing is remembered, recorded or read. On allowed pages it remembers the StreetViewPanorama
  // (constructor + prototype hook; GeoGuessr builds it from the global google.maps), counts pano
  // changes and draws, records Google's projection matrix, keeps the WebGL drawing buffer readable
  // and keeps the whitelisted metadata of the page's own game / challenge responses.
  function installPageHooks(cfg) {
    if (window.__geoscr) return window.__geoscr;
    cfg = Object.assign({ replay: false, preserve: true, submit: false, expose: false }, cfg || {});
    // called from every WebGL draw: cached per path and replay switch
    let memo = null;
    const allowed = () => {
      const k = location.pathname + (cfg.replay ? '|r' : '');
      if (!memo || memo.k !== k) memo = { k, ok: isAllowedPath(location.pathname) || (!!cfg.replay && REPLAY_PATH.test(location.pathname)) };
      return memo.ok;
    };
    const S = { id: Math.random().toString(36).slice(2), api: API, cfg, panos: [], seq: 0, hidden: [],
      store: {}, panoChanges: 0, panoAt: 0, mark: null, tileAt: 0, onDraw: null };
    if (cfg.expose) window.__geoscr = S;   // play_live_visual.js and the replay test mode only
    S.allowed = allowed;
    S.setReplay = on => { cfg.replay = !!on; };
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

    // A readable WebGL canvas: the drawing buffer is kept after compositing (allowed pages only).
    const HC = window.HTMLCanvasElement;
    if (HC && HC.prototype && typeof HC.prototype.getContext === 'function' && cfg.preserve) {
      const gc = HC.prototype.getContext;
      HC.prototype.getContext = function (type, attrs) {
        let a = attrs;
        try {
          if (/^(webgl2?|experimental-webgl)$/.test(type) && allowed())
            a = Object.assign({}, attrs && typeof attrs === 'object' ? attrs : {}, { preserveDrawingBuffer: true });
        } catch (e) { a = attrs; }
        const ctx = a === attrs ? gc.apply(this, arguments) : gc.call(this, type, a);
        try { if (ctx && a !== attrs) this.__geoscrGl = ctx; } catch (e) { /* never break the page */ }
        return ctx;
      };
    }

    const drawn = c => {
      c.__geoscrDraws = (c.__geoscrDraws || 0) + 1;
      c.__geoscrDrawAt = performance.now();
      if (S.onDraw && !S.drawQueued) {
        S.drawQueued = true;
        // after the current task (the whole frame) and before the frame is composited
        Promise.resolve().then(() => { S.drawQueued = false; const f = S.onDraw; if (f) f(c); });
      }
    };
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
      for (const m of ['drawArrays', 'drawElements']) {
        const od = C.prototype[m];
        if (typeof od !== 'function') continue;
        C.prototype[m] = function () {
          const r = od.apply(this, arguments);
          try { if (this.canvas && allowed()) drawn(this.canvas); } catch (e) { /* never break the renderer */ }
          return r;
        };
      }
    }

    // 2D renderer: count draws per canvas, so a capture can wait until the new view is painted
    const C2 = window.CanvasRenderingContext2D;
    for (const m of C2 ? ['drawImage', 'putImageData'] : []) {
      const orig = C2.prototype[m];
      if (typeof orig !== 'function') continue;
      C2.prototype[m] = function () {
        const r = orig.apply(this, arguments);
        try { if (this.canvas && !this.canvas.__geoscrOwn && allowed()) drawn(this.canvas); } catch (e) { /* never break the page */ }
        return r;
      };
    }

    // Street View tile downloads (names only, from the resource timing of allowed pages)
    try {
      new PerformanceObserver(list => {
        if (!allowed()) return;
        for (const e of list.getEntries()) if (TILE_RE.test(e.name)) S.tileAt = performance.now();
      }).observe({ type: 'resource', buffered: false });
    } catch (e) { /* no PerformanceObserver */ }

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
      const canv = S.canvas(), p = allowed() ? S.pano() : null;
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
    // Street View draws its navigation arrows into the panorama itself (on the road): switch them off
    // while capturing, put the player's setting back afterwards.
    S.arrows = (show, p) => {
      const pano = S.here(p) ? S.pano() : null;
      if (!pano) return;
      try {
        if (!show && S.arrowsWas === undefined) {
          S.arrowsWas = pano.get('linksControl') !== false;
          if (S.arrowsWas) pano.setOptions({ linksControl: false });
        } else if (show && S.arrowsWas !== undefined) {
          if (S.arrowsWas) pano.setOptions({ linksControl: true });
          S.arrowsWas = undefined;
        }
      } catch (e) { /* not a Maps JS panorama */ }
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
      const pano = REPLAY_PATH.test(p || '') && S.here(p) ? S.pano() : null;
      if (!pano) return false;
      pano.setPano(v.pano);
      pano.setPov({ heading: v.heading, pitch: v.pitch });
      pano.setZoom(v.zoom);
      return true;
    };
    if (cfg.submit) {
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
    }
    // Hide everything but the panorama canvas and its ancestors (HUD, game UI, Google logo and
    // attribution); returns the screenshot clip. isolate(false) restores the page.
    S.isolate = (on, p) => {
      for (const [el, v, prio] of S.hidden) {
        if (v) el.style.setProperty('visibility', v, prio); else el.style.removeProperty('visibility');
      }
      S.hidden = [];
      S.arrows(!on, p);
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

    installCapture(S);
    installController(S);
    return S;
  }

  // ---------------------------------------------------------------- in-page capture (canvas readback)

  const sleep = ms => new Promise(r => setTimeout(r, ms));

  function installCapture(S) {
    const now = () => performance.now();
    const fail = msg => { throw new Error(msg); };
    const check = p => { if (!S.here(p)) fail(PAGE_CHANGED); };

    // Wait until the view is painted and stable: a draw at or after `after` (the setView; a view that
    // needs no redraw is accepted after 400 ms), then no draw and no Street View tile for idleMs (tiles
    // arrive progressively and each one triggers a redraw).
    S.waitSettled = async ({ after = null, minMs = 0, idleMs = S.idleMs || 150, maxMs = 3000 } = {}) => {
      const t0 = now();
      if (minMs) await sleep(minMs);
      for (;;) {
        const c = S.canvas();
        const drawAt = c && c.__geoscrDrawAt || 0;
        const drew = after === null || drawAt >= after || now() - t0 > 400;
        if (now() - t0 >= maxMs || (drew && now() - Math.max(drawAt, S.tileAt || 0) >= idleMs)) break;
        await sleep(25);
      }
      await new Promise(r => { requestAnimationFrame(() => requestAnimationFrame(r)); setTimeout(r, 500); });
    };

    const glOf = c => {
      if (c.__geoscrGl) return c.__geoscrGl;
      for (const t of ['webgl2', 'webgl']) {
        try { const g = c.getContext(t); if (g) return g; } catch (e) { /* other context type */ }
      }
      return null;
    };
    const readCanvas = (c, maxW) => {
      const s = Math.min(1, maxW / c.width);
      const w = Math.max(1, Math.round(c.width * s)), h = Math.max(1, Math.round(c.height * s));
      const t = document.createElement('canvas');
      t.__geoscrOwn = true;
      t.width = w; t.height = h;
      const g = t.getContext('2d');
      g.drawImage(c, 0, 0, w, h);
      const k = document.createElement('canvas');
      k.__geoscrOwn = true;
      k.width = 16; k.height = 8;
      const kg = k.getContext('2d', { willReadFrequently: true });
      kg.drawImage(t, 0, 0, 16, 8);
      const d = kg.getImageData(0, 0, 16, 8).data;
      let mx = 0;
      for (let i = 0; i < d.length; i += 4) mx = Math.max(mx, Math.min(d[i + 3], Math.max(d[i], d[i + 1], d[i + 2])));
      return { url: t.toDataURL('image/jpeg', 0.92), width: w, height: h, blank: mx < 4 };
    };

    // The current frame of the panorama canvas as a JPEG data URL. A preserved (or 2D) canvas is read
    // directly; otherwise the next frame is read right after it is drawn (measured redraw).
    S.grab = (p, maxW = MAX_WIDTH) => {
      check(p);
      const c = S.canvas();
      if (!c) fail('панорами немає на сторінці');
      const gl = glOf(c);
      let attrs = null;
      try { attrs = gl ? gl.getContextAttributes() : null; } catch (e) { attrs = null; }
      if (!gl || (attrs && attrs.preserveDrawingBuffer)) {
        const r = readCanvas(c, maxW);
        if (!r.blank) return Promise.resolve(Object.assign(r, { method: gl ? 'preserved' : '2d' }));
      }
      return new Promise((resolve, reject) => {
        const timer = setTimeout(() => { S.onDraw = null; reject(new Error('кадр не вдалося прочитати (canvas не читається)')); }, 3000);
        S.onDraw = cv => {
          if (cv !== c) return;
          S.onDraw = null;
          clearTimeout(timer);
          try { resolve(Object.assign(readCanvas(c, maxW), { method: 'redraw' })); } catch (e) { reject(e); }
        };
        const pano = S.pano();
        try { const v = pano.getPov(); pano.setPov({ heading: v.heading + 1e-3, pitch: v.pitch }); } catch (e) { /* draw anyway */ }
      });
    };

    // GeoGuessr reuses one StreetViewPanorama for all rounds, so a new round is ready only after a pano
    // change (after `since`, the count when the previous round ended), a fresh draw, SETTLE_MS and idle tiles.
    S.waitPanorama = async (p, since, timeoutMs = 20000) => {
      const t0 = now();
      let moved = !since;
      for (;;) {
        check(p);
        const st = S.state();
        if (st.renderer === 'tiles') fail('непідтримуваний рендерер: панорама GeoGuessr (Baidu / GG_)');
        if (!moved && st.panoChanges !== since.panoChanges) moved = true;
        if (!moved && now() - t0 > 12000) moved = true;   // no pano change seen: capture what is shown
        if (moved && st.renderer === 'google' && st.status === 'OK' && st.pov && !st.blur && st.canvas &&
          st.drawnSincePano && (st.sincePano === null || st.sincePano >= SETTLE_MS)) {
          await S.waitSettled({ idleMs: 400, maxMs: 4000 });
          return S.state();
        }
        if (now() - t0 > timeoutMs) {
          if (st.renderer === 'none') fail('панорами немає на сторінці');
          if (st.renderer === 'google-unhooked') fail('об’єкт Street View не перехоплено (hook не спрацював)');
          if (st.blur) fail('панорама розмита (затиснута кнопка миші?)');
          fail(`панорама не завантажилась (статус ${st.status})`);
        }
        await sleep(200);
      }
    };

    // Point the camera, wait for its draw and the tiles, read back POV / zoom / projection and take the
    // frame. pano: the pano-change count at the start of the capture (a move or a new round aborts it).
    const grabAt = async (p, view, maxW, pano) => {
      const before = S.state();
      const drawnSince = s => (s.fov ? s.fov.seq > before.seq : s.draws > before.draws);
      const t = now();
      if (!S.setView(view, p)) fail(PAGE_CHANGED);
      await S.waitSettled({ after: t });
      let st = S.state();
      for (let i = 0; !drawnSince(st) && i < 20; i++) {
        if (i === 10) S.setView({ heading: view.heading + 0.01, pitch: view.pitch }, p);
        await sleep(100);
        st = S.state();
      }
      check(p);
      if (S.panoChanges !== pano) fail(PANO_CHANGED);
      const img = await S.grab(p, maxW);
      st = S.state();
      const fresh = st.fov && st.fov.seq > before.seq ? st.fov : null;
      return { img, st, fresh };
    };

    // FOV at `zoom`: Google's projection matrix of the next frame (WebGL renderer, kept per canvas size
    // and zoom for this page) or the formula (2D renderer, which has no matrix). No frame is read.
    const fovSeen = {};
    const measureFov = async (p, zoom, base) => {
      const st0 = S.state();
      const key = `${st0.canvas.width}x${st0.canvas.height}|${zoom.toFixed(3)}`;
      if (fovSeen[key]) return Object.assign({}, fovSeen[key]);
      if (!S.setView({ zoom, heading: base.heading + 1, pitch: 0 }, p)) fail(PAGE_CHANGED);
      let st = S.state();
      for (let i = 0; st0.fov && !(st.fov && st.fov.seq > st0.seq) && i < 40; i++) {
        await sleep(40);
        check(p);
        st = S.state();
      }
      if (st.fov && st.fov.seq > st0.seq) return Object.assign({}, fovSeen[key] = Object.assign({ zoom: st.zoom }, matrixFov(st.canvas, st.fov)));
      return Object.assign({ zoom: st.zoom }, formulaFov(st.zoom, st.canvas.width / st.canvas.height));
    };

    // Zoom whose measured horizontal FOV is close to the target. Google caps the vertical FOV at 90 deg,
    // so the widest view depends on the canvas aspect. fixedZoom: keep the player's zoom (no-zoom games).
    const chooseZoom = async (p, base, canvas, fixedZoom, wantH) => {
      if (fixedZoom !== null) {
        const st = S.state();
        const f = st.fov ? matrixFov(st.canvas, st.fov) : formulaFov(st.zoom, canvas.width / canvas.height);
        return Object.assign({ zoom: fixedZoom }, f);
      }
      const widest = hfovFromVfov(MAX_VFOV, canvas.width / canvas.height);
      const target = Math.min(wantH, widest);
      let z = target >= widest - 0.01 ? 0 : Math.max(0, zoomForHfov(target)), best = null, last = null;
      for (let i = 0; i < 3; i++) {
        const f = Object.assign(await measureFov(p, z, base), { zoom: Math.round(z * 1000) / 1000 });
        if (!best || Math.abs(f.hfov - target) < Math.abs(best.hfov - target)) best = f;
        if (Math.abs(f.hfov - target) < 2 || f.method === 'formula' || (last && Math.abs(f.hfov - last.hfov) < 0.3)) break;
        last = f;
        const nz = Math.min(3, Math.max(0, z + Math.log2(Math.tan(f.hfov * RAD / 2) / Math.tan(target * RAD / 2))));
        if (Math.abs(nz - z) < 0.01) break;
        z = nz;
      }
      return best;
    };

    const view = (img, st, f, method) => ({ image_b64: img.url, width: img.width, height: img.height, yaw: st.pov.heading,
      pitch: st.pov.pitch, hfov: f.hfov, vfov: f.vfov, zoom: st.zoom, fov_method: method, read: img.method });

    // Full capture of the current round: {views, mode, fov, grid, canvas, zoomLocked, noRotate, ms, wait_ms}
    // (ms: the capture itself, wait_ms: waiting for the panorama before it).
    // opts: {since, flags: {noRotate, zoomLocked}, hfov, maxWidth, progress(i, n, text)}
    S.capture = async (p, opts = {}) => {
      const t0 = now();
      const progress = opts.progress || (() => null);
      const maxW = opts.maxWidth || MAX_WIDTH;
      let st = await S.waitPanorama(p, opts.since || null);
      const t1 = now();
      const pano = S.panoChanges;
      const flags = opts.flags || { noRotate: null, zoomLocked: true };
      let noRotate = flags.noRotate === null || flags.noRotate === undefined ? (st.nmpz ? true : null) : flags.noRotate;
      if (noRotate === null) {
        noRotate = await S.rotateLocked(p);
        if (noRotate === null) fail(PAGE_CHANGED);
      }
      if (noRotate) {
        progress(0, 1, 'Без обертання: знімаю поточний кадр…');
        for (const t2 = now(); st.blur && now() - t2 < 15000;) { await sleep(200); st = S.state(); }
        if (st.blur) fail('панорама розмита (затиснута кнопка миші?)');
        if (S.panoChanges !== pano) fail(PANO_CHANGED);
        const f = st.fov ? matrixFov(st.canvas, st.fov) : formulaFov(st.zoom, st.canvas.width / st.canvas.height);
        const img = await S.grab(p, maxW);
        const st2 = S.state();
        progress(1, 1);
        return { views: [view(img, st2, f, f.method)], mode: 'single', fov: f, grid: null, canvas: st.canvas,
          zoomLocked: true, noRotate: true, ms: now() - t1, wait_ms: t1 - t0 };
      }
      const zoomLocked = !!flags.zoomLocked;
      const fixedZoom = zoomLocked ? st.zoom : null;
      const base = { heading: st.pov.heading, pitch: st.pov.pitch, zoom: fixedZoom === null ? st.zoom : null };
      const views = [];
      let choice, grid;
      try {
        S.arrows(false, p);
        progress(0, 0, 'Вимірюю поле зору…');
        choice = await chooseZoom(p, base, st.canvas, fixedZoom, opts.hfov || HFOV_WANTED);
        grid = planGrid(choice.hfov, choice.vfov, base.heading);
        for (const g of grid) {
          progress(views.length, grid.length, 'Знімаю панораму…');
          const r = await grabAt(p, { heading: g.yaw, pitch: g.pitch, zoom: fixedZoom === null ? choice.zoom : null }, maxW, pano);
          const f = r.fresh ? matrixFov(r.st.canvas, r.fresh) : choice;
          views.push(view(r.img, r.st, f, r.fresh ? 'webgl-matrix' : choice.method));
        }
        progress(views.length, grid.length);
      } finally {
        try { if (S.panoChanges === pano) S.setView(base, p); } catch (e) { /* page gone */ }
        try { S.arrows(true, p); } catch (e) { /* page gone */ }
      }
      return { views, mode: 'sweep', fov: choice, grid, canvas: st.canvas, zoomLocked, noRotate: false, ms: now() - t1,
        wait_ms: t1 - t0 };
    };
  }

  // ---------------------------------------------------------------- round controller + HUD

  const TRIGGERS = { auto: 'авто', manual: 'вручну', replay: 'повтор' };
  const READY = 'Готово. Alt+G або кнопка «Аналіз» — аналізувати панораму.';
  const RESULT = 'Результат раунду. Alt+G — аналіз цієї панорами.';
  const same = (a, b) => a === b || JSON.stringify(a) === JSON.stringify(b);

  function installController(S) {
    const C = S.ctl = { attached: false, transport: null, port: null, css: null, settings: { auto: false, capture: 'page', maxWidth: MAX_WIDTH },
      busy: null, lastKey: null, roundAt: 0, since: null, resultKey: null, resultMark: null, rotate: null, healthPath: null,
      last: null, hud: null,
      view: { mode: 'off', status: { text: '', level: 'idle' }, auto: false, result: null, round: '', canLook: false } };
    let seq = 0;
    const waiting = {};

    // The HUD library: bundled before page.js in the extension (its global is removed right away),
    // injected after it by play_live_visual.js.
    let HUD = null;
    const hudLib = () => {
      const g = window.GeoscrHud;
      if (!HUD && g && typeof g.mount === 'function') {
        HUD = g;
        if (!S.cfg.expose) { try { delete window.GeoscrHud; } catch (e) { window.GeoscrHud = undefined; } }
      }
      return HUD;
    };
    hudLib();

    // ---- transport: the extension's content script (private MessagePort) or play_live_visual.js
    const fromExtension = d => {
      if (!d || typeof d.type !== 'string') return;
      if (d.type === 'reply' && waiting[d.id]) {
        const w = waiting[d.id];
        delete waiting[d.id];
        clearTimeout(w.t);
        if (d.ok) w.resolve(d.payload); else w.reject(new Error(d.error || 'помилка'));
      } else if (d.type === 'settings') {
        const s = d.payload || {};
        if (typeof s.replay === 'boolean') S.setReplay(s.replay);
        S.attach(viaExtension, { auto: !!s.auto, capture: 'page' });
      }
    };
    // The content script's first window message carries the port. This listener is registered at
    // document_start, before any page script's, and keeps the message from reaching them.
    window.addEventListener('message', e => {
      const d = e.data;
      if (e.source !== window || !d || d.__geoscr !== 'port' || !e.ports || !e.ports[0]) return;
      e.stopImmediatePropagation();
      if (C.port) return;
      C.port = e.ports[0];
      C.port.onmessage = m => fromExtension(m.data);
      C.port.postMessage({ type: 'hello' });
    }, true);
    const viaExtension = (type, payload) => new Promise((resolve, reject) => {
      if (!C.port) { reject(new Error('немає зв’язку з розширенням')); return; }
      const id = 'g' + (++seq);
      const t = setTimeout(() => { delete waiting[id]; reject(new Error('розширення не відповідає')); }, type === 'predict' ? 180000 : 15000);
      waiting[id] = { resolve, reject, t };
      C.port.postMessage({ id, type, payload });
    });

    C.send = (type, payload) => {
      if (typeof window.__geoscrNode === 'function') return window.__geoscrNode(type, payload);
      if (!C.transport) return Promise.reject(new Error('немає зв’язку з локатором'));
      return C.transport(type, payload);
    };

    S.attach = (transport, settings) => {
      C.transport = transport || C.transport;
      Object.assign(C.settings, settings || {});
      C.view.auto = !!C.settings.auto;
      if (!C.attached) {
        C.attached = true;
        setInterval(() => { tick().catch(() => null); }, 600);
      }
      render();
      return true;
    };

    // ---- HUD (mounted on the allowed pages only)
    const render = () => {
      if (C.view.mode === 'off') { if (C.hud) C.hud.update(C.view); return; }
      const H = hudLib();
      if (!H) return;
      if (C.port && C.css === null) {   // the extension: hud.css comes from the bundle through the content script
        C.css = false;
        C.send('hud-css', null).then(css => { C.css = String(css || ''); }, () => { C.css = ''; }).then(render);
        return;
      }
      if (C.css === false) return;
      if (!C.hud) {
        C.hud = H.mount({ css: C.port ? C.css : null, onAction: (name, arg) => C.action(name, arg) });
        if (H.api !== API) C.view.warning = 'HUD і розширення різних версій: оновіть розширення (chrome://extensions → ⟳) і сторінку.';
      }
      C.hud.update(C.view);
    };
    // Changes the view; the HUD is redrawn only when something changed.
    C.show = patch => {
      patch = patch || {};
      if (Object.keys(patch).every(k => same(C.view[k], patch[k]))) return C.view;
      Object.assign(C.view, patch);
      render();
      return C.view;
    };
    const status = (text, level = 'busy', extra) => C.show(Object.assign({ status: Object.assign({ text, level,
      since: level === 'busy' ? (C.view.status && C.view.status.level === 'busy' && C.view.status.since) || Date.now() : null }, extra || {}) }));

    C.action = (name, arg) => {
      if (name === 'analyse') return C.analyse('manual');
      if (name === 'auto') {
        C.settings.auto = !C.settings.auto;
        C.show({ auto: C.settings.auto });
        C.send('set', { auto: C.settings.auto }).catch(() => null);
        return null;
      }
      if (name === 'look') return C.look(arg);
      return null;
    };

    // ---- current page
    const replayRound = () => {
      const m = /[?&]round=(\d+)/.exec(location.search);
      return m ? +m[1] : null;
    };
    const pageInfo = () => {
      const p = location.pathname;
      const replay = S.cfg.replay && REPLAY_PATH.test(p);
      const meta = S.metaFor(p) || {};
      const verdict = replay ? { ok: true } : captureVerdict(p, meta);
      const st = S.state();
      const pr = parseRound(st.roundText);
      const round = replay ? replayRound() || 1 : pr ? pr.round : meta.game && meta.game.round || null;
      const of = pr ? pr.of : meta.game && meta.game.roundCount || null;
      const label = replay ? `Повтор, раунд ${round}` : round ? `Раунд ${round}${of ? '/' + of : ''}` : '';
      return { path: p, key: p.split('/')[2], round, label, meta, verdict, st, replay };
    };

    const VERDICT_UA = {
      'game type not known yet': 'тип гри ще невідомий',
      'game data of this page not seen yet': 'дані цієї гри ще не отримано',
    };
    const why = v => {
      const t = v.refuse || v.wait || '';
      const m = /game type "(.*)" is not single-player/.exec(t);
      return m ? `тип гри «${m[1]}» не одиночний` : VERDICT_UA[t] || t;
    };

    // The server is checked once per page and, while it is down, at each new round; a failure shows in
    // the HUD of an allowed page only.
    C.checkServer = () => C.send('health', null).then(() => {
      const was = C.serverDown;
      C.serverDown = null;
      if (was && !C.busy && C.view.status.text === was) status(READY, 'idle');
      return true;
    }, e => {
      C.serverDown = String(e && e.message || e);
      if (!C.busy && C.view.mode === 'game' && !C.view.result) status(C.serverDown, 'error');
      return false;
    });

    // Nothing is shown outside single-player pages (multiplayer, Play-Along, menus).
    const off = () => {
      if (C.view.mode !== 'off') C.show({ mode: 'off', result: null, round: '', canLook: false });
      C.lastKey = null;
      C.healthPath = null;
    };

    const tick = async () => {
      if (C.busy) return;
      const p = location.pathname;
      if (!S.allowed()) { off(); return; }
      const info = pageInfo();
      if (info.verdict.refuse) { off(); return; }
      if (!info.verdict.ok) {
        C.show({ mode: 'game', round: info.label, result: null, status: { text: `Чекаю дані гри (${why(info.verdict)})…`, level: 'idle', since: null } });
        return;
      }
      if (C.view.mode !== 'game') C.show({ mode: 'game', round: info.label });
      if (C.healthPath !== p) { C.healthPath = p; C.checkServer(); }
      const st = info.st;
      // the result screen of the current round: the next round's panorama is the first one after it
      if (st.result && C.lastKey && C.resultKey !== C.lastKey) { C.resultKey = C.lastKey; C.resultMark = S.panoChanges; }
      if (!info.round || st.renderer === 'none') {
        if (!C.view.result && C.view.status.level !== 'error') status('Чекаю панораму…', 'idle');
        return;
      }
      const key = `${p}#${info.round}`;
      if (key === C.lastKey) {
        if (st.result && C.view.status.level === 'idle' && !C.view.result) status(RESULT, 'idle');
        return;
      }
      const prev = C.lastKey;
      C.lastKey = key;
      C.roundAt = performance.now();
      C.rotate = null;
      // auto-analyse waits for this round's panorama: a pano change after the previous round's result
      // screen; without one, the pano change of the last 3 s, else the next one
      C.since = prev && C.resultKey === prev ? { panoChanges: C.resultMark }
        : S.panoAt && performance.now() - S.panoAt < 3000 ? null : { panoChanges: S.panoChanges };
      C.show({ round: info.label, result: null, timing: null, score: null, canLook: false });
      if (C.settings.auto && !st.result && !info.replay) C.analyse('auto', info).catch(() => null);
      else status(st.result ? RESULT : READY, 'idle');
      if (C.serverDown) C.checkServer();
    };

    C.tick = tick;

    // ---- analysis of the current panorama
    C.analyse = (trigger = 'manual', info0 = null) => {
      if (C.busy) return C.busy;
      const run = async () => {
        const info = info0 || pageInfo();
        const p = info.path;
        if (!S.allowed() || !info.verdict.ok) {
          status(info.verdict && !info.verdict.ok ? `Тут не можна: ${why(info.verdict)}` : NOT_SUPPORTED, 'error');
          return null;
        }
        const key = `${p}#${info.round}`;
        if (C.lastKey !== key) { C.lastKey = key; C.since = null; C.rotate = null; }
        const t0 = Date.now();
        C.show({ mode: 'game', round: info.label, result: null, timing: null, score: null, canLook: false });
        status('Чекаю панораму…', 'busy', { trigger: TRIGGERS[trigger] || trigger });
        let extra = null;
        if (C.settings.node) extra = await C.send('meta', { path: p }).catch(() => null);
        const meta = extra ? mergeMeta(extra, info.meta) : info.meta;
        const since = trigger === 'auto' ? C.since : null;   // manual: the panorama shown now
        const job = { path: p, key: info.key, round: info.round, label: info.label, trigger, map: serverMap(meta),
          meta: { game: meta.game || null, map: meta.map || null, challenge: meta.challenge || null } };
        let out;
        if (C.settings.capture === 'screenshot') {   // play_live_visual.js --capture screenshot
          out = await C.send('capture', Object.assign({ since }, job));
        } else {
          // replay of a finished game: no game rules apply, the sweep picks its own zoom
          const flags = info.replay ? { noRotate: false, zoomLocked: C.settings.keepZoom === true } : gameFlags(meta, info.st.nmpz);
          const cap = await S.capture(p, { since, flags, maxWidth: C.settings.maxWidth,
            progress: (i, n, text) => status(text || C.view.status.text, 'busy', { progress: n ? [i, n] : null }) });
          C.last = { capture: cap, job };
          status('Аналіз…', 'busy', { progress: null });
          const t1 = Date.now();
          const result = await C.send('predict', { views: cap.views, map: job.map,
            capture: { mode: cap.mode, fov: cap.fov, grid: cap.grid, canvas: cap.canvas, zoomLocked: cap.zoomLocked, ms: cap.ms },
            job });
          out = { result, noRotate: cap.noRotate, timing: { capture_ms: Math.round(cap.ms), wait_ms: Math.round(cap.wait_ms),
            server_ms: result && result.timing_ms ? result.timing_ms.total : null, request_ms: Date.now() - t1,
            views: cap.views.length, mode: cap.mode, fov: cap.fov, read: cap.views[0] && cap.views[0].read } };
        }
        if (!out || !out.result) throw new Error('локатор не повернув результату');
        if (typeof out.noRotate === 'boolean') C.rotate = { key, noRotate: out.noRotate };
        const timing = Object.assign({}, out.timing || {}, { total_ms: Date.now() - t0 });
        C.last = Object.assign(C.last || {}, { result: out.result, timing });
        const top = out.result.countries && out.result.countries[0];
        C.show({ result: out.result, timing, score: out.score || null, canLook: !!info.replay || out.noRotate === false,
          status: { text: top ? `Готово: найімовірніше ${countryName(top)}` : 'Готово', level: 'ok', trigger: TRIGGERS[trigger] || trigger } });
        return out.result;
      };
      C.busy = run().catch(e => {
        status(String(e && e.message || e), 'error');
        return null;
      }).finally(() => { C.busy = null; });
      return C.busy;
    };

    // Point the camera at a card's direction, as the player could: only when rotation is known to be
    // allowed (game settings, the capture's probe of this round, or a probe now), the zoom only when
    // zooming is allowed.
    C.look = async v => {
      const info = pageInfo();
      if (C.busy || !info.verdict.ok || !v || typeof v.heading !== 'number') return false;
      const key = `${info.path}#${info.round}`;
      const flags = info.replay ? { noRotate: false, zoomLocked: true } : gameFlags(info.meta, info.st.nmpz);
      let noRotate = flags.noRotate;
      if (noRotate === null && C.rotate && C.rotate.key === key) noRotate = C.rotate.noRotate;
      if (noRotate === null) {
        noRotate = await S.rotateLocked(info.path);
        if (noRotate === null) return false;
        C.rotate = { key, noRotate };
      }
      if (noRotate !== false) {
        C.show({ canLook: false });
        status('У цій грі обертати камеру не можна.', 'warn');
        return false;
      }
      const view = { heading: v.heading, pitch: typeof v.pitch === 'number' ? v.pitch : 0 };
      if (flags.zoomLocked === false && typeof v.zoom === 'number') view.zoom = v.zoom;
      return S.setView(view, info.path);
    };
  }

  function countryName(c) {
    try {
      const n = new Intl.DisplayNames(['uk'], { type: 'region' }).of(c.code);
      if (n && n !== c.code) return n;
    } catch (e) { /* no Intl.DisplayNames */ }
    return c.name || c.code;
  }

  function boot() {
    let cfg = window.__geoscrConfig;   // play_live_visual.js
    if (cfg && typeof cfg === 'object') cfg = Object.assign({}, cfg, { expose: true });
    else {
      let stored = null;
      try { stored = JSON.parse(localStorage.getItem('__geoscr_cfg') || 'null'); } catch (e) { stored = null; }
      // from storage only the replay switch; that test mode also exposes window.__geoscr for automation
      const replay = !!(stored && stored.replay === true);
      cfg = { replay, expose: replay };
    }
    return installPageHooks(cfg);
  }

  return {
    API, ALLOWED_PATHS, REPLAY_PATH, SINGLE_PLAYER_TYPES, NOT_SUPPORTED, MAX_VFOV,
    isAllowedPath, safeBounds, safeMapMeta, safeGameMeta, challengeMeta, extractMeta, mergeMeta, metaKey, serverMap,
    parseRound, captureVerdict, gameFlags, fovFromMatrix, hfovFromVfov, vfovFromHfov, hfovForZoom, zoomForHfov,
    formulaFov, inFrustum, halfWidth, planGrid, predictBody, countryName, installPageHooks, boot,
  };
});
