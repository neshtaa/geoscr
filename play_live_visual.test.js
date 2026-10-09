// Tests for the pure helpers, the page hooks (extension/page.js), the HUD helpers (web/hud/hud.js) and
// the extension of the live helper:  node --test play_live_visual.test.js
'use strict';
const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');
const vm = require('node:vm');
const L = require('./play_live_visual.js');
const H = require('./web/hud/hud.js');

const SECRET = { lat: 47.1234567, lng: 13.7654321, pano: '4a667848a4e4e4e724c5a506b3555367778616630494' };
const ORIGIN = 'https://www.geoguessr.com';
const EXT = path.join(__dirname, 'extension');

function fakeGame(extra = {}) {
  return Object.assign({
    token: 'x78UXZE1PAnz4oLR', type: 'challenge', mode: 'standard', state: 'started', round: 2, roundCount: 5,
    timeLimit: 180, forbidMoving: false, forbidZooming: false, forbidRotating: true, map: 'world', mapName: 'World',
    bounds: { min: { lat: -54.8, lng: -159.7 }, max: { lat: 78.6, lng: 178.4 } },
    rounds: [{ lat: SECRET.lat, lng: SECRET.lng, panoId: SECRET.pano, heading: 10, pitch: 0, zoom: 0, streakLocationCode: 'at',
      startTime: '2026-10-05T15:33:07.13Z' }],
    player: { id: 'p', guesses: [{ lat: 1.5, lng: 2.5, roundScoreInPoints: 4321, distanceInMeters: 12345.6, timedOut: false }] },
  }, extra);
}

const noSecret = obj => {
  const s = JSON.stringify(obj);
  return !s.includes(String(SECRET.lat)) && !s.includes(String(SECRET.lng)) && !s.includes(SECRET.pano) &&
    !/rounds|panoId|streakLocationCode|startTime/.test(s);
};

test('URL allow-list: single-player challenge and game pages only', () => {
  for (const p of ['/challenge/CM0GZgAucV9Aw4EA', '/game/x78UXZE1PAnz4oLR']) assert.ok(L.isAllowedPath(p), p);
  for (const p of ['/game/x78UXZE1PAnz4oLR/replay', '/challenge/rematch/abc', '/duels/abc', '/team-duels/abc',
    '/battle-royale/abc', '/multiplayer', '/multiplayer/battle-royale-countries', '/party', '/live-challenge/abc',
    '/bullseye/abc', '/play-along/abc', '/play-along/lobby/abc', '/tournaments/1', '/', '/maps/world', '/game/', '/challenge/a-b'])
    assert.ok(!L.isAllowedPath(p), p);
});

test('safeGameMeta copies settings only, never the round location', () => {
  const m = L.safeGameMeta(fakeGame());
  assert.ok(noSecret(m), JSON.stringify(m));
  assert.ok(!/"lat":1.5/.test(JSON.stringify(m)));
  assert.strictEqual(m.forbidRotating, true);
  assert.strictEqual(m.round, 2);
  assert.strictEqual(m.mapId, 'world');
  assert.strictEqual(m.guessCount, 1);
  assert.strictEqual(m.lastGuess.points, 4321);
  assert.deepStrictEqual(m.bounds.min, { lat: -54.8, lng: -159.7 });
  assert.strictEqual(L.safeGameMeta(null), null);
});

// Records every property the whitelist reads from a payload.
function traced(obj, log, at = '') {
  if (!obj || typeof obj !== 'object') return obj;
  return new Proxy(obj, {
    get(t, k, r) {
      const p = `${at}.${String(k)}`;
      log.push(p);
      return traced(Reflect.get(t, k, r), log, p);
    },
    has(t, k) { log.push(`${at}.${String(k)}?`); return Reflect.has(t, k); },
    ownKeys(t) { log.push(`${at}.*`); return Reflect.ownKeys(t); },
  });
}

test('extractMeta never reads rounds, coordinates (outside map bounds) or pano ids', () => {
  const g = fakeGame();
  const payloads = {
    game: g,
    guessResponse: fakeGame({ state: 'finished', round: 5 }),
    joinWrapper: { game: g },
    nextChallenge: { page: '/challenge/[token]', query: { token: 'CM0GZgAucV9Aw4EA' }, props: { pageProps: {
      gameSnapshot: g, challenge: { roundCount: 5, forbidRotating: false }, map: { id: 'm', bounds: g.bounds } } } },
    nextDehydrated: { pageProps: { dehydratedState: { queries: [{ queryKey: ['other'], state: { data: { rounds: [1] } } },
      { queryKey: ['classic-game', g.token], state: { data: g } }] } } },
  };
  for (const [name, payload] of Object.entries(payloads)) {
    const log = [];
    const m = L.extractMeta(traced(payload, log));
    assert.ok(m && m.game && m.game.token === g.token, name);
    assert.ok(noSecret(m), name);
    const bad = log.filter(p => /\.(rounds|panoId|streakLocationCode|startTime|heading|pitch)\??$|\.\*$/.test(p) ||
      (/\.(lat|lng)\??$/.test(p) && !/\.bounds\.(min|max)\.(lat|lng)$/.test(p)));
    assert.deepStrictEqual(bad, [], `${name} read ${bad.join(', ')}`);
  }
});

test('extractMeta reads Next.js page data and challenge responses through the whitelist', () => {
  const next = { page: '/challenge/[token]', query: { token: 'CM0GZgAucV9Aw4EA' }, props: { pageProps: {
    gameSnapshot: fakeGame(),
    map: { id: '52d15ff6850ffb3b847cb3b4', slug: 'world', name: 'World', maxErrorDistance: 14916862,
      bounds: { min: { lat: -54.8, lng: -159.7 }, max: { lat: 78.6, lng: 178.4 } }, customCoordinates: [{ lat: 1, lng: 2 }] },
    challenge: { token: 'CM0GZgAucV9Aw4EA', mapSlug: 'world', roundCount: 5, timeLimit: 180, forbidRotating: false } } } };
  const m = L.extractMeta(next);
  assert.strictEqual(m.game.token, 'x78UXZE1PAnz4oLR');
  assert.strictEqual(m.map.maxErrorDistance, 14916862);
  assert.ok(!('customCoordinates' in m.map));
  assert.strictEqual(m.challenge.roundCount, 5);
  assert.ok(noSecret(m));
  const c = L.challengeMeta({ challenge: { mapSlug: 'world', roundCount: 5, timeLimit: 180, forbidMoving: false },
    map: { id: 'id1', name: 'World', bounds: null }, creator: { nick: 'GeoGuessr' } });
  assert.deepStrictEqual(c.challenge, { mapSlug: 'world', roundCount: 5, timeLimit: 180, forbidMoving: false });
  const merged = L.mergeMeta(L.mergeMeta({}, c), m);
  const sm = L.serverMap(merged);
  assert.strictEqual(sm.id, '52d15ff6850ffb3b847cb3b4');
  assert.strictEqual(sm.name, 'World');
  assert.strictEqual(sm.maxErrorDistance, 14916862);
  assert.strictEqual(L.extractMeta({ token: 'abc' }), null);
  const fromGame = L.serverMap({ game: L.safeGameMeta(fakeGame()) });
  assert.strictEqual(fromGame.slug, 'world');
  assert.strictEqual(fromGame.name, 'World');
  assert.strictEqual(fromGame.maxErrorDistance, null);
});

test('metaKey maps the page\'s own API and Next.js data URLs to their page', () => {
  assert.strictEqual(L.metaKey('/api/v3/games/abc123?client=web', ORIGIN), '/game/abc123');
  assert.strictEqual(L.metaKey(`${ORIGIN}/api/v3/challenges/CM0G/game`, ORIGIN), '/challenge/CM0G');
  assert.strictEqual(L.metaKey(`${ORIGIN}/api/v3/challenges/CM0G`, ORIGIN), '/challenge/CM0G');
  assert.strictEqual(L.metaKey('/_next/data/build1/game/abc123.json?token=abc123', ORIGIN), '/game/abc123');
  assert.strictEqual(L.metaKey('/api/maps/world', ORIGIN), 'map:world');
  for (const u of ['/api/v3/challenges/daily-challenges/today', '/api/v4/clues/abc', '/api/v3/games/abc/replace',
    'https://game-server.geoguessr.com/api/duels/abc', 'https://evil.example/api/v3/games/abc', '/api/v3/social/events/unfinishedgames'])
    assert.strictEqual(L.metaKey(u, ORIGIN), null, u);
});

test('capture and --submit only on single-player games; Play-Along and unknown types refused', () => {
  const game = extra => ({ game: L.safeGameMeta(fakeGame(Object.assign({ token: 'abc123' }, extra))) });
  assert.ok(L.captureVerdict('/game/abc123', game({ type: 'standard' })).ok);
  assert.ok(L.captureVerdict('/game/abc123', game({ type: 'challenge' })).ok);
  assert.ok(L.captureVerdict('/game/abc123', game({ type: 'playalong' })).refuse);
  assert.ok(L.captureVerdict('/challenge/CM0G', game({ type: 'playalong' })).refuse);
  assert.ok(L.captureVerdict('/game/abc123', {}).wait);
  assert.ok(L.captureVerdict('/game/abc123', game({ type: undefined })).wait);
  assert.ok(L.captureVerdict('/game/zzz999', game({ type: 'standard' })).wait);   // data of another game
  assert.ok(L.captureVerdict('/challenge/CM0G', {}).ok);
  assert.ok(L.captureVerdict('/duels/abc', game({ type: 'standard' })).refuse);

  assert.strictEqual(L.submitProblem('/game/abc123', game({ type: 'standard' })), null);
  assert.strictEqual(L.submitProblem('/challenge/CM0G', game({ type: 'challenge' })), null);
  assert.ok(L.submitProblem('/game/abc123', game({ type: 'playalong' })));
  assert.ok(L.submitProblem('/game/abc123', game({ type: 'standard', mode: undefined })));
  assert.ok(L.submitProblem('/game/abc123', game({ type: 'standard', mode: 'streak' })));
  assert.ok(L.submitProblem('/game/abc123', game({ type: 'standard', state: 'finished' })));
  assert.ok(L.submitProblem('/challenge/CM0G', {}));
  assert.ok(L.submitProblem('/battle-royale/abc', game({ type: 'standard' })));
});

test('gameFlags: unknown forbidRotating is probed, unknown forbidZooming keeps the player\'s zoom', () => {
  const g = (r, z) => ({ game: { token: 't', forbidRotating: r, forbidZooming: z } });
  assert.deepStrictEqual(L.gameFlags(g(true, false), false), { noRotate: true, zoomLocked: false });
  assert.deepStrictEqual(L.gameFlags(g(false, true), false), { noRotate: false, zoomLocked: true });
  assert.deepStrictEqual(L.gameFlags({}, false), { noRotate: null, zoomLocked: true });
  assert.deepStrictEqual(L.gameFlags({}, true), { noRotate: true, zoomLocked: true });
  assert.deepStrictEqual(L.gameFlags({ challenge: { forbidRotating: false, forbidZooming: false } }, false),
    { noRotate: false, zoomLocked: false });
});

test('clipProblem: the screenshot must be the whole, unstretched canvas', () => {
  const canvas = { width: 1112, height: 740 };
  const box = { x: 0, y: 60, width: 1112, height: 740 };
  assert.strictEqual(L.clipProblem({ canvasBox: box, clip: { x: 0, y: 60, width: 1112, height: 740 } }, canvas), null);
  assert.strictEqual(L.clipProblem({ canvasBox: Object.assign({}, box, { x: 0.4 }), clip: { x: 1, y: 60, width: 1111, height: 740 } }, canvas), null);
  assert.ok(L.clipProblem({ canvasBox: box, clip: { x: 0, y: 60, width: 1112, height: 680 } }, canvas));
  assert.ok(L.clipProblem({ canvasBox: box, clip: { x: 30, y: 60, width: 1082, height: 740 } }, canvas));
  assert.ok(L.clipProblem({ canvasBox: box, clip: box }, { width: 1112, height: 800 }));
});

test('replayRound: only finished standard games of the local history, no API call', () => {
  const rows = [
    { game: 'x78UXZE1PAnz4oLR', kind: 'standard', map: 'World', round: 1, pano_id: 'P1', lat: 48, lng: 14, gg_heading: 184.5 },
    { game: 'duel1', kind: 'duel', map: 'The World', round: 1, pano_id: 'P2', lat: 1, lng: 2, gg_heading: 0 },
  ];
  const r = L.replayRound(rows, 'x78UXZE1PAnz4oLR', 1);
  assert.deepStrictEqual(r.view, { pano: 'P1', heading: 184.5, pitch: 0, zoom: 0 });
  assert.deepStrictEqual(r.map, { name: 'World' });
  assert.throws(() => L.replayRound(rows, 'x78UXZE1PAnz4oLR', 2), /no round 2/);
  assert.throws(() => L.replayRound(rows, 'duel1', 1), /not a finished game/);
  assert.throws(() => L.replayRound(rows, 'unknownGame1', 1), /not a finished game/);
  const hist = path.join(__dirname, 'data', 'calibration', 'history_rounds.json');
  if (fs.existsSync(hist)) {
    const real = L.replayRound(JSON.parse(fs.readFileSync(hist, 'utf8')), 'x78UXZE1PAnz4oLR', 2);
    assert.ok(real.view.pano && typeof real.truth.lat === 'number');
  }
});

// A minimal browser for the page hooks: location, fetch, google.maps, WebGL / 2D contexts, canvases,
// window messages (with transferred ports), localStorage, __NEXT_DATA__ and a few game DOM nodes.
function fakePage(pathname, { body = null, nextData = null, script = L.PAGE_SCRIPT, storage = {}, canvas = false, dom = {} } = {}) {
  class Pano {
    constructor(div) { this.div = div; this.pov = { heading: 0, pitch: 0 }; this.ls = {}; this.pano = null; }
    addListener(ev, fn) { (this.ls[ev] = this.ls[ev] || []).push(fn); }
    setPano(p) { this.pano = p; (this.ls.pano_changed || []).forEach(f => f()); }
    setPov(v) { this.pov = v; }
    getPov() { return this.pov; }
    getZoom() { return 0; }
    setZoom() {}
    getVisible() { return true; }
  }
  class GL { uniformMatrix4fv() { return 'drawn'; } drawElements() { return 'drawn'; } }
  class C2D { drawImage() { return 'drawn'; } }
  class Canvas { getContext(type, attrs) { this.ctx = { type, attrs }; return this.ctx; } }
  const calls = [], listeners = {}, posted = [];
  const sceneCanvas = { getBoundingClientRect: () => ({ width: 800, height: 500, left: 0, top: 0 }), width: 800, height: 500, style: {} };
  const find = sel => {
    if (canvas && /widget-scene-canvas/.test(sel)) return sceneCanvas;
    for (const [k, v] of Object.entries(dom)) if (sel.includes(k) && v) return v;
    return null;
  };
  const win = {
    location: { pathname, origin: ORIGIN, href: ORIGIN + pathname, search: '' },
    performance: { now: () => Date.now() }, setTimeout, clearTimeout, setInterval: () => 0, URL, devicePixelRatio: 1, Promise,
    google: { maps: { StreetViewPanorama: Pano } }, WebGLRenderingContext: GL, CanvasRenderingContext2D: C2D,
    HTMLCanvasElement: Canvas,
    localStorage: { getItem: k => (k in storage ? storage[k] : null), setItem: (k, v) => { storage[k] = String(v); },
      removeItem: k => { delete storage[k]; } },
    addEventListener: (t, f) => { (listeners[t] = listeners[t] || []).push(f); },
    postMessage: (data, origin, transfer) => {
      posted.push(data);
      setImmediate(() => {
        let stop = false;
        const ev = { data, origin, source: win.self, ports: transfer || [], stopImmediatePropagation: () => { stop = true; } };
        for (const f of (listeners.message || []).slice()) { if (stop) break; f(ev); }
      });
    },
    document: {
      readyState: 'complete', addEventListener() {},
      querySelectorAll: sel => { const e = find(sel); return e ? [e] : []; },
      querySelector: find,
      getElementById: id => (id === '__NEXT_DATA__' && nextData ? { textContent: JSON.stringify(nextData) } : null),
    },
    fetch(url) {
      calls.push(url);
      const resp = { ok: true, headers: { get: () => 'application/json; charset=utf-8' }, clone: () => ({ json: async () => body }) };
      return Promise.resolve(resp);
    },
  };
  win.window = win;
  vm.createContext(win);
  win.self = vm.runInContext('this', win);   // the context's global as page code sees `window`
  vm.runInContext(script, win);
  // the extension's content script: its private port, as content.js hands it over
  const connect = () => {
    const ch = new MessageChannel();
    const got = [];
    ch.port1.onmessage = e => got.push(e.data);
    win.postMessage({ __geoscr: 'port' }, ORIGIN, [ch.port2]);
    return { port: ch.port1, got, close: () => { ch.port1.close(); ch.port2.close(); } };
  };
  return { win, calls, posted, storage, Canvas, listeners, connect, Pano };
}

const flush = () => new Promise(r => setImmediate(r));
const MATRIX = [-91.94588, -3.43787, -39.25201, -39.25201, 39.24281, 1.28808, -91.97427, -91.97427,
  -2.44064, 150.2254, -0.10966, -0.10966, 0, 0, -0.66667, 0];

test('page hooks keep only whitelisted metadata of the page\'s own responses', async () => {
  const { win, calls } = fakePage('/game/x78UXZE1PAnz4oLR', { body: fakeGame() });
  const resp = await win.fetch('/api/v3/games/x78UXZE1PAnz4oLR?client=web');
  assert.strictEqual(resp.ok, true);
  assert.strictEqual(calls.length, 1);
  await flush();
  const S = win.__geoscr;
  const m = S.metaFor('/game/x78UXZE1PAnz4oLR');
  assert.strictEqual(m.game.type, 'challenge');
  assert.strictEqual(m.game.forbidRotating, true);
  assert.ok(noSecret(S.store), JSON.stringify(S.store));
  assert.strictEqual(S.metaFor('/game/other1'), null);   // only the current page
  const pano = new win.google.maps.StreetViewPanorama({ contains: () => false });
  assert.strictEqual(S.panos.length, 1);
  pano.setPano('abc');
  assert.ok(S.panoChanges >= 1);
  const gl = new win.WebGLRenderingContext();
  gl.canvas = {};
  assert.strictEqual(gl.uniformMatrix4fv(null, false, Float32Array.from(MATRIX)), 'drawn');
  assert.ok(Math.abs(gl.canvas.__geoscrFov.vfov - 67.28) < 0.05);
  assert.strictEqual(gl.drawElements(), 'drawn');
  assert.strictEqual(gl.canvas.__geoscrDraws, 1);
  assert.strictEqual(S.loadPano({ pano: 'x', heading: 0, pitch: 0, zoom: 0 }, '/game/x78UXZE1PAnz4oLR'), false);  // replay only
  assert.strictEqual(S.post, undefined);   // no submit outside play_live_visual.js --submit
  // an in-app navigation away from the game stops every camera action
  win.location.pathname = '/duels/abc123';
  assert.strictEqual(S.setView({ heading: 50, pitch: 0 }, '/game/x78UXZE1PAnz4oLR'), false);
  assert.strictEqual(S.metaFor('/game/x78UXZE1PAnz4oLR'), null);
  assert.strictEqual(await S.rotateLocked('/game/x78UXZE1PAnz4oLR'), null);
  assert.deepStrictEqual(pano.getPov(), { heading: 0, pitch: 0 });
});

test('page hooks stay inert outside the allowed pages', async () => {
  const { win, calls } = fakePage('/duels/abc123', { body: fakeGame({ token: 'abc123' }) });
  const resp = await win.fetch('/api/v3/games/abc123?client=web');
  assert.strictEqual(resp.ok, true);
  assert.strictEqual(calls.length, 1);
  await flush();
  const S = win.__geoscr;
  assert.deepStrictEqual(Object.keys(S.store), []);
  assert.strictEqual(S.metaFor('/duels/abc123'), null);
  const pano = new win.google.maps.StreetViewPanorama({});
  pano.setPano('abc');
  pano.setPov({ heading: 10, pitch: 0 });
  assert.strictEqual(S.panos.length, 0);
  assert.strictEqual(S.panoChanges, 0);
  assert.strictEqual(S.setView({ heading: 50, pitch: 0 }, '/duels/abc123'), false);
  assert.strictEqual(S.isolate(true, '/duels/abc123'), null);
  const gl = new win.WebGLRenderingContext();
  gl.canvas = {};
  assert.strictEqual(gl.uniformMatrix4fv(null, false, Float32Array.from(MATRIX)), 'drawn');
  gl.drawElements();
  assert.strictEqual(gl.canvas.__geoscrFov, undefined);
  assert.strictEqual(gl.canvas.__geoscrDraws, undefined);
  const c2 = new win.CanvasRenderingContext2D();
  c2.canvas = {};
  c2.drawImage();
  assert.strictEqual(c2.canvas.__geoscrDraws, undefined);
  const cv = new win.HTMLCanvasElement();
  cv.getContext('webgl2', { alpha: false });
  assert.deepStrictEqual(cv.ctx.attrs, { alpha: false });   // untouched outside the game pages
});

test('WebGL canvases on game pages keep a readable drawing buffer', () => {
  const { win } = fakePage('/challenge/CM0GZgAucV9Aw4EA');
  for (const type of ['webgl', 'webgl2', 'experimental-webgl']) {
    const cv = new win.HTMLCanvasElement();
    cv.getContext(type, { alpha: false });
    assert.deepStrictEqual(Object.assign({}, cv.ctx.attrs), { alpha: false, preserveDrawingBuffer: true }, type);
    assert.strictEqual(cv.__geoscrGl, cv.ctx);
  }
  const c2 = new win.HTMLCanvasElement();
  c2.getContext('2d', { willReadFrequently: true });
  assert.deepStrictEqual(c2.ctx.attrs, { willReadFrequently: true });
  const off = fakePage('/challenge/CM0GZgAucV9Aw4EA', { script: L.pageScript({ preserve: false }) }).win;
  const cv = new off.HTMLCanvasElement();
  cv.getContext('webgl', undefined);
  assert.strictEqual(cv.ctx.attrs, undefined);
});

test('replay pages of finished games only with the replay switch; storage carries no paths', () => {
  const p = '/game/x78UXZE1PAnz4oLR/replay';
  assert.strictEqual(fakePage(p).win.__geoscr.allowed(), false);
  const on = fakePage(p, { script: L.pageScript({ replay: true }) }).win.__geoscr;
  assert.strictEqual(on.allowed(), true);
  // the extension: switches mirrored into localStorage, read at document_start
  const src = fs.readFileSync(path.join(EXT, 'page.js'), 'utf8');
  const stored = fakePage(p, { script: src, storage: { __geoscr_cfg: JSON.stringify({ replay: true }) } }).win.__geoscr;
  assert.strictEqual(stored.allowed(), true);
  const sneaky = fakePage('/duels/abc', { script: src, storage: { __geoscr_cfg: JSON.stringify({ replay: true, paths: ['.*'],
    submit: true }) } }).win.__geoscr;
  assert.strictEqual(sneaky.allowed(), false);
  assert.strictEqual(sneaky.post, undefined);
  assert.strictEqual(fakePage(p, { script: src }).win.__geoscr, undefined);   // no global outside the test mode
  // the content script can switch it on and off for the current document
  stored.setReplay(false);
  assert.strictEqual(stored.allowed(), false);
});

test('loadPano works on replay pages only', () => {
  const { win } = fakePage('/game/x78UXZE1PAnz4oLR/replay', { script: L.pageScript({ replay: true }) });
  const S = win.__geoscr;
  const pano = new win.google.maps.StreetViewPanorama({ contains: () => false });
  assert.strictEqual(S.loadPano({ pano: 'P1', heading: 10, pitch: 0, zoom: 0 }, '/game/x78UXZE1PAnz4oLR/replay'), true);
  assert.strictEqual(pano.pano, 'P1');
});

test('page hooks read __NEXT_DATA__ under its own page token only', () => {
  const nextData = { page: '/challenge/[token]', query: { token: 'CM0GZgAucV9Aw4EA' }, props: { pageProps: {
    gameSnapshot: fakeGame(), challenge: { roundCount: 5, forbidRotating: false, forbidZooming: false } } } };
  const here = fakePage('/challenge/CM0GZgAucV9Aw4EA', { nextData }).win.__geoscr;
  const m = here.metaFor('/challenge/CM0GZgAucV9Aw4EA');
  assert.strictEqual(m.game.token, 'x78UXZE1PAnz4oLR');
  assert.strictEqual(m.challenge.forbidZooming, false);
  assert.ok(noSecret(here.store));
  const stale = fakePage('/challenge/OtherToken1', { nextData }).win.__geoscr;
  assert.strictEqual(stale.metaFor('/challenge/OtherToken1'), null);
});

const TEST_MODE = { __geoscr_cfg: JSON.stringify({ replay: true }) };   // the replay test mode exposes window.__geoscr

test('controller: private port handshake; nothing shown or requested outside single-player pages', async () => {
  const src = fs.readFileSync(path.join(EXT, 'page.js'), 'utf8');
  const pg = fakePage('/duels/abc123', { script: src, canvas: true, storage: Object.assign({}, TEST_MODE) });
  const { win, posted } = pg;
  assert.deepStrictEqual(posted, []);   // page.js posts nothing on the window
  const S = win.__geoscr;
  assert.strictEqual(S.ctl.attached, false);
  // a page script listening after page.js never sees the port message
  let seen = 0;
  win.addEventListener('message', () => { seen++; });
  const cs = pg.connect();
  await flush(); await flush(); await flush();
  assert.strictEqual(seen, 0);
  assert.deepStrictEqual(cs.got, [{ type: 'hello' }]);
  cs.port.postMessage({ type: 'settings', payload: { auto: true, replay: false } });
  await flush(); await flush(); await flush();
  assert.strictEqual(S.ctl.attached, true);
  assert.strictEqual(S.ctl.settings.auto, true);
  // a second "port" (another script) is swallowed and ignored
  const other = pg.connect();
  await flush(); await flush(); await flush();
  assert.deepStrictEqual(other.got, []);
  assert.strictEqual(seen, 0);
  await S.ctl.tick();
  assert.strictEqual(S.ctl.view.mode, 'off');
  const r = await S.ctl.analyse('manual');
  assert.strictEqual(r, null);
  await flush(); await flush();
  assert.deepStrictEqual(cs.got, [{ type: 'hello' }]);   // no HUD, health or predict request
  // Play-Along on /game/<t>: off once the game type is known
  const pa = fakePage('/game/abc123', { script: src, body: fakeGame({ token: 'abc123', type: 'playalong' }), storage: Object.assign({}, TEST_MODE) });
  await pa.win.fetch('/api/v3/games/abc123');
  await flush();
  pa.win.__geoscr.attach(() => Promise.reject(new Error('no')), { auto: true });
  await pa.win.__geoscr.ctl.tick();
  assert.strictEqual(pa.win.__geoscr.ctl.view.mode, 'off');
  cs.close(); other.close();
});

test('extension without the test mode: no globals, no storage, no window messages', async () => {
  const hud = fs.readFileSync(path.join(EXT, 'hud', 'hud.js'), 'utf8');
  const src = fs.readFileSync(path.join(EXT, 'page.js'), 'utf8');
  const pg = fakePage('/game/abc123', { script: hud + '\n' + src, canvas: true });
  assert.strictEqual(pg.win.__geoscr, undefined);
  assert.strictEqual(pg.win.GeoscrHud, undefined);   // taken by page.js and removed
  assert.deepStrictEqual(pg.storage, {});
  assert.deepStrictEqual(pg.posted, []);
});

test('controller on a single-player game: checks the server, asks for hud.css over the port', async () => {
  const src = fs.readFileSync(path.join(EXT, 'page.js'), 'utf8');
  const roundEl = { innerText: 'Round 1 / 5' };
  const pg = fakePage('/game/abc123', { script: src, canvas: true, body: fakeGame({ token: 'abc123', type: 'standard' }),
    storage: Object.assign({}, TEST_MODE), dom: { "[data-qa='round-number']": roundEl } });
  await pg.win.fetch('/api/v3/games/abc123');
  const cs = pg.connect();
  await flush(); await flush(); await flush();
  cs.port.postMessage({ type: 'settings', payload: { auto: false, replay: true } });
  await flush(); await flush(); await flush();
  pg.win.GeoscrHud = { api: 2, mount: () => ({ update() {} }) };   // stands in for the bundled HUD
  cs.port.onmessage = e => {
    cs.got.push(e.data);
    if (e.data.id) cs.port.postMessage({ type: 'reply', id: e.data.id, ok: e.data.type !== 'health', payload: '.gsh{}', error: 'Сервер локатора недоступний' });
  };
  await pg.win.__geoscr.ctl.tick();
  await flush(); await flush(); await flush(); await flush();
  assert.ok(/недоступний/.test(pg.win.__geoscr.ctl.view.status.text), pg.win.__geoscr.ctl.view.status.text);
  assert.strictEqual(pg.win.__geoscr.ctl.css, '.gsh{}');
  const types = cs.got.map(m => m.type);
  assert.ok(types.includes('health'), types.join());
  assert.ok(types.includes('hud-css'), types.join());
  assert.ok(!types.includes('predict'));
  assert.strictEqual(pg.win.__geoscr.ctl.view.mode, 'game');
  cs.close();
});

test('auto-analyse waits for the panorama after the previous round\'s result screen', async () => {
  const roundEl = { innerText: 'Round 1 / 5' };
  const dom = { "[data-qa='round-number']": roundEl, "[data-qa='standard-round-result']": null };
  const pg = fakePage('/game/abc123', { body: fakeGame({ token: 'abc123', type: 'standard' }), canvas: true, dom });
  await pg.win.fetch('/api/v3/games/abc123');
  await flush();
  const S = pg.win.__geoscr;
  S.attach(() => Promise.reject(new Error('no')), { auto: false });
  const pano = new pg.win.google.maps.StreetViewPanorama({ contains: () => true });
  pano.setPano('round1');
  await S.ctl.tick();
  assert.strictEqual(S.ctl.lastKey, '/game/abc123#1');
  assert.strictEqual(S.ctl.since, null);   // the pano change of the last 3 s is this round's
  pano.setPano('moved1');   // the player moves
  pano.setPano('moved2');
  dom["[data-qa='standard-round-result']"] = {};
  await S.ctl.tick();
  const mark = S.panoChanges;
  assert.strictEqual(S.ctl.resultMark, mark);
  // the round number changes before GeoGuessr loads the new panorama
  dom["[data-qa='standard-round-result']"] = null;
  roundEl.innerText = 'Round 2 / 5';
  await S.ctl.tick();
  assert.strictEqual(S.ctl.lastKey, '/game/abc123#2');
  assert.strictEqual(JSON.stringify(S.ctl.since), JSON.stringify({ panoChanges: mark }));   // needs a pano change after the result screen
});

test('a capture stops when the panorama changes during the sweep (a move or the next round)', async () => {
  const pg = fakePage('/game/abc123', { body: fakeGame({ token: 'abc123', type: 'standard' }), canvas: true });
  const { win } = pg;
  let skew = 0;
  win.performance.now = () => Date.now() + skew;
  win.requestAnimationFrame = cb => setTimeout(cb, 1);
  win.document.createElement = () => ({ width: 0, height: 0, toDataURL: () => 'data:image/jpeg;base64,AAAA',
    getContext: () => ({ drawImage() {}, getImageData: () => ({ data: new Uint8ClampedArray(512).fill(200) }) }) });
  const scene = win.document.querySelector('canvas.widget-scene-canvas');
  scene.getContext = () => null;
  const draw = () => { const c = new win.CanvasRenderingContext2D(); c.canvas = scene; c.drawImage(); };
  const P = win.google.maps.StreetViewPanorama;
  P.prototype.getStatus = () => 'OK';
  const S = win.__geoscr;
  const pano = new P({ contains: () => true });
  let povs = 0;
  const setPov = pano.setPov;
  pano.setPov = function (v) { setPov.call(this, v); draw(); if (++povs === 3) this.setPano('next-round'); };
  pano.setPano('round1');
  draw();
  skew = 2000;   // the panorama has been still for longer than SETTLE_MS
  const t0 = Date.now();
  await assert.rejects(S.capture('/game/abc123', { flags: { noRotate: false, zoomLocked: false } }), /панорама змінилася/);
  assert.ok(povs >= 3 && Date.now() - t0 < 8000, `${povs} views, ${Date.now() - t0} ms`);
  assert.strictEqual(pano.getPov().heading !== undefined, true);
});

test('pageScript is valid JavaScript for watch and replay paths', () => {
  assert.doesNotThrow(() => new Function(L.PAGE_SCRIPT));
  assert.doesNotThrow(() => new Function(L.pageScript({ replay: true })));
  assert.doesNotThrow(() => new Function(L.hudScript()));
});

test('parseRound', () => {
  assert.deepStrictEqual(L.parseRound('Round 2 / 5'), { round: 2, of: 5 });
  assert.strictEqual(L.parseRound('Streak 3'), null);
});

test('FOV from a projection matrix captured from Google Street View (1112x740, zoom 1)', () => {
  const f = L.fovFromMatrix(Float32Array.from(MATRIX));
  assert.ok(Math.abs(f.hfov - 90) < 0.05, f.hfov);
  assert.ok(Math.abs(L.hfovFromVfov(f.vfov, 1112 / 740) - 90) < 0.05);
  assert.strictEqual(L.fovFromMatrix(new Float32Array(16)), null);
  assert.ok(Math.abs(L.hfovForZoom(1) - 90) < 1e-9);
  assert.ok(Math.abs(L.zoomForHfov(L.hfovForZoom(0.37)) - 0.37) < 1e-9);
  // measured on the 1112x740 replay canvas: zoom 0 and 0.25 both give vfov 90, hfov 112.72
  const z0 = L.formulaFov(0, 1112 / 740);
  assert.ok(Math.abs(z0.vfov - 90) < 1e-9 && Math.abs(z0.hfov - 112.72) < 0.01, JSON.stringify(z0));
  assert.ok(Math.abs(L.formulaFov(0.5, 1112 / 740).hfov - 109.47) < 0.01);
});

function covered(grid, hfov, vfov, az, el) {
  const tx = Math.tan(hfov * Math.PI / 360), ty = Math.tan(vfov * Math.PI / 360);
  return grid.filter(g => L.inFrustum(az - g.yaw, el, g.pitch, tx, ty)).length;
}

test('planGrid covers about -85..85 with overlapping neighbours, fewer views when wide', () => {
  let prev = Infinity;
  for (const [hfov, aspect] of [[90, 1.5], [110, 1.75], [118, 1.5], [126.9, 1.5]]) {
    const vfov = L.vfovFromHfov(hfov, aspect);
    const grid = L.planGrid(hfov, vfov, 37);
    assert.ok(grid.length <= prev, `${hfov}: ${grid.length} views`);
    prev = grid.length;
    for (const g of grid) assert.ok(Math.abs(g.pitch) + vfov / 2 <= 85.01);
    // the view centres reach +-85; between views the polar caps are covered up to about +-80
    for (let el = -80; el <= 80; el += 2.5)
      for (let az = 0; az < 360; az += 2.5) assert.ok(covered(grid, hfov, vfov, az, el) >= 1, `${hfov}: hole at ${az},${el}`);
    // rows overlap by >= 10 % of the vertical FOV at the row centre
    const pitches = [...new Set(grid.map(g => g.pitch))].sort((a, b) => a - b);
    for (let i = 1; i < pitches.length; i++) assert.ok(pitches[i] - pitches[i - 1] <= 0.9 * vfov + 1e-6);
  }
});

test('predictBody sends only views and map settings', () => {
  const b = L.predictBody([{ image_b64: 'data:image/jpeg;base64,AAA', yaw: '10', pitch: -40, hfov: 112.7, vfov: 90, zoom: 0,
    read: 'preserved', file: '/etc/passwd' }], { id: 'm', name: 'World', bounds: null, debug_dir: '/tmp/x', extra: 1 });
  assert.deepStrictEqual(b, { views: [{ image_b64: 'data:image/jpeg;base64,AAA', yaw: 10, pitch: -40, hfov: 112.7, vfov: 90 }],
    map: { id: 'm', name: 'World' } });
  assert.ok(!('map' in L.predictBody([], null)));
});

test('pageCapture decodes the page\'s JPEG data URLs', () => {
  const cap = L.pageCapture({ views: [{ image_b64: 'data:image/jpeg;base64,' + Buffer.from('jpeg!').toString('base64'), yaw: 1,
    pitch: 2, hfov: 100, vfov: 80, width: 640, height: 400 }], capture: { mode: 'sweep', fov: { hfov: 100, vfov: 80 } } });
  assert.strictEqual(cap.views[0].buf.toString(), 'jpeg!');
  assert.strictEqual(cap.source, 'page');
  assert.strictEqual(cap.mode, 'sweep');
});

// ------------------------------------------------------------------ HUD helpers

test('HUD placement: free columns avoid GeoGuessr\'s controls, status bar and guess map', () => {
  assert.deepStrictEqual(H.freeIntervals(0, 100, [[10, 20], [15, 30], [80, 120]]), [[0, 10], [30, 80]]);
  assert.deepStrictEqual(H.freeIntervals(0, 100, []), [[0, 100]]);
  const area = { x: 0, y: 0, width: 1366, height: 768 };
  const status = { x: 1046, y: 16, width: 320, height: 64 };
  const controls = { x: 24, y: 468, width: 48, height: 268 };
  const logo = { x: 16, y: 16, width: 120, height: 24 };
  const map = { x: 1366 - 32 - 410, y: 768 - 16 - 400, width: 410, height: 400 };
  const w = H.panelWidth(1366, false);
  const p = H.autoPlace(area, [status, controls, logo, map], w);
  assert.strictEqual(p.side, 'left');
  assert.ok(p.fits);
  const overlaps = (r, q) => r.x < q.x + q.width && r.x + r.width > q.x && r.y < q.y + q.height && r.y + r.height > q.y;
  const box = { x: p.x, y: p.y, width: w, height: p.h };
  for (const o of [status, controls, logo, map]) assert.ok(!overlaps(box, o), JSON.stringify([box, o]));
  assert.ok(p.h > 380, `${p.h}`);
  // left column blocked from top to bottom, the right one short (status bar to map): right of the wall
  const wall = { x: 0, y: 0, width: 400, height: 768 };
  const q = H.autoPlace(area, [wall, status, map], w);
  assert.strictEqual(q.side, 'mid');
  for (const o of [wall, status, map]) assert.ok(!overlaps({ x: q.x, y: q.y, width: w, height: q.h }, o));
  assert.ok(q.h > 600);
  const tall = H.autoPlace(area, [{ x: 0, y: 0, width: 400, height: 768 }, status, { x: 924, y: 552, width: 410, height: 200 }], w);
  assert.strictEqual(tall.side, 'right');   // a right column of >= 300 px is kept
  // nothing fits: top-left corner, flagged
  assert.strictEqual(H.autoPlace({ x: 0, y: 0, width: 500, height: 150 }, [{ x: 0, y: 40, width: 500, height: 60 }], 300).fits, false);
  assert.ok(H.isCompact(1100, 700) && !H.isCompact(1366, 768) && !H.isCompact(1920, 1080));
  assert.ok(H.panelWidth(1100, true) <= 300 && H.panelWidth(1920, false) <= 380);
});

test('HUD placement keeps clear of the guess map at its largest size (65vw) and of short columns', () => {
  const overlaps = (r, q) => r.x < q.x + q.width && r.x + r.width > q.x && r.y < q.y + q.height && r.y + r.height > q.y;
  for (const [vw, vh] of [[1920, 1080], [1366, 768], [1100, 700], [1366, 657], [1100, 589]]) {
    const area = { x: 0, y: 0, width: vw, height: vh };
    const status = { x: vw - 320, y: 16, width: 320, height: 64 };
    const controls = { x: 24, y: vh - 300, width: 48, height: 268 };
    const logo = { x: 16, y: 16, width: 120, height: 24 };
    const inactive = { x: vw - 32 - 0.16 * vw, y: vh - 16 - 0.16 * vw / 1.25 - 56, width: 0.16 * vw, height: 0.16 * vw / 1.25 + 56 };
    for (const aw of [30, 45, 65]) {
      const map = H.mapReserve(inactive, vw, { width: aw * vw / 100, aspect: 1.25 });
      const expanded = { x: vw - 32 - aw * vw / 100, y: Math.max(136, vh - 16 - aw * vw / 125 - 56), width: aw * vw / 100,
        height: Math.min(vh - 152, aw * vw / 125 + 56) };
      assert.ok(map.x <= expanded.x && map.y <= expanded.y + 1, `${vw}x${vh} ${aw}vw ${JSON.stringify([map, expanded])}`);
      const w = H.panelWidth(vw, H.isCompact(vw, vh));
      const p = H.autoPlace(area, [status, controls, logo, map], w);
      const box = { x: p.x, y: p.y, width: w, height: p.h };
      for (const o of [status, controls, logo, map, expanded]) assert.ok(!overlaps(box, o), `${vw}x${vh} ${aw}vw ${JSON.stringify([box, o])}`);
      // 1100x589 with a 65vw map leaves only the left column (225 px; the panel scrolls)
      assert.ok(p.fits && p.h >= (vw * aw > 60000 && vh < 600 ? 200 : 250), `${vw}x${vh} ${aw}vw: ${p.h}`);
    }
  }
  // a short left column: the panel moves right of the controls instead of shrinking
  const area = { x: 0, y: 0, width: 1100, height: 589 };
  const controls = { x: 24, y: 300, width: 48, height: 268 };
  const logo = { x: 16, y: 16, width: 120, height: 24 };
  const map = { x: 1100 - 32 - 715, y: 0, width: 715 + 16, height: 589 };
  const p = H.autoPlace(area, [controls, logo, map], 253);
  assert.strictEqual(p.side, 'mid');
  assert.ok(p.x >= 84 && p.h > 400, JSON.stringify(p));
  assert.strictEqual(H.cssLength(' 65vw', 1366, 768), 1366 * 0.65);
  assert.strictEqual(H.cssLength('13rem', 1366, 768), 208);
  assert.strictEqual(H.cssLength('calc(65vw / 1.25)', 1366, 768), null);
});

test('HUD shows images only as data: URLs or from Plonk It, never from GeoGuessr', () => {
  assert.strictEqual(H.safeImage('https://www.geoguessr.com/images/resize:fit:600:600/plain/clueimage/ab.jpg'), null);
  assert.strictEqual(H.safeImage('http://localhost:8080/hud/img?u=x'), null);
  assert.strictEqual(H.safeImage('data:image/webp;base64,AAAA'), 'data:image/webp;base64,AAAA');
  assert.strictEqual(H.safeImage('data:text/html;base64,AAAA'), null);
  assert.strictEqual(H.safeImage('https://www.plonkit.net/images/austria/a.png'), 'https://www.plonkit.net/images/austria/a.png');
  assert.strictEqual(H.safeImage('https://www.plonkit.net/a.png" onerror="x'), null);
  for (const f of [path.join(__dirname, 'web', 'hud', 'hud.js'), path.join(EXT, 'hud', 'hud.js')])
    assert.ok(!/geoguessr\.com/.test(fs.readFileSync(f, 'utf8')), f);
});

test('the extension bundles the same HUD files as web/hud', () => {
  for (const f of ['hud.js', 'hud.css'])
    assert.strictEqual(fs.readFileSync(path.join(EXT, 'hud', f), 'utf8'), fs.readFileSync(path.join(__dirname, 'web', 'hud', f), 'utf8'),
      `extension/hud/${f} is stale: cp web/hud/${f} extension/hud/`);
});

test('HUD reads the optional detected direction in any of its likely shapes', () => {
  assert.strictEqual(H.normalizeDetected(undefined), null);
  assert.strictEqual(H.normalizeDetected(null), null);
  assert.strictEqual(H.normalizeDetected([]), null);
  const one = (d, k) => H.normalizeDetected(d)[0][k];
  assert.strictEqual(one(370, 'heading'), 10);
  // engine.clue_detect (engine/hints.py): score is the detector's, probability the calibrated one
  const d = H.normalizeDetected({ heading: 230.4, pitch: -6, score: 3.71, probability: 0.62, true_north: true });
  assert.deepStrictEqual(d, [{ heading: 230.4, pitch: -6, zoom: null, score: 0.62, text: null, absolute: true }]);
  const rel = H.normalizeDetected({ heading: 300, pitch: 0, score: 0.5, probability: 0.4, true_north: false })[0];
  assert.strictEqual(rel.absolute, false);
  assert.strictEqual(rel.heading, -60);
  assert.strictEqual(one({ yaw: -90, pitch: -5, score: 0.7, label: 'знак' }, 'heading'), 270);
  assert.strictEqual(one({ yaw: -90, pitch: -5, score: 0.7, label: 'знак' }, 'score'), 0.7);
  assert.strictEqual(one({ yaw: 10, score: 4.2 }, 'score'), null);
  assert.strictEqual(one({ azimuth: 45, confidence: 80 }, 'score'), 0.8);
  assert.strictEqual(H.normalizeDetected([{ azimuth: 45 }, { foo: 1 }, 'текст']).length, 2);
  assert.strictEqual(H.compassName(0), 'Пн');
  assert.strictEqual(H.compassName(135), 'ПдСх');
  assert.strictEqual(H.compassName(-90), 'Зх');
  assert.strictEqual(H.esc('<img onerror=x>'), '&lt;img onerror=x&gt;');
});

// ------------------------------------------------------------------ extension

test('extension manifest: page world at document_start on GeoGuessr only, minimal permissions', () => {
  const m = JSON.parse(fs.readFileSync(path.join(EXT, 'manifest.json'), 'utf8'));
  assert.strictEqual(m.manifest_version, 3);
  const [page, cs] = m.content_scripts;
  assert.deepStrictEqual(page.js, ['hud/hud.js', 'page.js']);
  assert.strictEqual(page.world, 'MAIN');
  assert.strictEqual(page.run_at, 'document_start');
  assert.deepStrictEqual(cs.js, ['content.js']);
  assert.strictEqual(cs.run_at, 'document_start');
  for (const c of m.content_scripts) assert.deepStrictEqual(c.matches, ['https://www.geoguessr.com/*']);
  assert.deepStrictEqual(m.permissions, ['storage']);
  assert.ok(!m.host_permissions.some(h => /<all_urls>|\*:\/\/\*\//.test(h)));
  assert.deepStrictEqual(m.optional_host_permissions, ['http://*/*']);
  assert.ok(!m.web_accessible_resources);
  for (const f of ['page.js', 'content.js', 'background.js', 'options.html', 'options.js', 'hud/hud.js', 'hud/hud.css', m.background.service_worker,
    ...Object.values(m.icons)]) assert.ok(fs.existsSync(path.join(EXT, f)), f);
});

test('extension never talks to the GeoGuessr API and never submits', () => {
  for (const f of ['content.js', 'background.js', 'options.js', 'hud/hud.js']) {
    const s = fs.readFileSync(path.join(EXT, f), 'utf8');
    assert.ok(!/geoguessr\.com\/api|\/api\/v[34]\//.test(s), f);
  }
  // the page world never reads the panorama's id or position; the only POST is the --submit hook
  const page = fs.readFileSync(path.join(EXT, 'page.js'), 'utf8');
  const noSubmit = page.replace(/ {4}if \(cfg\.submit\) \{[\s\S]*?\n {4}\}\n/, '');
  assert.ok(noSubmit.length < page.length);
  for (const re of [/getPano\(/, /getPosition\(/, /getLocation\(/, /\.rounds\b/, /panoId/, /'POST'/])
    assert.ok(!re.test(noSubmit), String(re));
});

function loadBackground(server = 'http://localhost:8094') {
  const listeners = {};
  const store = { server };
  const fetched = [];
  const result = { countries: [{ code: 'EE', probability: 0.5 }], hints: [{ country_code: 'EE', geoguessr: [
    { id: 'a', image_url: 'https://www.geoguessr.com/images/resize:fit:600:600/plain/clueimage/aa.jpg' },
    { id: 'b', image_url: 'https://www.geoguessr.com/images/resize:fit:600:600/plain/clueimage/bb.jpg' },
    { id: 'c', image_url: 'https://evil.example/x.jpg' }],
    plonkit: [{ text: 't', image_url: 'https://www.plonkit.net/images/estonia/p.png' }] }] };
  const resp = (status, type, body) => ({ ok: status === 200, status, headers: { get: () => type },
    json: async () => JSON.parse(JSON.stringify(body)), text: async () => String(body), arrayBuffer: async () => Uint8Array.from([1, 2, 3]).buffer });
  const chrome = {
    runtime: { id: 'ext1', onMessage: { addListener: f => { listeners.msg = f; } }, openOptionsPage() {},
      getURL: p => `chrome-extension://ext1/${p}` },
    action: { onClicked: { addListener() {} } },
    storage: { local: { get: async () => Object.assign({}, store), set: async s => Object.assign(store, s) } },
  };
  const ctx = { chrome, URL, AbortController, setTimeout, clearTimeout, console, btoa, Uint8Array, String,
    fetch: async (url, init) => {
      fetched.push({ url, init });
      if (url.endsWith('/api/predict')) return resp(200, 'application/json', result);
      if (url.includes('/hud/img?u=') && url.includes('aa.jpg')) return resp(200, 'image/webp', null);
      if (url.includes('/hud/img?u=')) return resp(404, 'application/json', { error: 'not cached' });
      if (url.startsWith('chrome-extension://ext1/hud/hud.css')) return resp(200, 'text/css', '.gsh{}');
      return resp(200, 'application/json', { ok: true });
    } };
  vm.createContext(ctx);
  vm.runInContext(fs.readFileSync(path.join(EXT, 'background.js'), 'utf8'), ctx);
  const send = (msg, url = 'https://www.geoguessr.com/game/abc') => new Promise(r => listeners.msg(msg, { id: 'ext1', url }, r));
  return { send, fetched, store };
}

test('extension service worker: views and map only, local server only, clue images inlined', async () => {
  const bgw = loadBackground();
  const img = 'data:image/jpeg;base64,AAAA';
  let r = await bgw.send({ type: 'predict', payload: { views: [{ image_b64: img, yaw: 1, pitch: 2, hfov: 100, vfov: 80, x: 1 }],
    map: { name: 'World', debug_dir: '/tmp' }, debug_dir: '/tmp' } });
  assert.ok(r.ok, r.error);
  const f = bgw.fetched.find(x => x.url.endsWith('/api/predict'));
  assert.strictEqual(f.url, 'http://localhost:8094/api/predict');
  assert.deepStrictEqual(JSON.parse(f.init.body), { views: [{ image_b64: img, yaw: 1, pitch: 2, hfov: 100, vfov: 80 }], map: { name: 'World' } });
  // the page gets no GeoGuessr URL: cached images as data: URLs, the rest dropped
  const cards = r.payload.hints[0].geoguessr;
  assert.strictEqual(cards[0].image_url, 'data:image/webp;base64,AQID');
  assert.strictEqual(cards[1].image_url, null);
  assert.strictEqual(cards[2].image_url, null);
  assert.ok(!/geoguessr\.com|evil\.example/.test(JSON.stringify(r.payload)));
  assert.ok(bgw.fetched.every(x => /^(http:\/\/localhost:8094\/|chrome-extension:)/.test(x.url)), bgw.fetched.map(x => x.url).join());
  r = await bgw.send({ type: 'predict', payload: { views: [{ image_b64: 'javascript:alert(1)' }] } });
  assert.ok(!r.ok);
  r = await bgw.send({ type: 'asset', name: 'hud.js' });
  assert.ok(!r.ok);
  r = await bgw.send({ type: 'hud-css' });
  assert.ok(r.ok && r.payload === '.gsh{}', JSON.stringify(r));
  r = await bgw.send({ type: 'health' }, 'https://evil.example/');
  assert.ok(!r.ok);
  r = await bgw.send({ type: 'set', payload: { auto: true, server: 'http://evil.example' } });
  assert.ok(r.ok);
  assert.strictEqual(bgw.store.auto, true);
  assert.strictEqual(bgw.store.server, 'http://localhost:8094');
  // a public server address is refused
  const pub = loadBackground('http://evil.example:8080');
  r = await pub.send({ type: 'health' });
  assert.ok(!r.ok && /не локальна/.test(r.error), r.error);
  assert.deepStrictEqual(pub.fetched, []);
  for (const ok of ['http://127.0.0.1:8080', 'http://172.20.1.5:8080', 'http://192.168.1.2:8080']) {
    r = await loadBackground(ok).send({ type: 'health' });
    assert.ok(r.ok, ok);
  }
});

// content.js with a fake window and chrome: answers on single-player pages only; storage only in test mode.
function loadContent(pathname, settings) {
  const storage = {};
  const sent = [];
  let pagePort = null;
  const win = {
    location: { pathname, origin: ORIGIN },
    postMessage: (data, origin, transfer) => { if (data && data.__geoscr === 'port') pagePort = transfer[0]; },
    localStorage: { getItem: k => (k in storage ? storage[k] : null), setItem: (k, v) => { storage[k] = String(v); }, removeItem: k => { delete storage[k]; } },
  };
  const chrome = {
    runtime: { id: 'ext1', lastError: null, sendMessage: (m, cb) => { sent.push(m); cb({ ok: true, payload: { ok: true } }); } },
    storage: { local: { get: (d, cb) => cb(Object.assign({}, d, settings)) }, onChanged: { addListener() {} } },
  };
  const ctx = Object.assign({ chrome, MessageChannel, JSON, Object, Promise, setTimeout }, win, { window: win });
  vm.createContext(ctx);
  vm.runInContext(fs.readFileSync(path.join(EXT, 'content.js'), 'utf8'), ctx);
  const got = [];
  pagePort.onmessage = e => got.push(e.data);
  return { port: pagePort, got, sent, storage };
}

test('extension content script: requests answered on single-player pages only; localStorage only in test mode', async () => {
  const duel = loadContent('/duels/abc', { replay: false });
  assert.strictEqual(JSON.stringify(duel.storage), '{}');
  duel.port.postMessage({ type: 'health', id: 'h1' });
  await flush(); await flush(); await flush();
  assert.strictEqual(duel.sent.length, 0);
  assert.strictEqual(duel.got[0].ok, false);
  duel.port.close();
  const game = loadContent('/game/abc123', { replay: false });
  game.port.postMessage({ type: 'hello' });
  game.port.postMessage({ type: 'health', id: 'h2' });
  await flush(); await flush(); await flush();
  assert.strictEqual(JSON.stringify(game.got[0]), JSON.stringify({ type: 'settings', payload: { auto: false, replay: false } }));
  assert.strictEqual(game.got[1].ok, true);
  assert.strictEqual(JSON.stringify(game.sent), JSON.stringify([{ type: 'health' }]));
  game.port.close();
  const test = loadContent('/game/abc123/replay', { replay: true });
  assert.strictEqual(JSON.stringify(test.storage), JSON.stringify({ __geoscr_cfg: '{"replay":true}' }));
  test.port.close();
});
