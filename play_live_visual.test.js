// Tests for the pure helpers and page hooks of play_live_visual.js:  node --test play_live_visual.test.js
'use strict';
const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');
const vm = require('node:vm');
const L = require('./play_live_visual.js');

const SECRET = { lat: 47.1234567, lng: 13.7654321, pano: '4a667848a4e4e4e724c5a506b3555367778616630494' };
const ORIGIN = 'https://www.geoguessr.com';

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

// A minimal browser for the page hooks: location, fetch, google.maps, WebGL / 2D contexts, __NEXT_DATA__.
function fakePage(pathname, { body = null, nextData = null } = {}) {
  class Pano {
    constructor(div) { this.div = div; this.pov = { heading: 0, pitch: 0 }; this.ls = {}; }
    addListener(ev, fn) { (this.ls[ev] = this.ls[ev] || []).push(fn); }
    setPano() { (this.ls.pano_changed || []).forEach(f => f()); }
    setPov(v) { this.pov = v; }
    getPov() { return this.pov; }
    getZoom() { return 0; }
    setZoom() {}
    getVisible() { return true; }
  }
  class GL { uniformMatrix4fv() { return 'drawn'; } }
  class C2D { drawImage() { return 'drawn'; } }
  const calls = [];
  const win = {
    location: { pathname, origin: ORIGIN, href: ORIGIN + pathname },
    performance: { now: () => Date.now() }, setTimeout, URL, devicePixelRatio: 1,
    google: { maps: { StreetViewPanorama: Pano } }, WebGLRenderingContext: GL, CanvasRenderingContext2D: C2D,
    document: {
      readyState: 'complete', addEventListener() {}, querySelectorAll: () => [], querySelector: () => null,
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
  vm.runInContext(L.PAGE_SCRIPT, win);
  return { win, calls };
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
  assert.strictEqual(gl.canvas.__geoscrFov, undefined);
  const c2 = new win.CanvasRenderingContext2D();
  c2.canvas = {};
  c2.drawImage();
  assert.strictEqual(c2.canvas.__geoscrDraws, undefined);
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

test('pageScript is valid JavaScript for watch and replay paths', () => {
  assert.doesNotThrow(() => new Function(L.PAGE_SCRIPT));
  assert.doesNotThrow(() => new Function(L.pageScript(L.ALLOWED_PATHS.concat([L.REPLAY_PATH]))));
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
