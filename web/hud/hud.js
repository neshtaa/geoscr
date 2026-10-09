/*
 * On-screen HUD of the live helper, shared by play_live_visual.js (evaluateOnNewDocument) and the
 * Chrome extension (bundled as extension/hud/hud.js, a copy of this file, run before extension/page.js,
 * which takes the global and removes it). It renders the view state of the round controller
 * (extension/page.js) in a closed shadow root, so GeoGuessr's CSS and scripts do not reach it; the
 * CSS text (web/hud/hud.css) comes in mount({css}) or window.__geoscrHudCss.
 *
 *   const hud = GeoscrHud.mount({css, onAction(name, arg)});  hud.update(view);  hud.destroy();
 *   actions: analyse, auto, look {heading, pitch, zoom}
 *
 * The HUD makes no network request: card images are shown only as data: URLs (the clue images
 * come from the local server's cache) or from Plonk It; a GeoGuessr image URL is never loaded.
 *
 * Placement: inside the panorama, in the tallest free column on the left or right that does not
 * cover the guess map (at its expanded size), the status bar, the compass, the zoom controls or the
 * timer, or next to the left-hand controls when those columns are short; draggable by the header
 * (position kept in localStorage, double click = automatic placement again), collapsible, compact on
 * small windows. Keys: Alt+G analyse, Alt+H hide/show, Alt+1..3 country details, Alt+A
 * auto-analyse, Alt+C collapse.
 */
(function (root, factory) {
  const api = factory(root);
  if (typeof module === 'object' && module && module.exports) module.exports = api;
  else root.GeoscrHud = api;
})(typeof window !== 'undefined' ? window : globalThis, function (root) {
  'use strict';

  const API = 2;
  const STORE = 'geoscr-hud';
  const GAP = 12;
  const COMPASS = ['Пн', 'ПнСх', 'Сх', 'ПдСх', 'Пд', 'ПдЗх', 'Зх', 'ПнЗх'];
  // GeoGuessr's own in-game UI the HUD must not cover
  const OBSTACLES = [
    "[class*='game_status']", "[class*='game_controls']", "[class*='game_inGameLogos']", "[class*='game_topHud'] > *",
    "[data-qa='guess-map']", "[class*='guess-map_guessMap']", "[data-qa='compass']", "[class*='panorama-compass']",
    "[data-qa='pano-zoom-in']", "[data-qa='pano-zoom-out']", "[data-qa='return-to-start']", "[data-qa='undo-move']",
    "[data-qa='round-number']", "[class*='clock-timer']", '[data-game-hud]', "[class*='replay']", 'header',
  ].join(',');
  const MAP_SIZES = { 1: 16, 2: 30, 3: 45, 4: 65 };   // GeoGuessr's guess-map size classes: --active-width in vw

  // ---------------------------------------------------------------- pure helpers

  const esc = t => String(t === null || t === undefined ? '' : t)
    .replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const num = (...xs) => { for (const x of xs) if (typeof x === 'number' && isFinite(x)) return x; return null; };
  const str = (...xs) => { for (const x of xs) if (typeof x === 'string' && x.trim()) return x.trim(); return null; };
  const norm360 = h => { const r = h % 360; return r < 0 ? r + 360 : r; };
  const pct = p => (p >= 0.095 ? Math.round(p * 100) : Math.round(p * 1000) / 10) + '%';
  const sec = ms => (ms >= 10000 ? Math.round(ms / 1000) : Math.round(ms / 100) / 10) + ' с';
  const signed = x => (x < 0 ? '−' : '') + Math.abs(Math.round(x));
  const deg = x => (typeof x === 'number' && isFinite(x) ? x.toFixed(3) : esc(x));

  function compassName(h) {
    return COMPASS[Math.round(norm360(h) / 45) % 8];
  }

  function countryName(code, fallback) {
    try {
      const n = new Intl.DisplayNames(['uk'], { type: 'region' }).of(code);
      if (n && n !== code) return n;
    } catch (e) { /* no Intl.DisplayNames */ }
    return fallback || code;
  }

  // The only image sources the HUD shows: data: URLs and Plonk It (third party, no referrer).
  function safeImage(u) {
    if (typeof u !== 'string') return null;
    if (/^data:image\/(jpeg|png|webp|gif);base64,[A-Za-z0-9+/=]+$/.test(u)) return u;
    return /^https:\/\/(www\.)?plonkit\.net\/[^\s"'<>]+$/.test(u) ? u : null;
  }

  // hints[i].geoguessr[j].detected (optional; engine.clue_detect: {heading, pitch, score, probability,
  // true_north}): a heading, a text, an object or a list of them. A heading relative to the image centre
  // (true_north false) cannot be pointed at.
  function normalizeDetected(d) {
    if (d === null || d === undefined || d === false) return null;
    const prob = x => {
      let p = num(x.probability, x.prob, x.p, x.confidence);
      if (p !== null && p > 1 && p <= 100) p /= 100;
      if (p === null) { const sc = num(x.score, x.strength); p = sc !== null && sc >= 0 && sc <= 1 ? sc : null; }
      return p !== null && p >= 0 && p <= 1 ? p : null;
    };
    const out = [];
    for (const x of Array.isArray(d) ? d : [d]) {
      if (typeof x === 'number' && isFinite(x)) out.push({ heading: norm360(x), pitch: null, zoom: null, score: null, text: null, absolute: true });
      else if (typeof x === 'string' && x.trim()) out.push({ heading: null, pitch: null, zoom: null, score: null, text: x.trim(), absolute: true });
      else if (x && typeof x === 'object') {
        const h = num(x.heading, x.yaw, x.azimuth, x.az, x.direction);
        const t = str(x.text, x.label, x.what, x.title, x.reason, x.note);
        if (h === null && !t) continue;
        const absolute = x.true_north !== false;
        out.push({ heading: h === null ? null : absolute ? norm360(h) : norm360(h + 180) - 180,
          pitch: num(x.pitch, x.elevation, x.el), zoom: num(x.zoom), score: prob(x), text: t, absolute });
      }
    }
    return out.length ? out.slice(0, 3) : null;
  }

  // A CSS length of GeoGuessr's custom properties (65vw, 13rem, 208px) in px, else null.
  function cssLength(v, vw, vh, rem = 16) {
    const m = /^\s*(-?\d*\.?\d+)(vw|vh|px|rem)\s*$/.exec(v || '');
    if (!m) return null;
    const n = +m[1];
    return m[2] === 'vw' ? n * vw / 100 : m[2] === 'vh' ? n * vh / 100 : m[2] === 'rem' ? n * rem : n;
  }

  // The area the guess map takes when expanded (hover, or pinned): its active size anchored at its
  // bottom right corner, plus the guess button below the map and the size controls above it.
  // active: {width (px), aspect} from GeoGuessr's CSS (--active-width, --aspect-ratio), else 30vw, 1.25.
  function mapReserve(r, vw, active) {
    const aw = active && active.width > 0 ? active.width : 0.3 * vw;
    const ar = active && active.aspect > 0 ? active.aspect : 1.25;
    const right = r.x + r.width, bottom = r.y + r.height;
    const w = Math.max(r.width, aw + 16), h = Math.min(bottom, Math.max(r.height, aw / ar + 72));
    return { x: right - w, y: bottom - h, width: w, height: h };
  }

  // Free vertical intervals of [lo, hi] outside the blocks [[top, bottom], ...].
  function freeIntervals(lo, hi, blocks) {
    const bs = blocks.filter(b => b[1] > lo && b[0] < hi).sort((a, b) => a[0] - b[0]);
    const out = [];
    let y = lo;
    for (const [t, b] of bs) {
      if (t > y) out.push([y, t]);
      y = Math.max(y, b);
    }
    if (hi > y) out.push([y, hi]);
    return out;
  }

  // Where a panel of width w fits inside `area` without touching `rects`: the tallest free interval of
  // the left or the right column (left and higher preferred); when both are shorter than 300 px, also
  // the columns right of the left-hand controls (over the panorama). {x, y, h, side, fits}
  function autoPlace(area, rects, w, gap = GAP) {
    const cands = [];
    const column = (x, side) => {
      const blocks = rects.filter(r => r.x < x + w + gap && r.x + r.width > x - gap)
        .map(r => [r.y - gap, r.y + r.height + gap]);
      for (const [a, b] of freeIntervals(area.y + gap, area.y + area.height - gap, blocks)) cands.push({ x, y: a, h: b - a, side });
    };
    const score = c => Math.min(c.h, 900) + (c.side === 'left' ? 60 : c.side === 'mid' ? -40 : 0) - 0.1 * (c.y - area.y);
    const pick = () => cands.slice().sort((a, b) => score(b) - score(a))[0];
    column(area.x + gap, 'left');
    column(area.x + area.width - w - gap, 'right');
    let best = pick();
    if (!best || best.h < 300) {
      const xs = rects.filter(r => r.x < area.x + 0.3 * area.width).map(r => Math.round(r.x + r.width + gap))
        .filter(x => x > area.x + gap && x + w + gap <= area.x + area.width);
      for (const x of [...new Set(xs)]) column(x, 'mid');
      best = pick();
    }
    if (!best || best.h < 120) return { x: area.x + gap, y: area.y + gap, h: Math.max(120, area.height - 2 * gap), side: 'left', fits: false };
    return Object.assign({}, best, { fits: true });
  }

  function panelWidth(vw, compact) {
    return Math.round(compact ? Math.max(232, Math.min(300, vw * 0.23)) : Math.max(290, Math.min(380, vw * 0.2)));
  }

  const isCompact = (vw, vh) => vw < 1280 || vh < 740;

  // ---------------------------------------------------------------- HUD

  function mount(opts = {}) {
    const doc = root.document;
    const old = doc.getElementById('geoscr-hud-host');
    if (old) old.remove();
    const host = doc.createElement('div');
    host.id = 'geoscr-hud-host';
    host.style.cssText = 'position:fixed;z-index:2147483000;left:16px;top:16px;display:none;margin:0;padding:0';
    const shadow = host.attachShadow({ mode: 'closed' });
    const style = doc.createElement('style');
    style.textContent = typeof opts.css === 'string' ? opts.css : root.__geoscrHudCss || '';
    shadow.appendChild(style);
    const panel = doc.createElement('div');
    panel.className = 'gsh';
    shadow.appendChild(panel);
    const pill = doc.createElement('button');
    pill.className = 'gsh-pill';
    pill.title = 'Показати локатор (Alt+H)';
    pill.textContent = 'Л';
    shadow.appendChild(pill);
    // in the page only while it shows something
    const attach = on => {
      if (on && !host.isConnected) (doc.body || doc.documentElement).appendChild(host);
      else if (!on && host.isConnected) host.remove();
    };

    let view = { mode: 'off', status: { text: '', level: 'idle' } };
    let prefs = load();
    let destroyed = false, place = null, timer = null, html = null;
    const open = {};   // cards whose description is unfolded (compact mode)

    function load() {
      let p = null;
      try { p = JSON.parse(root.localStorage.getItem(STORE) || 'null'); } catch (e) { p = null; }
      p = p && typeof p === 'object' ? p : {};
      return { pos: p.pos && isFinite(p.pos.x) && isFinite(p.pos.y) ? { x: +p.pos.x, y: +p.pos.y } : null,
        collapsed: !!p.collapsed, hidden: !!p.hidden, sel: Number.isInteger(p.sel) && p.sel >= -1 && p.sel < 5 ? p.sel : 0 };
    }
    function save() {
      try { root.localStorage.setItem(STORE, JSON.stringify(prefs)); } catch (e) { /* storage blocked */ }
    }
    const act = (name, arg) => { try { if (opts.onAction) opts.onAction(name, arg); } catch (e) { /* controller gone */ } };

    // ---- layout
    function area() {
      const vw = root.innerWidth, vh = root.innerHeight;
      const cont = doc.querySelector('#panorama-container') || doc.querySelector("[data-qa='panorama']") ||
        doc.querySelector('canvas.widget-scene-canvas');
      let a = { x: 0, y: 0, width: vw, height: vh };
      if (cont) {
        const r = cont.getBoundingClientRect();
        const x = Math.max(0, r.left), y = Math.max(0, r.top);
        const b = { x, y, width: Math.min(vw, r.right) - x, height: Math.min(vh, r.bottom) - y };
        if (b.width > 360 && b.height > 300) a = b;
      }
      return a;
    }
    // The guess map's expanded size from GeoGuessr's CSS (custom properties inherit from the size class).
    function mapActive(el, vw, vh) {
      const cs = root.getComputedStyle(el);
      const rem = parseFloat(root.getComputedStyle(doc.documentElement).fontSize) || 16;
      let w = cssLength(cs.getPropertyValue('--active-width'), vw, vh, rem);
      if (w === null) {
        let size = 2;
        for (let e = el, i = 0; e && i < 4; e = e.parentElement, i++) {
          const m = /(?:^|[\s_-])size([1-4])(?:__|\b)/.exec(e.getAttribute('class') || '');
          if (m) { size = +m[1]; break; }
        }
        w = MAP_SIZES[size] * vw / 100;
      }
      const minW = cssLength(cs.getPropertyValue('--min-width'), vw, vh, rem) || 0;
      return { width: Math.max(w, minW), aspect: parseFloat(cs.getPropertyValue('--aspect-ratio')) || 1.25 };
    }
    function obstacles(vw, vh) {
      const out = [];
      let controls = false;
      for (const el of doc.querySelectorAll(OBSTACLES)) {
        if (host.contains(el)) continue;
        const r = el.getBoundingClientRect();
        if (r.width < 2 || r.height < 2) continue;
        const id = (el.getAttribute('data-qa') || '') + ' ' + (el.getAttribute('class') || '');
        const map = /guess-map/.test(id);
        if (!map && r.width > 0.6 * vw && r.height > 0.6 * vh) continue;   // a container
        const cs = root.getComputedStyle(el);
        if (cs.visibility === 'hidden' || cs.display === 'none' || (+cs.opacity === 0 && !map)) continue;
        const b = { x: r.left, y: r.top, width: r.width, height: r.height };
        out.push(map ? mapReserve(b, vw, mapActive(el, vw, vh)) : b);
        if (/zoom|controls|return-to-start|undo/.test(id)) controls = true;
      }
      if (!controls && doc.querySelector("[data-qa='guess-map']")) {
        // game page without recognised controls: keep the usual bottom-left corner free
        out.push({ x: 0, y: vh - 300, width: 110, height: 300 });
      }
      return out;
    }
    function layout() {
      if (destroyed) return;
      attach(view.mode !== 'off');
      if (view.mode === 'off') return;
      const vw = root.innerWidth, vh = root.innerHeight;
      const compact = isCompact(vw, vh);
      const w = panelWidth(vw, compact);
      host.style.width = w + 'px';
      let x, y, maxH;
      if (prefs.pos) {
        x = Math.max(4, Math.min(vw - w - 4, prefs.pos.x));
        y = Math.max(4, Math.min(vh - 60, prefs.pos.y));
        maxH = vh - y - GAP;
      } else {
        place = autoPlace(area(), obstacles(vw, vh), w);
        x = place.x; y = place.y; maxH = place.h;
      }
      host.style.left = Math.round(x) + 'px';
      host.style.top = Math.round(y) + 'px';
      panel.style.maxHeight = Math.max(110, Math.round(maxH)) + 'px';
      pill.style.display = prefs.hidden ? '' : 'none';
      panel.style.display = prefs.hidden ? 'none' : '';
    }

    // ---- rendering
    function statusHtml() {
      const s = view.status || {};
      const lvl = s.level || 'idle';
      const prog = s.progress && s.progress[1] ? ` <span class="prog">${+s.progress[0]}/${+s.progress[1]}</span>` : '';
      const el = lvl === 'busy' && s.since ? ` <span class="elapsed" data-since="${+s.since}"></span>` : '';
      const round = view.round ? `<span class="round">${esc(view.round)}</span>` : '';
      return `<div class="status ${esc(lvl)}"><span class="dot"></span><span class="stext">${esc(s.text || '')}${prog}${el}</span>${round}</div>`;
    }

    function cardHtml(k, i, j, compact) {
      const det = normalizeDetected(k.detected);
      const img = safeImage(k.image_url);
      const key = `${i}:${j}`;
      let h = `<div class="card gg${det ? ' found' : ''}">`;
      if (img) h += `<img class="thumb" data-act="zoomimg" loading="lazy" referrerpolicy="no-referrer" src="${esc(img)}" alt="">`;
      h += `<div class="ctext"><div class="ctitle"${compact && k.text ? ` data-act="card" data-k="${key}" title="Опис"` : ''}>${esc(k.title || k.id || '')}` +
        `${compact && k.text ? `<span class="more">${open[key] ? '▴' : '▾'}</span>` : ''}</div>`;
      if (k.text && (!compact || open[key])) h += `<div class="cdesc">${esc(k.text)}</div>`;
      (det || []).forEach((d, n) => {
        const parts = [];
        if (d.heading !== null) parts.push(d.absolute ? `напрям ${Math.round(d.heading)}° (${compassName(d.heading)})`
          : `${signed(d.heading)}° від центру знімка`);
        if (d.pitch !== null) parts.push(`нахил ${signed(d.pitch)}°`);
        if (d.score !== null) parts.push(pct(d.score));
        h += `<div class="det"><b>Знайдено:</b> ${esc(d.text ? d.text + (parts.length ? ' · ' : '') : '')}${esc(parts.join(' · '))}` +
          (d.heading !== null && d.absolute && view.canLook ? ` <button class="mini" data-act="look" data-i="${i}" data-j="${j}" data-n="${n}" title="Повернути камеру">Показати</button>` : '') + '</div>';
      });
      if (k.look || k.view) {
        const v = k.view || {};
        const extra = [typeof v.pitch === 'number' ? `нахил ${signed(v.pitch)}°` : null,
          typeof v.zoom === 'number' ? `зум ${Math.round(v.zoom * 10) / 10}` : null].filter(Boolean).join(', ');
        const where = k.look ? esc(k.look) + (extra ? ` <span class="muted">(${esc(extra)})</span>` : '') : esc(extra);
        if (where) h += `<div class="look"><b>Куди дивитися:</b> ${where}</div>`;
      }
      const why = (k.matched || []).filter(Boolean);
      if (why.length && !compact) h += `<div class="why">${esc(why.join(' · '))}</div>`;
      return h + '</div></div>';
    }

    function countryDetails(c, hint, i, compact) {
      let h = `<div class="sec details"><div class="dhead">${esc(countryName(c.code, c.name))}` +
        `${hint && hint.driving_side ? ` <span class="muted">· рух ${hint.driving_side === 'left' ? 'лівосторонній' : 'правосторонній'}</span>` : ''}</div>`;
      if (!hint) return h + '<div class="muted small">Підказок для цієї країни немає.</div></div>';
      if (hint.regions && hint.regions.length)
        h += `<div class="regions">${hint.regions.slice(0, compact ? 3 : 4)
          .map(r => `<span class="chip reg">${esc(r.name)} <b>${pct(r.probability)}</b></span>`).join('')}</div>`;
      // cards with a detected placement first
      const cards = (hint.geoguessr || []).map((k, j) => ({ k, j }));
      cards.sort((a, b) => (normalizeDetected(b.k.detected) ? 1 : 0) - (normalizeDetected(a.k.detected) ? 1 : 0) || a.j - b.j);
      cards.slice(0, compact ? 2 : 3).forEach(({ k, j }) => { h += cardHtml(k, i, j, compact); });
      for (const t of (hint.plonkit || []).slice(0, compact ? 0 : 1)) {
        const img = safeImage(t.image_url);
        h += `<div class="card pk">${img ? `<img class="thumb" data-act="zoomimg" loading="lazy" referrerpolicy="no-referrer" src="${esc(img)}" alt="">` : ''}` +
          `<div class="ctext"><div class="ctitle">Plonk It${t.section ? ` · <span class="muted">${esc(t.section)}</span>` : ''}</div>` +
          `<div class="cdesc clamp">${esc(t.text)}</div></div></div>`;
      }
      return h + '</div>';
    }

    function guessHtml(r, countries, compact) {
      const g = r.guess || {};
      let h = `<div class="sec guess"><div><span class="lbl">Здогадка</span> <span class="coords" data-act="copy" data-copy="${esc(g.lat)}, ${esc(g.lng)}" title="Скопіювати координати">${deg(g.lat)}, ${deg(g.lng)}</span>` +
        ` · <b class="score">~${esc(g.expected_score)}</b>${compact ? '' : ' балів'}</div>`;
      const tp = r.top_country_point;
      if (tp && countries[0] && !compact)
        h += `<div class="muted small">У межах ${esc(countryName(countries[0].code, countries[0].name))}: ${deg(tp.lat)}, ${deg(tp.lng)} · ~${esc(tp.expected_score_if_country_right)}, якщо країна вірна</div>`;
      return h + '</div>';
    }

    function bodyHtml(compact) {
      const r = view.result;
      let h = '';
      if (view.warning) h += `<div class="warn">${esc(view.warning)}</div>`;
      if (view.score) h += `<div class="sec scorebox">Раунд: +${esc(view.score.points)} балів, ${esc(view.score.distance)} км · всього ${esc(view.score.total)}</div>`;
      if (!r) {
        if (view.mode === 'game' && view.auto && (view.status || {}).level !== 'busy')
          h += '<div class="empty">Автоаналіз увімкнено: панорама аналізується на початку кожного раунду.</div>';
        return h + footerHtml(compact);
      }
      const hints = {};
      (r.hints || []).forEach((x, i) => { hints[x.country_code] = { hint: x, i }; });
      const countries = (r.countries || []).slice(0, compact ? 3 : 5);
      const top = Math.max(0.01, countries[0] ? countries[0].probability : 1);
      h += '<div class="sec countries">';
      countries.forEach((c, i) => {
        const hit = hints[c.code];
        h += `<div class="crow${prefs.sel === i ? ' open' : ''}" data-act="country" data-i="${i}" title="${hit ? 'Підказки' : 'Без підказок'} (Alt+${i + 1})">` +
          `<span class="rank">${i + 1}</span><span class="cname">${esc(countryName(c.code, c.name))}</span>` +
          `<span class="code">${esc(c.code)}</span><span class="bar"><i style="width:${Math.max(2, Math.min(100, c.probability * 100 / top))}%"></i></span>` +
          `<span class="p">${pct(c.probability)}</span><span class="chev">${hit ? '▾' : ''}</span></div>`;
      });
      h += '</div>';
      const sel = countries[prefs.sel];
      const hit = sel ? hints[sel.code] : null;
      const details = sel ? countryDetails(sel, hit && hit.hint, hit ? hit.i : -1, compact) : '';
      // compact: the chosen country's regions and cards right below the countries
      h += compact ? details + guessHtml(r, countries, compact) : guessHtml(r, countries, compact) + details;
      if (r.observations && r.observations.length)
        h += `<div class="sec"><div class="lbl">Що видно</div>${r.observations.slice(0, compact ? 5 : 10)
          .map(o => `<span class="chip obs" style="opacity:${0.55 + 0.45 * Math.max(0, Math.min(1, +o.strength || 0))}">${esc(o.text)}</span>`).join('')}</div>`;
      const selRegions = !!(hit && hit.hint.regions && hit.hint.regions.length);
      if (r.regions && r.regions.length && !(compact && selRegions))
        h += `<div class="sec"><div class="lbl">Регіони</div>${r.regions.slice(0, compact ? 3 : 5)
          .map(x => `<span class="chip reg">${esc(x.name)} <span class="muted">${esc(x.country)}</span> <b>${pct(x.probability)}</b></span>`).join('')}</div>`;
      for (const w of r.warnings || []) h += `<div class="warn small">${esc(w)}</div>`;
      const t = view.timing;
      if (t || (r.map && r.map.name)) {
        const parts = [];
        if (r.map && r.map.name) parts.push(`карта ${r.map.name}`);
        if (t && t.capture_ms !== undefined && t.capture_ms !== null) parts.push(`знімання ${sec(t.capture_ms)}`);
        if (t && t.wait_ms) parts.push(`очікування ${sec(t.wait_ms)}`);
        if (t && t.server_ms !== undefined && t.server_ms !== null) parts.push(`сервер ${sec(t.server_ms)}`);
        if (t && t.total_ms) parts.push(`усього ${sec(t.total_ms)}`);
        if (t && t.fov && t.fov.hfov) parts.push(`${t.mode === 'single' ? 'один кадр' : `${t.views || '?'} кадр.`} ${Math.round(t.fov.hfov * 10) / 10}°×${Math.round(t.fov.vfov * 10) / 10}°`);
        h += `<div class="timing">${esc(parts.join(' · '))}</div>`;
      }
      return h + footerHtml(compact);
    }

    function footerHtml(compact) {
      if (compact || view.mode !== 'game') return '';
      return '<div class="keys">Alt+G аналіз · Alt+H сховати · Alt+1..3 країна · Alt+A авто · Alt+C згорнути</div>';
    }

    function summary() {
      const r = view.result;
      if (!r || !r.countries) return '';
      return `<div class="summary">${r.countries.slice(0, 3).map(c => `${esc(countryName(c.code, c.name))} <b>${pct(c.probability)}</b>`).join(' · ')}</div>`;
    }

    const tickElapsed = () => {
      for (const el of panel.querySelectorAll('.elapsed')) el.textContent = sec(Date.now() - +el.getAttribute('data-since'));
    };

    function render() {
      if (destroyed) return;
      if (view.mode === 'off') { layout(); return; }
      host.style.display = '';
      const compact = isCompact(root.innerWidth, root.innerHeight);
      const busy = (view.status || {}).level === 'busy';
      const game = view.mode === 'game';
      const cls = `gsh ${compact ? 'compact' : ''} ${prefs.collapsed ? 'collapsed' : ''} ${view.mode}`;
      const next =
        `<div class="head" data-drag="1" title="Перетягніть; подвійний клік — автоматичне місце">` +
        `<span class="logo"></span><span class="title">Локатор</span><span class="grow"></span>` +
        (game ? `<button class="btn primary" data-act="analyse" ${busy ? 'disabled' : ''} title="Аналізувати зараз (Alt+G)">${busy ? '…' : 'Аналіз'}</button>` +
          `<button class="btn toggle ${view.auto ? 'on' : ''}" data-act="auto" title="Автоаналіз на початку раунду (Alt+A)">Авто</button>` : '') +
        `<button class="btn icon" data-act="collapse" title="${prefs.collapsed ? 'Розгорнути' : 'Згорнути'} (Alt+C)">${prefs.collapsed ? '+' : '−'}</button>` +
        `<button class="btn icon" data-act="hide" title="Сховати (Alt+H)">×</button></div>` +
        statusHtml() + (prefs.collapsed ? summary() : `<div class="body">${bodyHtml(compact)}</div>`);
      if (panel.className !== cls) panel.className = cls;
      // the same content is not rebuilt, so a click is never cut in two by a redraw
      if (next !== html) {
        const body = panel.querySelector('.body');
        const scroll = body ? body.scrollTop : 0;
        panel.innerHTML = next;
        html = next;
        const nb = panel.querySelector('.body');
        if (nb) nb.scrollTop = scroll;
        tickElapsed();
      }
      layout();
    }

    // ---- interaction
    panel.addEventListener('click', e => {
      const t = e.target.closest('[data-act]');
      if (!t || !panel.contains(t)) return;
      const a = t.getAttribute('data-act');
      if (a === 'analyse' || a === 'auto') act(a);
      else if (a === 'collapse') { prefs.collapsed = !prefs.collapsed; save(); render(); }
      else if (a === 'hide') setHidden(true);
      else if (a === 'country') toggleCountry(+t.getAttribute('data-i'));
      else if (a === 'card') { const k = t.getAttribute('data-k'); open[k] = !open[k]; render(); }
      else if (a === 'zoomimg') t.classList.toggle('big');
      else if (a === 'copy') {
        try { root.navigator.clipboard.writeText(t.getAttribute('data-copy') || t.textContent); t.classList.add('copied'); } catch (err) { /* no clipboard */ }
      } else if (a === 'look') {
        const r = view.result || {};
        const k = ((r.hints || [])[+t.getAttribute('data-i')] || {}).geoguessr || [];
        const d = (normalizeDetected((k[+t.getAttribute('data-j')] || {}).detected) || [])[+t.getAttribute('data-n')];
        if (d) act('look', { heading: d.heading, pitch: d.pitch === null ? 0 : d.pitch, zoom: d.zoom });
      }
      e.stopPropagation();
    });
    panel.addEventListener('error', e => {
      if (e.target && e.target.tagName === 'IMG') e.target.classList.add('broken');
    }, true);
    pill.addEventListener('click', () => setHidden(false));
    for (const ev of ['wheel', 'mousedown', 'pointerdown', 'dblclick', 'contextmenu']) {
      // keep the panorama and the page from reacting to the HUD's own input
      host.addEventListener(ev, e => e.stopPropagation());
    }

    function toggleCountry(i) {
      if (!(i >= 0)) return;
      prefs.sel = prefs.sel === i ? -1 : i;
      if (prefs.collapsed) prefs.collapsed = false;
      save();
      render();
    }
    function setHidden(on) {
      prefs.hidden = on;
      save();
      layout();
    }

    // drag by the header
    let drag = null;
    panel.addEventListener('pointerdown', e => {
      const h = e.target.closest('[data-drag]');
      if (!h || e.target.closest('button') || e.button !== 0) return;
      const r = host.getBoundingClientRect();
      drag = { dx: e.clientX - r.left, dy: e.clientY - r.top, id: e.pointerId, moved: false };
      try { h.setPointerCapture(e.pointerId); } catch (err) { /* no capture */ }
      e.preventDefault();
    });
    panel.addEventListener('pointermove', e => {
      if (!drag || e.pointerId !== drag.id) return;
      drag.moved = true;
      prefs.pos = { x: Math.round(e.clientX - drag.dx), y: Math.round(e.clientY - drag.dy) };
      layout();
    });
    const endDrag = e => {
      if (!drag || e.pointerId !== drag.id) return;
      if (drag.moved) save();
      drag = null;
    };
    panel.addEventListener('pointerup', endDrag);
    panel.addEventListener('pointercancel', endDrag);
    panel.addEventListener('dblclick', e => {
      if (!e.target.closest('[data-drag]') || e.target.closest('button')) return;
      prefs.pos = null;
      save();
      layout();
    });

    const onKey = e => {
      if (!e.altKey || e.ctrlKey || e.metaKey || view.mode === 'off') return;
      const tg = e.target;
      if (tg && (tg.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(tg.tagName || ''))) return;
      const c = e.code;
      let done = true;
      if (c === 'KeyG') { if (view.mode === 'game') act('analyse'); }
      else if (c === 'KeyH') setHidden(!prefs.hidden);
      else if (c === 'KeyA') { if (view.mode === 'game') act('auto'); }
      else if (c === 'KeyC') { prefs.collapsed = !prefs.collapsed; save(); render(); }
      else if (/^Digit[1-3]$/.test(c)) { if (prefs.hidden) setHidden(false); toggleCountry(+c.slice(5) - 1); }
      else done = false;
      if (done) { e.preventDefault(); e.stopImmediatePropagation(); }
    };
    root.addEventListener('keydown', onKey, true);
    const onResize = () => render();
    root.addEventListener('resize', onResize);
    timer = setInterval(() => {
      if (destroyed || view.mode === 'off') return;
      tickElapsed();
      if (!drag) layout();
    }, 1000);

    return {
      api: API,
      host,
      shadow,
      update(v) {
        view = Object.assign({}, v || {});
        if (!view.status) view.status = { text: '', level: 'idle' };
        render();
      },
      layout,
      place: () => place,
      prefs: () => Object.assign({}, prefs),
      destroy() {
        destroyed = true;
        clearInterval(timer);
        root.removeEventListener('keydown', onKey, true);
        root.removeEventListener('resize', onResize);
        host.remove();
      },
    };
  }

  return { api: API, mount, autoPlace, freeIntervals, mapReserve, cssLength, safeImage, normalizeDetected, compassName,
    countryName, panelWidth, isCompact, esc };
});
