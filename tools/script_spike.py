#!/usr/bin/env python3
"""
Feasibility spike: can closed-form maths recognise the WRITING on signs (script family)?

  python3 tools/script_spike.py fetch [--types language,road-signs,license-plates] [--refresh]
      crops of GeoGuessr's own clue placements (data/calibration/pano_clues.json, read through a
      snapshot) rendered from zoom-5 Street View tiles at ~0.03 deg/px, at most 28 x 17 deg
      -> scratch/script_spike/crops/<key>.jpg, scratch/script_spike/placements.json
  python3 tools/script_spike.py fetch --types nature,pole,architecture,bollards,road --max-per-type 250
      control crops (negatives for the text detector, Noise lines of the script model)
  python3 tools/script_spike.py recrop
      cut crops rendered with the old 40 deg limit to the central MAX_T view (disk)
  python3 tools/script_spike.py codebook [--k 64] | codebook --k 512 --words
      k-means codebooks of glyph shapes (train-split crops)
  python3 tools/script_spike.py synth [--n 15000] [--res 0.03] [--show]
      font-rendered signs of every script (system fonts) in real control crops, through the same
      line detector -> scratch/script_spike/synth_lines[_<res>].npz
  python3 tools/script_spike.py lines [--res 0.03] [--source tiles|equirect]
      text lines in the 20 x 12 deg window of every crop -> scratch/script_spike/lines2_<tag>.npz
  python3 tools/script_spike.py eval [--res ...] [--split cv|calib|test] [--cfg key=value ...] [--save]
      detection hit rate, script accuracy (5 folds grouped by panorama over train + calib; 'calib':
      fit on train, evaluate on CALIB; 'test': fit on train + calib, TEST once), confusion matrices,
      majority baseline, gain over the prior with panorama-bootstrap CIs, control views, runtime
      -> scratch/script_spike/eval_<tag>.json
  python3 tools/script_spike.py frames [--width 1112] [--n 40]
      ScriptReader on whole frames (28 deg crops resized to a zoom-3 live frame, no window):
      runtime per frame and the rate of frames with a line above the saved threshold
  python3 tools/script_spike.py show -- KEY...   ScriptReader overlays in scratch/script_spike/debug/

Script labels: the country's dominant script on signs, overridden by the card (English cards in
India, Latin licence plates almost everywhere); Mixed / Khmer / Lao / Tibetan countries are left
out.  The labels are per country, not per line: English / digit lines on non-Latin signs are common.
Licence plates are blurred by Street View: detection only, no script label used.
Offline only: render_from_tiles needs the panorama's Street View metadata (it holds the location).
BLAS threads default to 4 (OPENBLAS_NUM_THREADS / OMP_NUM_THREADS / MKL_NUM_THREADS override).
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import unicodedata  # noqa: E402
from collections import Counter  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from calibrate import split_of  # noqa: E402
from engine import script_detect as sd  # noqa: E402
from engine import streetview  # noqa: E402
from engine.panorama import bilinear  # noqa: E402

SPIKE = os.path.join(ROOT, "scratch", "script_spike")
CROPS = os.path.join(SPIKE, "crops")
CLUES = os.path.join(ROOT, "data", "calibration", "pano_clues.json")
SNAPSHOT = os.path.join(SPIKE, "pano_clues_snapshot.json")
PLACEMENTS = os.path.join(SPIKE, "placements.json")
INDEX = os.path.join(ROOT, "scratch", "dataset", "index.jsonl")
PANOS = os.path.join(ROOT, "scratch", "dataset", "panos")

TEXT_TYPES = ("language", "road-signs", "license-plates")
DEG_PER_PX = 0.03
MAX_T = math.tan(math.radians(14.0))  # crops wider than 28 deg are cut to the central 28 deg
ASPECT = 0.6

COUNTRY_SCRIPT = {}
for _s, _cc in (("Cyrillic", "RU BG UA BY KZ KG MK MN TJ"),
                ("Greek", "GR CY"),
                ("CJK", "JP TW HK CN MO"),
                ("Hangul", "KR"),
                ("Thai", "TH"),
                ("Indic", "IN BD NP LK"),
                ("Hebrew", "IL"),
                ("Arabic", "AE JO OM QA SA EG LB IQ PK KW BH"),
                ("Khmer", "KH"),
                ("Lao", "LA"),
                ("Tibetan", "BT"),
                ("Mixed", "RS TN MA DZ MY")):
    for _c in _cc.split():
        COUNTRY_SCRIPT[_c] = _s
# licence plates: Latin-looking letters + digits except these
PLATE_SCRIPT = {"TH": "Thai", "JP": "CJK", "KR": "Hangul", "BD": "Indic"}
CARD_SCRIPT = (("arabic", "Arabic"), ("english", "Latin"), ("latin-script", "Latin"), ("cyrillic", "Cyrillic"),
               ("chinese-characters", "CJK"), ("area-code", "Latin"), ("phone", "Latin"),
               ("postcode", "Latin"), ("postal-code", "Latin"), ("house-number", "Latin"),
               ("devanagari", "Indic"), ("bengali", "Indic"), ("tamil", "Indic"), ("telugu", "Indic"),
               ("gujarati", "Indic"), ("kannada", "Indic"), ("oriya", "Indic"), ("assamese", "Indic"),
               ("sinhala", "Indic"))


def script_label(card):
    cc = card.get("countryCode") or ""
    title = (card.get("title") or "").lower()
    if card.get("type") == "license-plates":
        return PLATE_SCRIPT.get(cc, "Latin")
    if card.get("type") not in TEXT_TYPES:
        return "none"
    for kw, s in CARD_SCRIPT:
        if kw in title:
            return s
    return COUNTRY_SCRIPT.get(cc, "Latin")


def read_json_retry(path, tries=20):
    for i in range(tries):
        try:
            with open(path) as f:
                return json.load(f)
        except (ValueError, OSError):
            time.sleep(1.0 + i)
    raise RuntimeError("cannot read " + path)


def snapshot(refresh=False):
    if refresh or not os.path.exists(SNAPSHOT):
        d = read_json_retry(CLUES)
        os.makedirs(SPIKE, exist_ok=True)
        with open(SNAPSHOT + ".tmp", "w") as f:
            json.dump(d, f)
        os.replace(SNAPSHOT + ".tmp", SNAPSHOT)
    with open(SNAPSHOT) as f:
        return json.load(f)


def load_index():
    idx = {}
    with open(INDEX) as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            idx.setdefault(r["pano_id"], r)
    return idx


def view_size(zoom, res=DEG_PER_PX):
    """GeoGuessr zoom -> tan(hfov/2) (measured: 2^(1-zoom)), cut to MAX_T; crop size at res deg/px."""
    t = min(2.0 ** (1.0 - zoom), MAX_T)
    iw = int(round(2 * t / math.tan(math.radians(res))))
    return t, iw, int(round(iw * ASPECT))


def build_placements(types, refresh=False):
    clues = snapshot(refresh)
    idx = load_index()
    out, seen = [], {}
    for pid, cards in clues.items():
        r = idx.get(pid)
        if r is None or not isinstance(cards, list):
            continue
        for i, c in enumerate(cards):
            if c.get("type") not in types or c.get("heading") is None:
                continue
            vk = (pid, round(c["heading"], 1), round(c.get("pitch") or 0.0, 1), c.get("zoom"))
            if vk in seen:  # two cards on the same view share the crop
                seen[vk]["cards"].append(c.get("id"))
                continue
            p = {"key": "%s_%d" % (pid, i), "pano_id": pid, "type": c["type"], "id": c.get("id"),
                 "title": (c.get("title") or "").replace("clue.", "").replace("-title", ""),
                 "cc": c.get("countryCode"), "script": script_label(c), "heading": c["heading"],
                 "pitch": c.get("pitch") or 0.0, "zoom": c.get("zoom") or 2.0, "split": split_of(r),
                 "cards": [c.get("id")]}
            seen[vk] = p
            out.append(p)
    return out


def render_from_tiles(meta, yaw, pitch, t_h, iw, ih, fetch, level=None):
    """Perspective crop rendered from the tiles of zoom `level` (default: the highest); only the
    tiles that cover the view are fetched.  OFFLINE ONLY (finished / public rounds): meta is the
    panorama's Street View metadata (levels, tile_size, heading), which includes its location.
    fetch(x, y, z) -> PIL tile; the centre column of the panorama looks along meta["heading"]."""
    levels = meta["levels"]
    z = len(levels) - 1 if level is None else level
    H, W = levels[z]
    ts = (meta.get("tile_size") or [512, 512])[0]
    nx, ny = int(math.ceil(W / ts)), int(math.ceil(H / ts))
    az, el = sd.view_angles(yaw, pitch, t_h, iw, ih)
    phi = (az - (meta.get("heading") or 0.0) + 180.0) % 360.0 - 180.0
    px = (phi + 180.0) / 360.0 * W - 0.5
    py = (90.0 - el) / 180.0 * H - 0.5
    c = px[ih // 2, iw // 2]
    px = c + (px - c + W / 2) % W - W / 2  # unwrap around the view centre
    tx0, tx1 = int(np.floor(px.min() / ts)), int(np.floor((px.max() + 1) / ts))
    ty0, ty1 = max(int(np.floor(py.min() / ts)), 0), min(int(np.floor((py.max() + 1) / ts)), ny - 1)
    mosaic = np.zeros(((ty1 - ty0 + 1) * ts, (tx1 - tx0 + 1) * ts, 3), np.uint8)
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            tile = fetch(tx % nx, ty, z)
            a = np.asarray(tile.convert("RGB"))[:ts, :ts]
            mosaic[(ty - ty0) * ts:(ty - ty0) * ts + a.shape[0], (tx - tx0) * ts:(tx - tx0) * ts + a.shape[1]] = a
    out = bilinear(mosaic, px - tx0 * ts, py - ty0 * ts)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def fetch_one(p, idx):
    path = os.path.join(CROPS, p["key"] + ".jpg")
    if os.path.exists(path):
        return "cached"
    meta = idx[p["pano_id"]]
    t, iw, ih = view_size(p["zoom"])

    def fetch(x, y, z):
        return streetview._fetch_tile(p["pano_id"], x, y, z)

    try:
        img = render_from_tiles(meta, p["heading"], p["pitch"], t, iw, ih, fetch)
    except Exception as e:  # removed panorama / tile error
        return "error: %s" % e
    img.save(path + ".tmp.jpg", quality=88)
    os.replace(path + ".tmp.jpg", path)
    return "ok"


def cmd_fetch(args):
    types = set(args.types.split(","))
    plc = build_placements(types, refresh=args.refresh)
    if args.max_per_type:  # e.g. a random subset of 'nature' views as detector negatives
        rng = np.random.RandomState(0)
        keep = []
        for t in sorted(types):
            sub = [p for p in plc if p["type"] == t]
            if len(sub) > args.max_per_type:
                sub = [sub[i] for i in sorted(rng.choice(len(sub), args.max_per_type, replace=False))]
            keep += sub
        plc = keep
    if args.limit:
        plc = plc[:args.limit]
    os.makedirs(CROPS, exist_ok=True)
    idx = load_index()
    old = {}
    if os.path.exists(PLACEMENTS):
        old = {p["key"]: p for p in json.load(open(PLACEMENTS))}
    for p in plc:
        old[p["key"]] = p
    with open(PLACEMENTS, "w") as f:
        json.dump(sorted(old.values(), key=lambda p: p["key"]), f)
    print("placements", len(plc), Counter(p["script"] for p in plc).most_common())
    t0, stat = time.time(), Counter()
    with ThreadPoolExecutor(min(args.workers, 4)) as ex:
        for k, res in enumerate(ex.map(lambda p: fetch_one(p, idx), plc)):
            stat[res.split(":")[0]] += 1
            if res.startswith("error"):
                print(plc[k]["key"], res)
            if (k + 1) % 100 == 0:
                print("%d/%d %.0fs %s" % (k + 1, len(plc), time.time() - t0, dict(stat)), flush=True)
    print(dict(stat), "%.0fs" % (time.time() - t0))


def cmd_recrop(args):
    """Centre-cut crops larger than view_size (a centred sub-window of a pinhole view is the
    pinhole view of the smaller FOV)."""
    n = 0
    for p in json.load(open(PLACEMENTS)):
        path = os.path.join(CROPS, p["key"] + ".jpg")
        if not os.path.exists(path):
            continue
        _, iw, ih = view_size(p["zoom"])
        img = Image.open(path)
        W, H = img.size
        if W <= iw + 1 and H <= ih + 1:
            continue
        x0, y0 = (W - iw) // 2, (H - ih) // 2
        img.convert("RGB").crop((x0, y0, x0 + iw, y0 + ih)).save(path + ".tmp.jpg", quality=88)
        os.replace(path + ".tmp.jpg", path)
        n += 1
    print("recropped", n)


def crop_image(p, res, source, upsample=1.0):
    """The crop of placement p at res deg/px: from the cached 0.03 deg/px tile render (downsampled
    when res > 0.03), or rendered from the 2048 px equirectangular panorama of the dataset."""
    t, iw, ih = view_size(p["zoom"], res)
    if source == "equirect":
        meta = _INDEX[p["pano_id"]]
        rgb = np.asarray(Image.open(os.path.join(PANOS, p["pano_id"] + ".jpg")).convert("RGB"))
        img = sd.render_from_equirect(rgb, meta.get("heading") or 0.0, p["heading"], p["pitch"], t, iw, ih)
    else:
        img = Image.open(os.path.join(CROPS, p["key"] + ".jpg")).convert("RGB")
        if res > DEG_PER_PX * 1.01:
            img = img.resize((iw, ih), Image.LANCZOS)
    if upsample != 1.0:
        img = img.resize((int(round(img.size[0] * upsample)), int(round(img.size[1] * upsample))), Image.BICUBIC)
    return img


_INDEX, _CODEBOOK = {}, None
CODEBOOK = os.path.join(SPIKE, "glyph_codebook.npz")
WORDBOOK = os.path.join(SPIKE, "glyph_words.npz")  # larger codebook: glyph words of the naive Bayes
WINDOW_T = 2.0 ** (1.0 - 3.5)  # detection window: the 20 x 12 deg view of a zoom-3.5 placement


def window_frac(p):
    return min(1.0, WINDOW_T / view_size(p["zoom"])[0])


def _lines_init(source, use_codebook):
    global _INDEX, _CODEBOOK
    if source == "equirect":
        _INDEX = load_index()
    _CODEBOOK = None
    if use_codebook:
        _CODEBOOK = np.load(CODEBOOK)["centres"]
        if os.path.exists(WORDBOOK):
            _CODEBOOK = (_CODEBOOK, np.load(WORDBOOK)["centres"])


def _lines_one(job):
    p, res, source, up, glyphs = job
    try:
        img = crop_image(p, res, source, up)
    except (OSError, KeyError) as e:
        return p["key"], None, str(e)
    fr = window_frac(p)
    t0 = time.time()
    o = sd.detect_lines(img, codebook=_CODEBOOK, window=(fr, fr), glyphs=glyphs)
    o["size"], o["time"] = img.size, time.time() - t0
    return p["key"], o, ""


def run_lines(plc, res, source, up, workers, glyphs=False, use_codebook=True):
    """Text lines inside the detection window of every crop (worker pool) -> yields (key, dict)."""
    from multiprocessing import Pool
    jobs = [(p, res, source, up, glyphs) for p in plc]
    t0 = time.time()
    with Pool(min(workers, 4), initializer=_lines_init, initargs=(source, use_codebook),
              maxtasksperchild=200) as pool:
        for k, (key, o, err) in enumerate(pool.imap(_lines_one, jobs, chunksize=4)):
            if (k + 1) % 250 == 0:
                print("%d/%d %.0fs" % (k + 1, len(jobs), time.time() - t0), flush=True)
            if o is None:
                print(key, err)
                continue
            yield key, o


def all_placements():
    return [p for p in json.load(open(PLACEMENTS)) if os.path.exists(os.path.join(CROPS, p["key"] + ".jpg"))]


def cmd_codebook(args):
    """k-means codebook of glyph shape vectors (window lines of a sample of train-split text crops)."""
    plc = [p for p in all_placements() if p["split"] == "train" and p["type"] in ("language", "road-signs")]
    rng = np.random.RandomState(0)
    plc = [plc[i] for i in sorted(rng.choice(len(plc), min(args.n, len(plc)), replace=False))]
    G = []
    for key, o in run_lines(plc, DEG_PER_PX, "tiles", 1.0, args.workers, glyphs=True, use_codebook=False):
        G += [g for g, n in zip(o["glyphs"], o["ncomp"]) if n >= 3]
    X = np.concatenate(G)
    if len(X) > args.max_glyphs:
        X = X[rng.choice(len(X), args.max_glyphs, replace=False)]
    C = sd.kmeans(X, args.k)
    np.savez(WORDBOOK if args.words else CODEBOOK, centres=C)
    print("codebook", C.shape, "from", len(X), "glyphs of", len(G), "lines")


def lines_path(res, source, up):
    return os.path.join(SPIKE, "lines2_%s_%g%s.npz" % (source, res, "" if up == 1.0 else "_up%g" % up))


def cmd_lines(args):
    plc = all_placements()
    if args.source == "equirect":  # panoramas of the dataset only
        plc = [p for p in plc if os.path.exists(os.path.join(PANOS, p["pano_id"] + ".jpg"))]
    if args.limit:
        plc = plc[:args.limit]
    F, S, B, N, C, Wd, keys, sizes, times = [], [], [], [], [], [], [], [], []
    for key, o in run_lines(plc, args.res, args.source, args.upsample, args.workers):
        Wd += o.get("words") or []
        C.append(np.full(len(o["feats"]), len(keys), np.int32))
        keys.append(key)
        sizes.append(o["size"])
        times.append(o["time"])
        F.append(o["feats"].astype(np.float32))
        S.append(o["script"].astype(np.float16))
        B.append(o["boxes"].astype(np.int32))
        N.append(o["ncomp"].astype(np.int32))
    np.savez(lines_path(args.res, args.source, args.upsample), feats=np.concatenate(F), script=np.concatenate(S),
             boxes=np.concatenate(B), ncomp=np.concatenate(N), crop=np.concatenate(C), keys=np.array(keys),
             sizes=np.array(sizes), times=np.array(times),
             words=np.concatenate(Wd).astype(np.int16) if Wd else np.zeros(0, np.int16))
    print("lines", sum(len(f) for f in F), "crops", len(keys), "median %.2fs/crop" % np.median(times))


CLASSES = sd.SCRIPTS
SCRIPT_TYPES = ("language", "road-signs")  # licence plates are blurred by Street View


def fold_of(pano_id, k=5):
    return int(hashlib.md5(pano_id.encode()).hexdigest(), 16) % k


def crop_script(c):
    """Script label of a placement (recomputed: the mapping may have changed since the fetch)."""
    return script_label({"countryCode": c["cc"], "title": c["title"], "type": c["type"]})


def country_class(c):
    """Index in CLASSES of the dominant sign script of the placement's country (-1: not modelled)."""
    s = COUNTRY_SCRIPT.get(c.get("cc") or "", "Latin" if c.get("cc") else None)
    return CLASSES.index(s) if s in CLASSES else -1


def load_lines(path, plc_by_key):
    d = np.load(path)
    crops = [plc_by_key[str(k)] for k in d["keys"]]
    scr = [crop_script(c) for c in crops]
    L = {"F": d["feats"].astype(np.float64), "S": d["script"].astype(np.float64), "crop_of": d["crop"],
         "boxes": d["boxes"], "ctype": np.array([c["type"] for c in crops]),
         "lab": np.array([CLASSES.index(s) if s in CLASSES else -1 for s in scr]),
         "clab": np.array([country_class(c) for c in crops]),
         "pano": np.array([c["pano_id"] for c in crops]),
         "split": np.array([c["split"] for c in crops]), "times": d["times"], "crops": crops,
         "ncomp": d["ncomp"], "words": d["words"].astype(np.int64) if "words" in d else None}
    L["woff"] = np.concatenate([[0], np.cumsum(L["ncomp"])[:-1]])
    L["is_ctrl"] = ~np.isin(L["ctype"], TEXT_TYPES)
    L["is_script"] = np.isin(L["ctype"], SCRIPT_TYPES) & (L["lab"] >= 0)
    return L


def robust_scale(F):
    c = np.median(F, 0)
    s = np.percentile(F, 75, 0) - np.percentile(F, 25, 0)
    return c, np.where(s > 1e-6, s, np.maximum(F.std(0), 1e-3))


def fit_text_scorer(F, ip, ineg, rng, max_n=20000):
    if ip.size > max_n:
        ip = rng.choice(ip, max_n, replace=False)
    if ineg.size > max_n:
        ineg = rng.choice(ineg, max_n, replace=False)
    sel = np.concatenate([ip, ineg])
    y = np.concatenate([np.ones(ip.size, int), np.zeros(ineg.size, int)])
    c, s = robust_scale(F[sel])
    w = np.where(y == 1, 0.5 / ip.size, 0.5 / ineg.size) * len(sel)
    W = sd.fit_logit(sd.expand(F[sel], c, s), y, 2, w, lam=3.0, iters=15)
    return {"c": c, "s": s, "W": W}


def text_score(m, F, chunk=50000):
    out = np.empty(len(F))
    for i in range(0, len(F), chunk):
        z = sd.expand(F[i:i + chunk], m["c"], m["s"]) @ m["W"]
        out[i:i + chunk] = z[:, 1] - z[:, 0]
    return out


def top_lines(score, crop_of_line, n_crops, k=3, thr=-np.inf, mask=None):
    """Indices of the k best-scoring lines of every crop (score > thr, inside mask)."""
    cand = np.nonzero((score > thr) & (np.ones(len(score), bool) if mask is None else mask))[0]
    cand = cand[np.lexsort((-score[cand], crop_of_line[cand]))]
    out = [[] for _ in range(n_crops)]
    for i in cand:
        c = crop_of_line[i]
        if len(out[c]) < k:
            out[c].append(i)
    return out


def crop_max(score, crop_of_line, n_crops):
    out = np.full(n_crops, -np.inf)
    np.maximum.at(out, crop_of_line, score)
    return out


# writing features: 'stats' = line statistics (sd.FEATURES), 'writing' = zone profiles + glyph
# shapes + glyph codebook histogram, 'all' = both
def subset(L, rows):
    """Line set {F, S, words, off} of the given lines (words: their glyph words, flat; line i
    owns words[off[i]:off[i + 1]])."""
    rows = np.asarray(rows, int)
    D = {"F": L["F"][rows], "S": L["S"][rows], "words": None, "off": None}
    if L.get("words") is not None:
        st, n = L["woff"][rows], L["ncomp"][rows]
        idx = np.repeat(st, n) + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
        D["words"], D["off"] = L["words"][idx], np.concatenate([[0], np.cumsum(n)])
    return D


def concat(A, B):
    D = {"F": np.vstack([A["F"], B["F"]]), "S": np.vstack([A["S"], B["S"]]), "words": None, "off": None}
    if A["words"] is not None and B["words"] is not None:
        D["words"] = np.concatenate([A["words"], B["words"]])
        D["off"] = np.concatenate([A["off"], A["off"][-1] + B["off"][1:]])
    return D


def nb_loglik(logp, D):
    """Sum over every line's glyphs of log p(word | class) -> (lines, classes), and glyph counts."""
    G = logp[D["words"]]
    n = np.diff(D["off"])
    out = np.zeros((len(n), logp.shape[1]))
    nz = n > 0
    out[nz] = np.add.reduceat(G, D["off"][:-1][nz], axis=0)
    return out, n


def fit_nb(D, y, w, k, alpha=0.5):
    """Naive Bayes over glyph words: log p(word | class), every line's weight spread over its glyphs."""
    n = np.diff(D["off"])
    C = np.zeros((int(D["words"].max()) + 1 if len(D["words"]) else 1, k))
    np.add.at(C, (D["words"], np.repeat(y, n)), np.repeat(w / np.maximum(n, 1), n))
    C *= n.sum() / max(C.sum(), 1e-9)  # back to glyph units
    return np.log((C + alpha) / (C.sum(0, keepdims=True) + alpha * len(C)))


def script_design(D, m):
    extra = None
    if m.get("nb") is not None:  # per-glyph mean naive-Bayes log-likelihood, centred
        ll, n = nb_loglik(m["nb"], D)
        ll /= np.maximum(n, 1)[:, None]
        extra = ll - ll.mean(1, keepdims=True)
    return sd.line_design(D["F"], D["S"], m, extra)


def script_proba(m, D):
    if m.get("mode") == "nb":
        ll, n = nb_loglik(m["nb"], D)
        return sd.softmax(m["tau"] * ll)
    return sd.softmax(script_design(D, m) @ m["W"])


_SYN = {}


def load_synth(min_comp=2):
    if "F" not in _SYN:
        d = np.load(synth_path(_SYN.get("res", DEG_PER_PX)))
        A = {"F": d["feats"].astype(np.float64), "S": d["script"].astype(np.float64), "ncomp": d["ncomp"],
             "words": d["words"].astype(np.int64) if "words" in d else None}
        A["woff"] = np.concatenate([[0], np.cumsum(A["ncomp"])[:-1]])
        keep = np.nonzero(d["ncomp"] >= min_comp)[0]
        _SYN.update(subset(A, keep), y=d["y"][keep], sample=d["sample"][keep])
    return _SYN


def fit_script_mil(L, tl, crops_tr, noise_rows, cfg):
    """Line classifier over ALL_CLASSES (scripts + Digits + Noise).

    cfg['train'] 'real': the k best text lines of every training crop, labelled with the crop's
    script; multiple-instance refinement: a non-Latin crop needs only ONE line in its script (the
    others may be digits / English), so after the first fit its whole weight goes to the line most
    probable for its label.  'synth': the font-rendered lines (scripts + Digits).  'both': both,
    the real crops weighted cfg['real_w'].  Noise: the best lines of the training control crops.
    NOTE: 'both' (the default, used for the saved model) is no better than 'synth' in CV: the
    country-labelled real lines (often English / digits) add no script signal; drop that path or
    replace it by hand-checked lines if the spike is revisited.
    Classes weighted to equal total weight.  cfg['nb']: 'off' | 'feat' (naive Bayes over glyph
    words of the synthetic + noise lines as extra logit inputs) | 'only' (naive Bayes over all
    training lines, line posterior = softmax(cfg['nb_tau'] x log-likelihood))."""
    lab, train, real_w, k = L["lab"], cfg["train"], cfg["real_w"], len(ALL_CLASSES)
    crops_tr = [c for c in crops_tr if tl[c]] if train != "synth" else []
    rows = np.array([i for c in crops_tr for i in tl[c]], int)
    nreal = len(rows)
    D = subset(L, np.concatenate([rows, noise_rows]))
    y = np.concatenate([lab[L["crop_of"][rows]], np.full(len(noise_rows), NOISE, int)])
    w0 = np.concatenate([np.full(len(tl[c]), real_w / len(tl[c])) for c in crops_tr] + [np.ones(len(noise_rows))])
    nown = len(y)
    if train != "real":
        syn = load_synth()
        D = concat(D, syn)
        y = np.concatenate([y, syn["y"]])
        w0 = np.concatenate([w0, np.ones(len(syn["y"]))])
    m = {"feats": cfg["feats"], "k": k}
    if cfg["nb"] == "only":
        m.update(mode="nb", tau=cfg["nb_tau"], nb=fit_nb(D, y, w0, k))
        return m
    if cfg["nb"] == "feat":
        own = np.arange(len(y)) < nreal  # real script lines stay out of the naive Bayes
        m["nb"] = fit_nb(subset_rows(D, ~own), y[~own], w0[~own], k)
    m["cf"], m["sf"] = robust_scale(D["F"])
    m["cs"], m["ss"] = robust_scale(D["S"])
    X = script_design(D, m)

    def class_w(w):  # own (real + noise) and synthetic lines of a class weighted separately
        src = (np.arange(len(y)) >= nown).astype(int)
        tot = np.zeros((2, k))
        np.add.at(tot, (src, y), w)
        has = (tot > 0).sum(0)
        f = np.where(tot > 0, 1.0 / np.maximum(tot, 1e-12), 0.0) / np.maximum(has, 1)[None, :]
        if train == "both":
            f[0] *= 2 * real_w / (1 + real_w)
            f[1] *= 2 / (1 + real_w)
        return w * f[src, y] * len(y) / k

    W = sd.fit_logit_lbfgs(X, y, k, class_w(w0), lam=cfg["lam"])
    for _ in range(cfg["script_mil"] if crops_tr else 0):
        P = sd.softmax(X[:nreal] @ W)
        w = w0.copy()
        off = 0
        for c in crops_tr:
            n = len(tl[c])
            if lab[c] != 0 and n > 1:
                w[off:off + n] = 0.0
                w[off + int(np.argmax(P[off:off + n, lab[c]]))] = real_w
            off += n
        W = sd.fit_logit_lbfgs(X, y, k, class_w(w), lam=cfg["lam"], W0=W)
    m["W"] = W
    return m


def subset_rows(D, mask):
    """Rows of a line set selected by a boolean mask."""
    idx = np.nonzero(mask)[0]
    out = {"F": D["F"][idx], "S": D["S"][idx], "words": None, "off": None}
    if D["words"] is not None:
        n = np.diff(D["off"])[idx]
        st = D["off"][:-1][idx]
        g = np.repeat(st, n) + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
        out["words"], out["off"] = D["words"][g], np.concatenate([[0], np.cumsum(n)])
    return out


def fit_models(L, tr_c, cfg, seed=0):
    """Text scorer by multiple-instance learning (every language crop holds at least one text line
    inside the window, control crops hold none): all window lines of the language crops vs the
    control lines, then the 2 best lines of every positive bag, cfg['mil'] times.  Threshold =
    cfg['fpr'] false alarms on the training control crops.  Script model: fit_script_mil on the
    cfg['k'] best lines above the threshold of the training language / road-sign crops."""
    rng = np.random.RandomState(seed)
    F, crop_of = L["F"], L["crop_of"]
    nc = len(L["crops"])
    trl = tr_c[crop_of]
    bag = trl & (L["ctype"][crop_of] == "language")
    neg = np.nonzero(trl & L["is_ctrl"][crop_of])[0]
    m = fit_text_scorer(F, np.nonzero(bag)[0], neg, rng)
    for _ in range(cfg["mil"]):
        tl = top_lines(text_score(m, F), crop_of, nc, 2, mask=bag)
        m = fit_text_scorer(F, np.array([i for c in range(nc) for i in tl[c]], int), neg, rng)
    sc = text_score(m, F)
    thr = float(np.percentile(crop_max(sc, crop_of, nc)[tr_c & L["is_ctrl"]], 100 * (1 - cfg["fpr"])))
    tl = top_lines(sc, crop_of, nc, cfg["k"], thr)
    tn = top_lines(sc, crop_of, nc, cfg["k"], mask=trl & L["is_ctrl"][crop_of])
    noise = np.array([i for c in range(nc) for i in tn[c]], int)
    sm = fit_script_mil(L, tl, np.nonzero(tr_c & L["is_script"])[0], noise, cfg)
    return m, thr, sc, tl, sm


def roc_points(pos_scores, neg_scores, fprs=(0.05, 0.1, 0.2)):
    neg = np.sort(neg_scores)
    out = []
    for f in fprs:
        thr = neg[int(np.floor((1 - f) * (len(neg) - 1)))] if len(neg) else 0.0
        out.append((f, float((pos_scores > thr).mean()), float(thr)))
    return out


def auc(pos, neg):
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    r = np.cumsum(cnt) - (cnt - 1) / 2.0  # average rank of tied values
    r = r[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


DEFAULT_CFG = {"k": 3, "fpr": 0.1, "mil": 3, "feats": "all", "lam": 3.0, "script_mil": 2, "tau": 0.5,
               "train": "both", "real_w": 1.0, "nb": "off", "nb_tau": 0.3}


RULES = ["max_tau0.5"] + ["mix_q%g_pw%g" % (q, pw) for q in (0.3, 0.5, 0.7) for pw in (0.0, 0.5, 1.0)]


crop_loglik = sd.crop_loglik


def decide(P, rule, logprior):
    """Crop script from the class probabilities of its text lines -> (class, non-Latin score,
    posterior over CLASSES or None).  'max_tau<t>': the most probable non-Latin class of any line
    if above t, else Latin (one line in a script is enough; digits and English appear
    everywhere).  'mix_q<q>_pw<w>': crop_loglik + w x log prior of the training crops."""
    if rule.startswith("max"):
        nl = P[:, 1:len(CLASSES)]
        j = np.unravel_index(np.argmax(nl), nl.shape)
        return (j[1] + 1 if nl[j] > float(rule[7:]) else 0), float(nl[j]), None
    q, pw = [float(v[1:] if v[0] == "q" else v[2:]) for v in rule[4:].split("_")]
    ll = crop_loglik(P, q) + pw * logprior
    post = sd.softmax(ll[None, :])[0]
    return int(np.argmax(ll)), float(np.log(post[1:].sum() + 1e-12) - np.log(post[0] + 1e-12)), post


def script_metrics(lab, pred, sel):
    k = len(CLASSES)
    conf = np.zeros((k, k), int)
    np.add.at(conf, (lab[sel], pred[sel]), 1)
    rec = [conf[i, i] / conf[i].sum() for i in range(k) if conf[i].sum() >= 5]
    prec = conf.diagonal() / np.maximum(conf.sum(0), 1)
    return {"n": int(conf.sum()), "accuracy": float(np.trace(conf) / max(conf.sum(), 1)),
            "balanced_accuracy": float(np.mean(rec)) if rec else float("nan"), "balanced_classes": len(rec),
            "recall": {CLASSES[i]: round(float(conf[i, i] / conf[i].sum()), 3) for i in range(k) if conf[i].sum()},
            "precision": {CLASSES[i]: round(float(prec[i]), 3) for i in range(k) if conf[:, i].sum()},
            "confusion": conf.tolist()}


def boot_ci(groups, vals, denom=None, n=2000, seed=0):
    """95% interval of sum(vals) / sum(denom) (default: the mean), resampling the groups
    (panoramas) with replacement."""
    vals = np.asarray(vals, np.float64)
    denom = np.ones(len(vals)) if denom is None else np.asarray(denom, np.float64)
    if not len(vals):
        return [float("nan"), float("nan")]
    _, g = np.unique(groups, return_inverse=True)
    S, D = np.bincount(g, vals), np.bincount(g, denom)
    rng = np.random.RandomState(seed)
    est = []
    for i in range(0, n, 250):
        cnt = rng.multinomial(len(S), np.full(len(S), 1.0 / len(S)), size=min(250, n - i)).astype(np.float64)
        est.append((cnt @ S) / np.maximum(cnt @ D, 1e-12))
    return [round(float(v), 4) for v in np.percentile(np.concatenate(est), [2.5, 97.5])]


def evaluate(path, plc_by_key, split_mode="cv", cfg=None, verbose=True, rules=None):
    """Detection (crop-level ROC of the best window line vs the control crops) and script
    accuracy of the language / road-sign crops with a line above the text threshold.

    split_mode 'cv': 5 folds grouped by panorama over train + calib panoramas;
               'calib': fit on train, evaluate on CALIB (the user's rounds: out-of-distribution check);
               'test': fit on train + calib, evaluate on the test panoramas (report once)."""
    cfg = dict(DEFAULT_CFG, **(cfg or {}))
    rules = rules or RULES
    L = load_lines(path, plc_by_key)
    crops, crop_of = L["crops"], L["crop_of"]
    nc = len(crops)
    ctype, lab, is_ctrl = L["ctype"], L["lab"], L["is_ctrl"]
    dev = np.isin(L["split"], ["train", "calib"])
    if split_mode == "cv":
        fold = np.array([fold_of(c["pano_id"]) for c in crops])
        folds = [(dev & (fold != f), dev & (fold == f)) for f in range(5)]
    elif split_mode == "calib":
        folds = [(L["split"] == "train", L["split"] == "calib")]
    else:
        folds = [(dev, L["split"] == "test")]
    crop_score = np.full(nc, -np.inf)
    evaluated = np.zeros(nc, bool)
    best_line = np.full(nc, -1)
    probs, logprior = {}, np.zeros((nc, len(CLASSES)))
    t0 = time.time()
    for tr_c, te_c in folds:
        m, thr, sc, tl, sm = fit_models(L, tr_c, cfg)
        crop_score[te_c] = crop_max(sc, crop_of, nc)[te_c]
        evaluated |= te_c
        cnt = np.bincount(lab[tr_c & L["is_script"]], minlength=len(CLASSES)) + 1.0
        logprior[te_c] = np.log(cnt / cnt.sum())
        for c in np.nonzero(te_c)[0]:
            if tl[c]:
                best_line[c] = tl[c][0]
                r = np.array(tl[c])
                probs[c] = script_proba(sm, subset(L, r))
    ev = evaluated
    res = {"cfg": cfg, "split": split_mode, "n_crops": int(ev.sum()), "fit_time_s": round(time.time() - t0, 1)}
    # ROC points: threshold set on the evaluated control crops themselves; *_at_thr: the threshold
    # fitted on the training controls (the operating point a saved model uses)
    det = {t: roc_points(crop_score[ev & (ctype == t)], crop_score[ev & is_ctrl]) for t in TEXT_TYPES}
    det["n"] = {t: int((ev & (ctype == t)).sum()) for t in sorted(set(ctype))}
    det["n_ctrl"] = int((ev & is_ctrl).sum())
    hit = ev & (best_line >= 0)
    det["hit_rate_at_thr"] = {t: float(hit[ev & (ctype == t)].mean()) for t in TEXT_TYPES}
    det["false_alarm_at_thr"] = {t: float(hit[ev & (ctype == t)].mean()) for t in sorted(set(ctype[is_ctrl]))}
    det["false_alarm_pooled_at_thr"] = float(hit[ev & is_ctrl].mean())
    res["detection"] = det
    sel = hit & L["is_script"]
    lang = sel & (ctype == "language")
    sign = ev & L["is_script"]  # sign-aimed crops, with or without a detected line
    ctl = ev & is_ctrl & (L["clab"] >= 0)  # views not aimed at a sign, labelled with the country script
    ctl_hit = ctl & hit
    clab, pano = L["clab"], L["pano"]
    major = np.argmax(logprior, 1)
    res["script"], preds, scores = {}, {}, {}
    for rule in rules:
        pred, score, gain = np.full(nc, -1), np.full(nc, np.nan), np.zeros(nc)
        for c in np.nonzero(sel | ctl_hit)[0]:
            y = lab[c] if sel[c] else clab[c]
            pred[c], score[c], post = decide(probs[c], rule, logprior[c])
            if post is not None:  # log-loss gain over the prior (> 0: the writing adds information)
                gain[c] = math.log(max(post[y], 1e-6)) - logprior[c][y]
        nl, calls = sel & (lab != 0), sel & (pred > 0)
        r = {"all": script_metrics(lab, pred, sel), "language": script_metrics(lab, pred, lang),
             "majority_accuracy": float((major[sel] == lab[sel]).mean()) if sel.any() else float("nan"),
             "nonlatin_auc": auc(score[sel][lab[sel] != 0], score[sel][lab[sel] == 0]),
             "nonlatin": {"n": int(nl.sum()), "right": int((pred[nl] == lab[nl]).sum()), "calls": int(calls.sum()),
                          "calls_right": int((pred[calls] == lab[calls]).sum())}}
        if not rule.startswith("max"):
            cp = pred[ctl_hit]
            r.update(info_gain_nats=float(gain[sel].mean()), info_gain_ci95=boot_ci(pano[sel], gain[sel]),
                     info_gain_per_sign_crop=float(gain[sign].mean()),
                     info_gain_per_sign_crop_ci95=boot_ci(pano[sign], gain[sign]),
                     controls={"n": int(ctl.sum()), "with_line": int(ctl_hit.sum()),
                               "nonlatin_calls": int((cp > 0).sum()),
                               "nonlatin_right": int(((cp > 0) & (cp == clab[ctl_hit])).sum()),
                               "gain_per_crop": float(gain[ctl].mean()) if ctl.any() else float("nan"),
                               "gain_per_crop_ci95": boot_ci(pano[ctl], gain[ctl]),
                               "gain_per_line_crop": float(gain[ctl_hit].mean()) if ctl_hit.any() else float("nan")})
        res["script"][rule] = r
        preds[rule], scores[rule] = pred, score
    res["n_script_crops"] = {"evaluated": int((ev & L["is_script"]).sum()), "with_text": int(sel.sum())}
    res["runtime_median_s"] = float(np.median(L["times"]))
    res["runtime_p90_s"] = float(np.percentile(L["times"], 90))
    res["_pred"], res["_score"], res["_crop_score"], res["_best_line"], res["_crops"], res["_probs"] = (
        preds, scores, crop_score, best_line, crops, probs)
    res["_logprior"] = logprior
    if verbose:
        print_report(res)
    return res


def print_report(res):
    det = res["detection"]
    print("crops evaluated %d (%s)  per type %s" % (res["n_crops"], res["split"], det.get("n")))
    print("  ROC points (threshold from the %d evaluated control crops):" % det["n_ctrl"])
    for t in TEXT_TYPES:
        print("    detection %-15s" % t, "  ".join("TPR@FPR%.2f=%.3f" % (f, tp) for f, tp, _ in det[t]))
    print("  at the saved threshold (%.0f%% false alarms on the training controls): hit rate" % (
        100 * res["cfg"]["fpr"]), {k: round(v, 3) for k, v in det["hit_rate_at_thr"].items()},
        " false alarms %.3f" % det.get("false_alarm_pooled_at_thr", float("nan")),
        {k: round(v, 3) for k, v in det["false_alarm_at_thr"].items()})
    print("script crops", res["n_script_crops"])
    for name, r in res["script"].items():
        nl = r["nonlatin"]
        print("  %-16s acc %.3f (always-%s %.3f) bal %.3f over %d classes with >= 5 crops (n %d) | language acc %.3f"
              " bal %.3f (n %d) | AUC %.3f | non-Latin recall %d/%d precision %d/%d" % (
                  name, r["all"]["accuracy"], CLASSES[0], r["majority_accuracy"], r["all"]["balanced_accuracy"],
                  r["all"].get("balanced_classes", -1), r["all"]["n"], r["language"]["accuracy"], r["language"]["balanced_accuracy"], r["language"]["n"],
                  r["nonlatin_auc"], nl["right"], nl["n"], nl["calls_right"], nl["calls"]))
        if "info_gain_nats" in r:
            c = r["controls"]
            print("  %-16s gain %.3f nats/crop with a line %s, %.3f per sign crop %s | controls: %d/%d with a line,"
                  " %d non-Latin calls (%d right), gain %.3f per crop %s, %.3f per crop with a line" % (
                      "", r["info_gain_nats"], r["info_gain_ci95"], r["info_gain_per_sign_crop"],
                      r["info_gain_per_sign_crop_ci95"], c["with_line"], c["n"], c["nonlatin_calls"],
                      c["nonlatin_right"], c["gain_per_crop"], c["gain_per_crop_ci95"], c["gain_per_line_crop"]))
    best = max(res["script"], key=lambda k: res["script"][k]["all"]["balanced_accuracy"])
    for part in ("all", "language"):
        r = res["script"][best][part]
        conf = np.array(r["confusion"])
        print("confusion %s, %s (rows true, cols predicted)" % (best, part))
        print("%9s " % "" + " ".join("%5s" % c[:5] for c in CLASSES) + "   recall")
        for i, c in enumerate(CLASSES):
            n = conf[i].sum()
            print("%9s " % c + " ".join("%5d" % v for v in conf[i]) + ("   %.2f" % (conf[i, i] / n) if n else ""))
        print("precision", r["precision"])
    print("runtime per crop: median %.2fs p90 %.2fs" % (res["runtime_median_s"], res["runtime_p90_s"]))


def parse_cfg(args):
    cfg = dict(DEFAULT_CFG)
    for kv in args.cfg or []:
        k, v = kv.split("=")
        cfg[k] = type(DEFAULT_CFG[k])(v) if k in DEFAULT_CFG and not isinstance(DEFAULT_CFG[k], str) else v
    return cfg


def cmd_eval(args):
    plc_by_key = {p["key"]: p for p in json.load(open(PLACEMENTS))}
    path = lines_path(args.res, args.source, args.upsample)
    cfg = parse_cfg(args)
    _SYN["res"] = args.res
    res = evaluate(path, plc_by_key, args.split, cfg)
    out = {k: v for k, v in res.items() if not k.startswith("_")}
    tag = os.path.basename(path)[:-4] + "_" + args.split + ("_" + args.tag if args.tag else "")
    with open(os.path.join(SPIKE, "eval_%s.json" % tag), "w") as f:
        json.dump(out, f, indent=1)
    if args.save:
        fit_final(path, plc_by_key, cfg, args.rule)


# ------------------------------------------------------------------ synthetic text lines
# Reference writing rendered from the system fonts (the fonts are the alphabet, not a model):
# random words of each script on a sign board, pasted into a real control crop, scaled to the
# glyph sizes of the placements, blurred, noised and JPEG-compressed, then passed through the
# same line detector.  Lines inside the rendered text are labelled with its script.
ALL_CLASSES = sd.LINE_CLASSES  # scripts + Digits (synthetic) + Noise (best lines of control crops)
SYN_CLASSES = ALL_CLASSES[:-1]
DIGITS, NOISE = sd.DIGITS, sd.NOISE
SYNTH = os.path.join(SPIKE, "synth_lines.npz")


def synth_path(res=DEG_PER_PX):
    return SYNTH if abs(res - DEG_PER_PX) < 1e-9 else SYNTH.replace(".npz", "_%g.npz" % res)
FONT_DIRS = ("/usr/share/fonts", "/mnt/c/Windows/Fonts")
_LGC = ("DejaVuSans.ttf DejaVuSans-Bold.ttf DejaVuSansCondensed.ttf DejaVuSansCondensed-Bold.ttf DejaVuSerif.ttf "
        "DejaVuSerif-Bold.ttf LiberationSans-Regular.ttf LiberationSans-Bold.ttf LiberationSansNarrow-Regular.ttf "
        "LiberationSansNarrow-Bold.ttf LiberationSerif-Regular.ttf LiberationSerif-Bold.ttf Ubuntu-R.ttf Ubuntu-B.ttf "
        "Ubuntu-M.ttf Ubuntu-C.ttf NotoSans-Regular.ttf NotoSans-Bold.ttf NotoSerif-Regular.ttf arial.ttf "
        "segoeui.ttf segoeuib.ttf tahoma.ttf tahomabd.ttf times.ttf verdana.ttf verdanab.ttf calibri.ttf calibrib.ttf")
_INDIC = ("Devanagari", "Bengali", "Tamil", "Telugu", "Kannada", "Malayalam", "Gujarati", "Gurmukhi", "Oriya",
          "Sinhala")
FONTS = {
    "Latin": _LGC, "Cyrillic": _LGC, "Greek": _LGC, "Digits": _LGC,
    "CJK": "wqy-zenhei.ttc DroidSansFallbackFull.ttf ipag.ttf ipagp.ttf fonts-japanese-gothic.ttf msyh.ttc "
           "msyhbd.ttc msjh.ttc YuGothR.ttc YuGothB.ttc msgothic.ttc simsun.ttc",
    "Hangul": "wqy-zenhei.ttc DroidSansFallbackFull.ttf malgun.ttf malgunbd.ttf",
    "Thai": "NotoSansThai-Regular.ttf NotoSansThai-Bold.ttf NotoSerifThai-Regular.ttf Loma.otf Loma-Bold.otf "
            "LeelawUI.ttf LeelaUIb.ttf tahoma.ttf",
    "Hebrew": "NotoSansHebrew-Regular.ttf NotoSansHebrew-Bold.ttf NotoSerifHebrew-Regular.ttf DejaVuSans.ttf "
              "DejaVuSans-Bold.ttf LiberationSans-Regular.ttf LiberationSans-Bold.ttf arial.ttf tahoma.ttf",
    "Arabic": "NotoSansArabic-Regular.ttf NotoSansArabic-Bold.ttf NotoNaskhArabic-Regular.ttf "
              "NotoNaskhArabic-Bold.ttf NotoKufiArabic-Regular.ttf NotoKufiArabic-Bold.ttf DejaVuSans.ttf "
              "arial.ttf tahoma.ttf segoeui.ttf",
}
for _s in _INDIC:
    FONTS[_s] = "NotoSans%s-Regular.ttf NotoSans%s-Bold.ttf NotoSerif%s-Regular.ttf Nirmala.ttc" % (_s, _s, _s)
INDIC_WEIGHT = dict(zip(_INDIC, (0.35, 0.2, 0.1, 0.08, 0.08, 0.06, 0.04, 0.03, 0.02, 0.04)))
# (consonants, dependent vowel signs, virama) of the Indic blocks
INDIC_BLOCK = {"Devanagari": (0x915, 0x939, 0x93E, 0x94C, 0x94D), "Bengali": (0x995, 0x9B9, 0x9BE, 0x9CC, 0x9CD),
               "Gurmukhi": (0xA15, 0xA39, 0xA3E, 0xA4C, 0xA4D), "Gujarati": (0xA95, 0xAB9, 0xABE, 0xACC, 0xACD),
               "Oriya": (0xB15, 0xB39, 0xB3E, 0xB4C, 0xB4D), "Tamil": (0xB95, 0xBB9, 0xBBE, 0xBCC, 0xBCD),
               "Telugu": (0xC15, 0xC39, 0xC3E, 0xC4C, 0xC4D), "Kannada": (0xC95, 0xCB9, 0xCBE, 0xCCC, 0xCCD),
               "Malayalam": (0xD15, 0xD39, 0xD3E, 0xD4C, 0xD4D), "Sinhala": (0xD9A, 0xDC6, 0xDCF, 0xDDF, 0xDCA)}
CJK_COMMON = ("的一是不了人我在有他这中大来上国个到说们为子和你地出道也时年得就那要下以生会自着去之过家学对可她里后小"
              "么心多天而能好都然没日于起还发成事只作当想看文无开手十用主行方又如前所本见经头面公同三已老从动两长知民样"
              "现分将外但身些与高意进把法此实回二理美点月明其种声全工己话儿者向情部正名定女问力机给等几很业最间新什打便"
              "位因重被走电四第门相次东政海口使教西再平真听世气信北少关并内加化由却代军产入先山五太水万市眼体别处总才场"
              "师书比住员九笑性通目华报立马命张活难神数件安表原车白应路期叫死常提感金何更反合放做系计或司利受光王果亲界"
              "及今京务制解各任至清物台象记边共风战干接它许八特觉望直服毛林题建南度统色字请交爱让认算论百吃义科怎元社术"
              "结六功指思非流每青管夫连远资队跟带花快条院变联言权往展该领传近留红治决周保达办运武半候七必城父强步完革深"
              "区即求品士转量空甚众技轻程告江语英基派满式李息写呢识极令黄德收脸钱党倒未持取设始版双历越史商千片容研像找"
              "友孩站广改议形委早房音火际则首单据导影失拿网香似斯专石若兵弟谁校读志飞观争究包组造落视济喜离虽坏兴切营急"
              "银店街駅号線禁止注意左右後営業時間町村県丁目番地東京都大阪府福岡札幌横浜神奈川駐車場入口出口臺灣區號樓與門"
              "東車開關為個們這說來時會國長電話醫院藥局銀行郵便局学校病院公園橋寺神社料理食堂酒屋茶喫煙歯科内科眼科薬")
KANA = "".join(chr(c) for c in list(range(0x3041, 0x3094)) + list(range(0x30A1, 0x30FB)))


def _rand_word(rng, alphabet, weights=None, n=(3, 9)):
    k = rng.randint(n[0], n[1] + 1)
    idx = rng.choice(len(alphabet), k, p=weights)
    return "".join(alphabet[i] for i in idx)


def _latin_like(rng, letters, extra, p_extra):
    words = []
    for _ in range(rng.randint(1, 4)):
        w = "".join(rng.choice(list(extra)) if rng.rand() < p_extra else letters[i]
                    for i in rng.choice(len(letters), rng.randint(2, 10)))
        words.append(w)
    t = " ".join(words)
    r = rng.rand()
    return t.upper() if r < 0.45 else (t.title() if r < 0.75 else t)


def synth_text(rng, script):
    """A random line of text in a script (letter frequencies roughly as in running text)."""
    if script == "Latin":
        return _latin_like(rng, "eeeeettttaaaooooiiinnnssshhrrrdddllcuummwffggyppbvkjxqz", "éáíóúñüöäçãõàèêčšžłąęőűğşıăț", 0.04)
    if script == "Cyrillic":
        return _latin_like(rng, "оооееаааииннтттсссрррввллккммддппууяыьгзбчйхжшюцщэф", "іїєґәғқңөұүһјљњћџё", 0.04)
    if script == "Greek":
        t = _latin_like(rng, "ααοοιιεεττσνηυρπκμλωδγχθφβξζψ", "άέήίόύώ", 0.06)
        return " ".join(w[:-1] + "ς" if w[-1:] == "σ" else w for w in t.split(" "))
    if script == "Digits":
        r = rng.rand()
        if r < 0.4:
            return "".join(str(d) for d in rng.randint(0, 10, rng.randint(3, 4))) + rng.choice(["-", " ", ""]) + \
                "".join(str(d) for d in rng.randint(0, 10, rng.randint(3, 5)))
        if r < 0.7:
            return str(rng.choice([20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 5, 10, 15, 25]))
        return str(rng.randint(1, 3000)) + rng.choice(["", " km", " m", ".000", "/2", "A"])
    if script == "CJK":
        n = rng.randint(2, 9)
        if rng.rand() < 0.5:  # Japanese: kanji + kana
            return "".join(CJK_COMMON[rng.randint(len(CJK_COMMON))] if rng.rand() < 0.5 else KANA[rng.randint(len(KANA))]
                           for _ in range(n))
        return "".join(CJK_COMMON[rng.randint(len(CJK_COMMON))] for _ in range(n))
    if script == "Hangul":
        words = []
        for _ in range(rng.randint(1, 4)):
            words.append("".join(chr(0xAC00 + (rng.randint(19) * 21 + rng.randint(21)) * 28 +
                                     (0 if rng.rand() < 0.55 else rng.randint(1, 28))) for _ in range(rng.randint(2, 6))))
        return " ".join(words)
    if script == "Thai":
        cons = [chr(c) for c in range(0xE01, 0xE2F)]
        out = []
        for _ in range(rng.randint(3, 12)):
            s = ""
            if rng.rand() < 0.25:
                s += chr(rng.choice([0xE40, 0xE41, 0xE42, 0xE43, 0xE44]))
            s += cons[rng.randint(len(cons))]
            r = rng.rand()
            if r < 0.35:
                s += chr(rng.choice([0xE31, 0xE34, 0xE35, 0xE36, 0xE37, 0xE38, 0xE39]))
            if rng.rand() < 0.3:
                s += chr(rng.choice([0xE48, 0xE49, 0xE4A, 0xE4B]))
            if rng.rand() < 0.25:
                s += chr(rng.choice([0xE30, 0xE32, 0xE33]))
            if rng.rand() < 0.4:
                s += cons[rng.randint(len(cons))]
            out.append(s)
            if rng.rand() < 0.12:
                out.append(" ")
        return "".join(out).strip()
    if script == "Hebrew":
        letters = [chr(c) for c in range(0x5D0, 0x5EB)]
        return " ".join(_rand_word(rng, letters, n=(2, 7)) for _ in range(rng.randint(1, 4)))
    if script == "Arabic":
        letters = [chr(c) for c in list(range(0x627, 0x63B)) + list(range(0x641, 0x64B))]
        return " ".join(_rand_word(rng, letters, n=(2, 7)) for _ in range(rng.randint(1, 4)))
    if script in INDIC_BLOCK:
        c0, c1, v0, v1, vir = INDIC_BLOCK[script]
        cons = [chr(c) for c in range(c0, c1 + 1) if unicodedata.name(chr(c), "")]
        vow = [chr(c) for c in range(v0, v1 + 1) if unicodedata.name(chr(c), "")]
        words = []
        for _ in range(rng.randint(1, 4)):
            w = ""
            for _ in range(rng.randint(2, 5)):
                w += cons[rng.randint(len(cons))]
                if rng.rand() < 0.1:
                    w += chr(vir) + cons[rng.randint(len(cons))]
                if rng.rand() < 0.55:
                    w += vow[rng.randint(len(vow))]
            words.append(w)
        return " ".join(words)
    raise ValueError(script)


_FONT_FILES, _FONT_CACHE, _MISSING = None, {}, {}


def font_files():
    global _FONT_FILES
    if _FONT_FILES is None:
        _FONT_FILES = {}
        for d in FONT_DIRS:
            for root, _, files in os.walk(d):
                for f in files:
                    _FONT_FILES.setdefault(f, os.path.join(root, f))
    return _FONT_FILES


def get_font(name, size):
    key = (name, size)
    if key not in _FONT_CACHE:
        from PIL import ImageFont
        _FONT_CACHE[key] = ImageFont.truetype(font_files()[name], size, layout_engine=ImageFont.Layout.RAQM)
    return _FONT_CACHE[key]


def covers(font_name, text):
    """True when the font has a glyph for every letter of text (missing glyphs render as the
    .notdef box: compared with the mask of a private-use code point)."""
    f = get_font(font_name, 32)
    if font_name not in _MISSING:
        _MISSING[font_name] = np.asarray(f.getmask(""))
    miss = _MISSING[font_name]
    for ch in set(text) - {" "}:
        if unicodedata.category(ch).startswith("M"):
            continue
        m = np.asarray(f.getmask(ch))
        if m.size == 0 or (m.shape == miss.shape and np.array_equal(m, miss)):
            return False
    return True


def fonts_for(script):
    return [f for f in FONTS[script].split() if f in font_files()]


def render_text(rng, script, text, font_name, size, vertical=False):
    """Ink mask (L image) of the text."""
    from PIL import ImageDraw
    f = get_font(font_name, size)
    direction = "rtl" if script in ("Hebrew", "Arabic") else None
    if vertical:
        chars = [c for c in text if c != " "]
        cell = int(size * 1.15)
        img = Image.new("L", (cell + 4, cell * len(chars) + 4), 0)
        dr = ImageDraw.Draw(img)
        for i, c in enumerate(chars):
            dr.text((2, 2 + i * cell), c, font=f, fill=255)
        return img
    x0, y0, x1, y1 = f.getbbox(text, direction=direction)
    img = Image.new("L", (x1 - x0 + 8, y1 - y0 + 8), 0)
    ImageDraw.Draw(img).text((4 - x0, 4 - y0), text, font=f, fill=255, direction=direction)
    return img


def synth_image(rng, script, backgrounds, scale=1.0):
    """A rendered sign with a line (or two) of the script in a real background crop -> (RGB image,
    text box (x0, y0, x1, y1) in image pixels)."""
    from PIL import ImageFilter
    sub = script
    if script == "Indic":
        names = list(INDIC_WEIGHT)
        sub = names[rng.choice(len(names), p=np.array([INDIC_WEIGHT[n] for n in names]))]
    fonts = fonts_for(sub)
    for _ in range(20):
        text = synth_text(rng, sub)
        fn = fonts[rng.randint(len(fonts))]
        if covers(fn, text):
            break
    vertical = script in ("CJK", "Hangul") and rng.rand() < 0.2
    m = render_text(rng, script, text, fn, 48, vertical)
    ink_h = m.size[1] - 8.0
    if not vertical and rng.rand() < 0.3:  # a second line of the same script
        for _ in range(10):
            t2 = synth_text(rng, sub)
            if covers(fn, t2):
                m2 = render_text(rng, script, t2, fn, 48)
                gap = int(48 * rng.uniform(0.1, 0.6))
                mm = Image.new("L", (max(m.size[0], m2.size[0]), m.size[1] + m2.size[1] + gap), 0)
                mm.paste(m, ((mm.size[0] - m.size[0]) // 2, 0))
                mm.paste(m2, ((mm.size[0] - m2.size[0]) // 2, m.size[1] + gap))
                m = mm
                break
    if script in ("Latin", "Cyrillic", "Greek", "Digits") and rng.rand() < 0.2:  # condensed / wide sign fonts
        m = m.resize((max(int(m.size[0] * rng.uniform(0.7, 1.3)), 4), m.size[1]), Image.BILINEAR)
    pad = int(48 * rng.uniform(0.2, 0.9))
    M = np.zeros((m.size[1] + 2 * pad, m.size[0] + 2 * pad), np.float32)
    M[pad:pad + m.size[1], pad:pad + m.size[0]] = np.asarray(m, np.float32) / 255.0
    while True:
        cb, ct = rng.randint(0, 256, 3).astype(np.float32), rng.randint(0, 256, 3).astype(np.float32)
        lum = np.array([0.299, 0.587, 0.114])
        if abs(lum @ cb - lum @ ct) > rng.uniform(60, 140):
            break
    board = cb[None, None, :] * (1 - M[..., None]) + ct[None, None, :] * M[..., None]
    if rng.rand() < 0.3:  # sign frame
        b = max(2, pad // 4)
        board[:b], board[-b:], board[:, :b], board[:, -b:] = ct, ct, ct, ct
    # scale: the line's ink height to the glyph sizes of the placements (crop pixels at 0.03 deg/px)
    target = math.exp(rng.uniform(math.log(13), math.log(60))) * scale
    s = target / (48.0 if vertical else max(ink_h, 1.0))
    bw, bh = max(int(board.shape[1] * s), 4), max(int(board.shape[0] * s), 4)
    bimg = Image.fromarray(np.clip(board, 0, 255).astype(np.uint8)).resize((bw, bh), Image.LANCZOS)
    ang = rng.uniform(-4, 4)
    sh = rng.uniform(-0.12, 0.12)
    a = math.radians(ang)
    # affine (rotation + shear) around the board centre, expanded canvas
    cx, cy = bw / 2.0, bh / 2.0
    A = np.array([[math.cos(a), -math.sin(a) + sh], [math.sin(a), math.cos(a)]])
    corners = np.array([[0, 0], [bw, 0], [0, bh], [bw, bh]], float) - [cx, cy]
    cc = corners @ A.T
    W2, H2 = int(np.ceil(cc[:, 0].ptp())) + 2, int(np.ceil(cc[:, 1].ptp())) + 2
    Ai = np.linalg.inv(A)
    off = np.array([cx, cy]) - Ai @ np.array([W2 / 2.0, H2 / 2.0])
    data = (Ai[0, 0], Ai[0, 1], off[0], Ai[1, 0], Ai[1, 1], off[1])
    warped = bimg.transform((W2, H2), Image.AFFINE, data, Image.BILINEAR)
    alpha = Image.new("L", (bw, bh), 255).transform((W2, H2), Image.AFFINE, data, Image.BILINEAR)
    # text box: transform the ink box of the board
    tb = (np.array([[pad, pad], [pad + m.size[0], pad], [pad, pad + m.size[1]], [pad + m.size[0], pad + m.size[1]]],
                   float) * s - [cx, cy]) @ A.T + [W2 / 2.0, H2 / 2.0]
    bg = backgrounds[rng.randint(len(backgrounds))]
    bgi = Image.open(os.path.join(CROPS, bg + ".jpg")).convert("RGB")
    cw, ch = min(max(int(W2 * rng.uniform(1.6, 3.0)), W2 + 20), 900), min(max(int(H2 * rng.uniform(2.0, 4.0)), H2 + 20), 600)
    cw, ch = min(cw, bgi.size[0]), min(ch, bgi.size[1])
    if W2 + 4 > cw or H2 + 4 > ch:
        return None, None
    bx, by = rng.randint(0, bgi.size[0] - cw + 1), rng.randint(0, bgi.size[1] - ch + 1)
    canvas = bgi.crop((bx, by, bx + cw, by + ch))
    px, py = rng.randint(2, cw - W2 - 1), rng.randint(2, ch - H2 - 1)
    canvas.paste(warped, (px, py), alpha)
    r = rng.uniform(0, 1.2)
    if r > 0.2:
        canvas = canvas.filter(ImageFilter.GaussianBlur(r))
    arr = np.asarray(canvas, np.float32) + rng.randn(ch, cw, 1) * rng.uniform(0, 6)
    canvas = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    canvas.save(buf, "JPEG", quality=int(rng.uniform(40, 92)))
    canvas = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    box = (tb[:, 0].min() + px, tb[:, 1].min() + py, tb[:, 0].max() + px, tb[:, 1].max() + py)
    return canvas, box


def _synth_one(job):
    seed, script, backgrounds, scale = job
    rng = np.random.RandomState(seed)
    img, box = synth_image(rng, script, backgrounds, scale)
    if img is None:
        return seed, None
    o = sd.detect_lines(img, codebook=_CODEBOOK)
    keep = []
    for i, b in enumerate(o["boxes"]):
        ix = min(b[2], box[2]) - max(b[0], box[0]) + 1
        iy = min(b[3], box[3]) - max(b[1], box[1]) + 1
        if ix > 0 and iy > 0 and ix * iy >= 0.8 * (b[2] - b[0] + 1) * (b[3] - b[1] + 1):
            keep.append(i)
    keep = np.array(keep, int)
    wd = [o["words"][i] for i in keep] if "words" in o else []
    return seed, (o["feats"][keep], o["script"][keep], o["ncomp"][keep],
                  np.concatenate(wd).astype(np.int16) if wd else np.zeros(0, np.int16))


def cmd_synth(args):
    """Synthetic labelled text lines for every script -> scratch/script_spike/synth_lines.npz."""
    from multiprocessing import Pool
    bgs = [p["key"] for p in all_placements() if p["type"] not in TEXT_TYPES and p["split"] == "train"]
    if args.show:
        os.makedirs(os.path.join(SPIKE, "debug"), exist_ok=True)
        from PIL import ImageDraw
        rng = np.random.RandomState(args.seed)
        tiles = []
        for s in SYN_CLASSES:
            for _ in range(3):
                img, box = synth_image(rng, s, bgs)
                if img is None:
                    continue
                ImageDraw.Draw(img).rectangle([int(v) for v in box], outline=(255, 0, 0))
                img.thumbnail((320, 200))
                tiles.append(img)
        sheet = Image.new("RGB", (320 * 3, 200 * ((len(tiles) + 2) // 3)))
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % 3) * 320, (i // 3) * 200))
        sheet.save(os.path.join(SPIKE, "debug", "synth.jpg"), quality=88)
        return
    scale = DEG_PER_PX / args.res  # glyph sizes of the placements seen at args.res deg/px
    jobs = [(args.seed * 1000003 + i, SYN_CLASSES[i % len(SYN_CLASSES)], bgs, scale) for i in range(args.n)]
    F, S, N, Y, G, Wd = [], [], [], [], [], []
    t0 = time.time()
    with Pool(min(args.workers, 4), initializer=_lines_init, initargs=("tiles", True), maxtasksperchild=500) as pool:
        for k, (seed, out) in enumerate(pool.imap(_synth_one, jobs, chunksize=8)):
            if (k + 1) % 1000 == 0:
                print("%d/%d %.0fs" % (k + 1, len(jobs), time.time() - t0), flush=True)
            if out is None or not len(out[0]):
                continue
            f, s, n, wd = out
            Wd.append(wd)
            F.append(f.astype(np.float32))
            S.append(s.astype(np.float16))
            N.append(n)
            Y.append(np.full(len(f), SYN_CLASSES.index(jobs[k][1]), np.int32))
            G.append(np.full(len(f), k, np.int32))
    np.savez(synth_path(args.res), feats=np.concatenate(F), script=np.concatenate(S), ncomp=np.concatenate(N),
             y=np.concatenate(Y), sample=np.concatenate(G), words=np.concatenate(Wd))
    y = np.concatenate(Y)
    print("synthetic lines", len(y), {c: int((y == i).sum()) for i, c in enumerate(SYN_CLASSES)})


MODEL = os.path.join(SPIKE, "script_model.npz")


def fit_final(path, plc_by_key, cfg, rule):
    """Text scorer + line classifier fitted on all train + calib crops, with the crop rule
    'mix_q<q>_pw<w>' -> scratch/script_spike/script_model.npz (engine.script_detect.ScriptReader)."""
    L = load_lines(path, plc_by_key)
    dev = np.isin(L["split"], ["train", "calib"])
    m, thr, _, _, sm = fit_models(L, dev, cfg)
    q, pw = [float(v[1:] if v[0] == "q" else v[2:]) for v in rule[4:].split("_")]
    cnt = np.bincount(L["lab"][dev & L["is_script"]], minlength=len(CLASSES)) + 1.0
    np.savez(MODEL, text_c=m["c"], text_s=m["s"], text_W=m["W"], text_thr=thr, script_cf=sm["cf"],
             script_sf=sm["sf"], script_cs=sm["cs"], script_ss=sm["ss"], script_W=sm["W"], feats=sm["feats"],
             q=q, pw=pw, logprior=np.log(cnt / cnt.sum()), codebook=np.load(CODEBOOK)["centres"],
             wordbook=np.load(WORDBOOK)["centres"], classes=np.array(ALL_CLASSES), features=np.array(sd.FEATURES))
    print("saved", MODEL, "text threshold %.2f" % thr)


def cmd_frames(args):
    """ScriptReader on whole live-size frames: the 28 deg crops (zoom <= 3 placements) resized to
    args.width px (1112 px = the user's canvas: ~0.025 deg/px, a zoom-3 frame), read without the
    detection window and with it.  -> runtime per frame, rate of frames with a line above the
    saved threshold (control crops: false alarms)."""
    reader = sd.ScriptReader(MODEL)
    plc = [p for p in all_placements() if view_size(p["zoom"])[0] >= MAX_T - 1e-9]
    rng = np.random.RandomState(args.seed)
    out = {}
    for name, sub in (("control", [p for p in plc if p["type"] not in TEXT_TYPES]),
                      ("text", [p for p in plc if p["type"] in SCRIPT_TYPES])):
        sub = [sub[i] for i in rng.choice(len(sub), min(args.n, len(sub)), replace=False)]
        fr, win, times = [], [], []
        for p in sub:
            img = crop_image(p, DEG_PER_PX, "tiles")
            img = img.resize((args.width, int(round(args.width * img.size[1] / float(img.size[0])))), Image.BICUBIC)
            t0 = time.time()
            size = img.size
            fr.append(reader.read(img) is not None)
            times.append(time.time() - t0)
            w = WINDOW_T / MAX_T
            win.append(reader.read(img, window=(w, w)) is not None)
        out[name] = {"n": len(sub), "frame_line_rate": float(np.mean(fr)), "window_line_rate": float(np.mean(win)),
                     "time_median_s": float(np.median(times)), "time_p90_s": float(np.percentile(times, 90)),
                     "time_max_s": float(np.max(times))}
        print(name, {k: round(v, 3) for k, v in out[name].items()}, flush=True)
    out["size"] = list(size)
    out["blas_threads"] = os.environ.get("OPENBLAS_NUM_THREADS")
    with open(os.path.join(SPIKE, "frames_%d.json" % args.width), "w") as f:
        json.dump(out, f, indent=1)


def cmd_show(args):
    """Debug overlays: the best text lines of each crop with their line class (ScriptReader)."""
    from PIL import ImageDraw
    reader = sd.ScriptReader(MODEL)
    plc_by_key = {p["key"]: p for p in json.load(open(PLACEMENTS))}
    os.makedirs(os.path.join(SPIKE, "debug"), exist_ok=True)
    _lines_init(args.source, False)
    for key in args.keys:
        p = plc_by_key[key]
        img = crop_image(p, args.res, args.source, args.upsample)
        fr = window_frac(p)
        t0 = time.time()
        r = reader.read(img, window=(fr, fr), k=args.top)
        dt = time.time() - t0
        if r is None:
            print(key, p["script"], "no text line")
            continue
        dr = ImageDraw.Draw(img)
        for box, ts, cls, pr in r["lines"]:
            dr.rectangle(list(box), outline=(0, 255, 0), width=2)
            dr.text((box[0], max(box[1] - 11, 0)), "%.1f %s %.2f" % (ts, cls[:5], pr), fill=(0, 255, 0))
        img.save(os.path.join(SPIKE, "debug", "%s_%g.jpg" % (key, args.res)), quality=90)
        best = max(r["posterior"], key=r["posterior"].get)
        print(key, p["script"], p["title"], "->", best, "%.2f" % r["posterior"][best], "%.2fs" % dt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    a = sub.add_parser("fetch")
    a.add_argument("--types", default="language,road-signs,license-plates")
    a.add_argument("--workers", type=int, default=4)
    a.add_argument("--limit", type=int, default=0)
    a.add_argument("--max-per-type", type=int, default=0)
    a.add_argument("--refresh", action="store_true", help="re-read pano_clues.json")
    sub.add_parser("recrop")
    a = sub.add_parser("synth")
    a.add_argument("--n", type=int, default=15000, help="rendered signs (equal per class)")
    a.add_argument("--seed", type=int, default=1)
    a.add_argument("--workers", type=int, default=4)
    a.add_argument("--show", action="store_true", help="only a contact sheet of examples")
    a.add_argument("--res", type=float, default=DEG_PER_PX, help="glyph sizes as seen at this resolution")
    a = sub.add_parser("codebook")
    a.add_argument("--n", type=int, default=600, help="train-split text crops to sample")
    a.add_argument("--k", type=int, default=64)
    a.add_argument("--max-glyphs", type=int, default=150000)
    a.add_argument("--words", action="store_true", help="the larger word codebook (e.g. --k 512)")
    a.add_argument("--workers", type=int, default=4)
    for name in ("lines", "eval", "show"):
        a = sub.add_parser(name)
        a.add_argument("--res", type=float, default=DEG_PER_PX)
        a.add_argument("--source", default="tiles", choices=["tiles", "equirect"])
        a.add_argument("--upsample", type=float, default=1.0)
        if name == "lines":
            a.add_argument("--workers", type=int, default=4)
            a.add_argument("--limit", type=int, default=0)
        elif name == "eval":
            a.add_argument("--split", default="cv", choices=["cv", "calib", "test"])
            a.add_argument("--cfg", nargs="*", help="key=value overrides of %s" % DEFAULT_CFG)
            a.add_argument("--tag", default="")
            a.add_argument("--save", action="store_true", help="also fit on all train + calib crops and save")
            a.add_argument("--rule", default="mix_q0.3_pw1", help="crop rule of the saved model")
        else:
            a.add_argument("keys", nargs="+")
            a.add_argument("--top", type=int, default=3)
    a = sub.add_parser("frames")
    a.add_argument("--width", type=int, default=1112)
    a.add_argument("--n", type=int, default=40, help="crops per group (control / text)")
    a.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cmds = {"fetch": cmd_fetch, "recrop": cmd_recrop, "codebook": cmd_codebook, "lines": cmd_lines, "synth": cmd_synth,
            "eval": cmd_eval, "show": cmd_show, "frames": cmd_frames}
    if args.cmd in cmds:
        cmds[args.cmd](args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
