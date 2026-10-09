#!/usr/bin/env python3
"""
Build data/model/clue_detectors.npz: classical sliding-window detectors of GeoGuessr clue cards
(engine/clue_detect.py) trained on GeoGuessr's own clue placements, calibrated as card-presence,
country and region evidence.

  python3 tools/build_clue_detectors.py cells [--workers 4] [--extra 20]
        cell-feature cache (scratch/cards/cells/) of the labelled panoramas, every CALIB / TEST round
        and up to --extra unlabelled training panoramas per country
  python3 tools/build_clue_detectors.py fit [--k 200] [--shrink 0.1] [--bg 3000]
        background statistics (the --bg training panoramas with the smallest hashes) and LDA
        detectors (all placements / fold A / fold B; every detector has placements in both folds)
  python3 tools/build_clue_detectors.py score [--workers 4]
        best window of every detector on every cached panorama (training panoramas with the
        detectors of the other fold, everything else with the full bank)
  python3 tools/build_clue_detectors.py calibrate [--test]
        presence / direction / country / region models; the hint order (detected cards first) and
        the shown directions stay off unless they beat the plain ranking on CALIB (bootstrap lower
        bound > 0; >= 50% of the shown directions on the card); report (TEST only with --test)
  python3 tools/build_clue_detectors.py all [--test]
then python3 tools/train_model.py --calibrate-cards --save (the weight of the card evidence).
Everything after `cells` is deterministic given the cache (scratch/cards/, ~1.4 GB; cells/ is only
needed to rebuild).

Labels (data/calibration/pano_clues.json, read once by `cells` with a retry and frozen in
scratch/cards/clues_used.json) are split as in tools/build_clue_index.py: detectors, presence and
country models are fitted only on public duel rounds (+ unlabelled training panoramas for the country
model), the user's CALIB rounds calibrate the hint threshold, the region exponent and
(tools/train_model.py --calibrate-cards) the country weight; TEST rounds are only scored.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from build_clue_index import bootstrap, card_sets, labelled_rounds  # noqa: E402
from calibrate import DATASET, load_records, split_of  # noqa: E402
from engine import clue_detect as cd  # noqa: E402

CLUES = os.path.join(ROOT, "data", "calibration", "pano_clues.json")
CACHE = os.path.join(ROOT, "scratch", "cards")
CELLS = os.path.join(CACHE, "cells")
MIN_PLACEMENTS = 3   # own detector per (card, channel); rarer cards share a (country, type, channel) detector
MIN_FOLD = 1         # training placements a detector needs in each cross-fitting fold (fold_of)
ROW_MARGIN = 4.0     # deg around the placement pitches searched by a detector
NEIGH = np.array([[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]])  # positive windows around a placement
DIR_BAR = 0.5        # share of shown directions that must point at the card (CALIB) to show directions at all
DIR_MIN_N = 20       # ... measured on at least this many shown directions


def load_clues(path, tries=20):
    """The clue file is rewritten by a running crawler: retry on a partial file."""
    for i in range(tries):
        try:
            with open(path) as fh:
                return json.load(fh)
        except ValueError:
            time.sleep(1.5)
    raise RuntimeError("could not parse %s" % path)


def fold_of(pid):
    return int(hashlib.md5(pid.encode()).hexdigest(), 16) % 2


def pano_path(pid):
    return os.path.join(DATASET, "panos", pid + ".jpg")


def _file_hash(path):
    """Hash of the arrays of an npz (not of the file: zip entries carry their write time)."""
    h = hashlib.sha1()
    with np.load(path) as z:
        for k in sorted(z.files):
            h.update(k.encode())
            h.update(np.ascontiguousarray(z[k]).tobytes())
    return h.hexdigest()[:12]


def frozen_labels():
    clues = json.load(open(os.path.join(CACHE, "clues_used.json")))
    rounds = labelled_rounds(clues)
    return clues, rounds


def placements(clues, rounds):
    """[(pid, split, country, stem, placement)] of the country cards placed on labelled panoramas."""
    from engine.hints import clue_stem
    out = []
    for pid, r in rounds.items():
        for p in clues[pid]:
            if (p.get("countryCode") or "").upper() != r["label"] or abs(float(p.get("pitch", 0.0))) > 30:
                continue
            out.append((pid, r["split"], r["label"], clue_stem(p.get("title")) or p["id"].lower(), p))
    return out, card_sets(clues, rounds)


# ----------------------------------------------------------------------------- cell cache
LAYOUT = [(cd.grid(c)["n_rows"], cd.grid(c)["n_az"], cd.CF) for c in cd.CHANNELS]
SIZES = [int(np.prod(s)) for s in LAYOUT]
OFFS = np.cumsum([0] + SIZES)


class CellCache:
    """Quantised cell grids of full panoramas: shards scratch/cards/cells/shard_NNN.npy (n, bytes)."""

    def __init__(self, path=CELLS):
        self.path = path
        p = os.path.join(path, "index.json")
        self.index = json.load(open(p)) if os.path.exists(p) else {"layout": LAYOUT, "ids": {}, "shards": 0}
        if [list(x) for x in self.index["layout"]] != [list(x) for x in LAYOUT]:
            raise RuntimeError("cell cache layout differs from engine/clue_detect.py - delete %s" % path)
        self._mm = {}

    def __contains__(self, pid):
        return pid in self.index["ids"]

    def ids(self):
        return sorted(self.index["ids"])

    def add_shard(self, ids, rows):
        os.makedirs(self.path, exist_ok=True)
        s = self.index["shards"]
        np.save(os.path.join(self.path, "shard_%03d.npy" % s), np.asarray(rows, np.uint8))
        for k, pid in enumerate(ids):
            self.index["ids"][pid] = [s, k]
        self.index["shards"] = s + 1
        tmp = os.path.join(self.path, "index.json.tmp")
        json.dump(self.index, open(tmp, "w"))
        os.replace(tmp, os.path.join(self.path, "index.json"))

    def get(self, pid):
        """[(cells uint8, None)] of every channel."""
        s, k = self.index["ids"][pid]
        if s not in self._mm:
            self._mm[s] = np.load(os.path.join(self.path, "shard_%03d.npy" % s), mmap_mode="r")
        row = np.asarray(self._mm[s][k])
        return [(row[OFFS[c]:OFFS[c + 1]].reshape(LAYOUT[c]), None) for c in range(len(LAYOUT))]


def _cells_job(job):
    pid, heading = job
    from engine.panorama import SphericalImage
    try:
        sph = SphericalImage.from_equirect(pano_path(pid), heading=heading)
    except Exception as e:
        sys.stderr.write("%s: %r\n" % (pid, e))
        return pid, None
    return pid, np.concatenate([q.ravel() for q, _ in cd.sphere_cells(sph)])


def cache_set(recs, rounds, extra, have=(), seed=2):
    """Panoramas to cache: labelled ones, every CALIB / TEST round, up to `extra` unlabelled training
    panoramas per country (those already cached count)."""
    ids = [p for p in rounds if p in recs]
    ids += [p for p, r in recs.items() if split_of(r) in ("calib", "test")]
    by_cc = defaultdict(list)
    for p, r in recs.items():
        if split_of(r) == "train" and p not in rounds:
            by_cc[r["label"]].append(p)
    for cc in sorted(by_cc):  # stable choice (cached first, then the smallest hashes)
        v = [p for p in by_cc[cc] if p in have]
        if len(v) < extra:
            v += sorted((p for p in by_cc[cc] if p not in have),
                        key=lambda p: hashlib.md5((str(seed) + p).encode()).hexdigest())[:extra - len(v)]
        ids += v
    return sorted(set(ids))


def cmd_cells(args):
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.makedirs(CACHE, exist_ok=True)
    clues = load_clues(args.clues)
    json.dump(clues, open(os.path.join(CACHE, "clues_used.json"), "w"))
    rounds = labelled_rounds(clues)
    recs = {r["pano_id"]: r for r in load_records()}
    cache = CellCache()
    ids = cache_set(recs, rounds, args.extra, have=cache.index["ids"])
    todo = [p for p in ids if p not in cache and os.path.exists(pano_path(p))]
    print("%d labelled panoramas (%s); %d to cache, %d cached" % (
        len(rounds), dict(Counter(r["split"] for r in rounds.values())), len(todo), len(cache.index["ids"])),
          flush=True)
    t0 = time.time()
    buf_ids, buf = [], []
    with Pool(args.workers) as pool:
        for i, (pid, row) in enumerate(pool.imap_unordered(_cells_job, [(p, recs[p].get("heading")) for p in todo],
                                                           chunksize=8)):
            if row is not None:
                buf_ids.append(pid)
                buf.append(row)
            if len(buf) >= 2000:
                cache.add_shard(buf_ids, buf)
                buf_ids, buf = [], []
            if (i + 1) % 1000 == 0:
                print("  %d/%d  %.1f pano/s" % (i + 1, len(todo), (i + 1) / (time.time() - t0)), flush=True)
    if buf:
        cache.add_shard(buf_ids, buf)
    print("cached %d panoramas in %.0fs (%d in total)" % (len(todo), time.time() - t0, len(cache.index["ids"])))


# ----------------------------------------------------------------------------- fit
def enough(ps):
    """>= MIN_PLACEMENTS placements and >= MIN_FOLD in each fold: the out-of-fold template of either fold
    is then never empty (an empty one would give its held-out panoramas a constant score)."""
    f = Counter(fold_of(pid) for pid, _ in ps)
    return len(ps) >= MIN_PLACEMENTS and min(f[0], f[1]) >= MIN_FOLD


def define_detectors(pl):
    """Detectors from the training placements: one per (card, channel) with enough() placements there; a
    card without one but with enough() placements in all gets one in its most common channel; the
    remaining cards share a (country, type, channel) detector when the group has enough()."""
    by_card = defaultdict(list)
    for pid, split, cc, stem, p in pl:
        if split == "train":
            by_card[(cc, stem)].append((pid, p))
    dets, members = [], []
    groups = defaultdict(list)
    for (cc, stem), ps in sorted(by_card.items()):
        by_ch = defaultdict(list)
        for pid, p in ps:
            by_ch[cd.channel_of(p.get("zoom", 2.0))].append((pid, p))
        own = [c for c in sorted(by_ch) if enough(by_ch[c])]
        if not own and enough(ps):
            own = [max(by_ch, key=lambda c: len(by_ch[c]))]
            by_ch = {own[0]: ps}
        for c in own:
            dets.append({"country": cc, "cards": [stem], "type": ps[0][1].get("type"), "group": False,
                         "channel": int(c)})
            members.append(by_ch[c])
        if not own:
            c = max(by_ch, key=lambda c: len(by_ch[c]))
            groups[(cc, ps[0][1].get("type") or "misc", c)].append((stem, ps))
    for (cc, t, c), items in sorted(groups.items()):
        ps = [x for _, q in items for x in q]
        if not enough(ps):
            continue
        dets.append({"country": cc, "cards": sorted(s for s, _ in items), "type": t, "group": True, "channel": int(c)})
        members.append(ps)
    for d, ps in zip(dets, members):
        g = cd.grid(cd.CHANNELS[d["channel"]])
        rows = [cd.window_at(g, 0.0, float(p.get("pitch", 0.0)))[0] for _, p in ps]
        m = int(np.ceil(ROW_MARGIN / g["cell"]))
        d["rows"] = [int(np.clip(min(rows) - m, 0, g["n_wrows"] - 1)), int(np.clip(max(rows) + m, 0, g["n_wrows"] - 1))]
        d["n"] = len(ps)
        d["n_fold"] = [sum(fold_of(pid) == f for pid, _ in ps) for f in (0, 1)]
    return dets, members


def background_stats(cache, pids, per_pano, seed=0):
    """Per channel: window-row means mu (n_wrows, DIM) and the pooled within-row covariance S."""
    rng = np.random.RandomState(seed)
    out = []
    for c, ch in enumerate(cd.CHANNELS):
        g = cd.grid(ch)
        R, A = g["n_wrows"], g["n_az"]
        s = np.zeros((R, cd.DIM))
        n = np.zeros(R)
        Q = np.zeros((cd.DIM, cd.DIM))
        for pid in pids:
            X = cd.windows(cache.get(pid)[c][0]).astype(np.float64)
            k = rng.choice(len(X), min(per_pano, len(X)), replace=False)
            r = k // A
            np.add.at(s, r, X[k])
            n += np.bincount(r, minlength=R)
            Q += X[k].T @ X[k]
        mu = s / np.maximum(n, 1)[:, None]
        S = (Q - (s.T / np.maximum(n, 1)) @ s) / max(n.sum() - R, 1)
        out.append((mu, S, int(n.sum())))
    return out


def whitening(S, k, shrink):
    Ss = (1 - shrink) * S + shrink * np.diag(np.diag(S))
    lam, U = np.linalg.eigh(Ss)
    o = np.argsort(-lam)[:k]
    return (U[:, o] / np.sqrt(np.maximum(lam[o], 1e-9))).T  # (k, DIM)


def placement_vector(cells, g, mu, yaw, pitch):
    """Weighted mean deviation from the row means of the windows around a placement direction."""
    i, j = cd.window_at(g, yaw, pitch)
    i = min(max(i, 1), g["n_wrows"] - 2)
    rows = np.arange(i - 1, i + 2)
    X = cd.windows(cells, rows).reshape(3, g["n_az"], cd.DIM)
    cols = (np.arange(j - 1, j + 2)) % g["n_az"]
    V = X[:, cols].astype(np.float64) - mu[rows][:, None, :]
    return (NEIGH[..., None] * V).sum((0, 1)) / NEIGH.sum()


def cmd_fit(args):
    clues, rounds = frozen_labels()
    pl, _ = placements(clues, rounds)
    recs = {r["pano_id"]: r for r in load_records()}
    cache = CellCache()
    pl = [x for x in pl if x[0] in cache]
    dets, members = define_detectors(pl)
    print("%d training placements -> %d detectors (%d card, %d group) covering %d cards" % (
        sum(s == "train" for _, s, _, _, _ in pl), len(dets), sum(not d["group"] for d in dets),
        sum(d["group"] for d in dets), len({(d["country"], s) for d in dets for s in d["cards"]})), flush=True)
    tr = [p for p in cache.ids() if split_of(recs[p]) == "train"]
    # background panoramas: the smallest hashes (reproducible from the cache contents alone)
    bg = sorted(sorted(tr, key=lambda p: hashlib.md5(("bg" + p).encode()).hexdigest())[:args.bg])
    t0 = time.time()
    bgf = os.path.join(CACHE, "background.npz")
    if args.reuse_bg and os.path.exists(bgf):
        with np.load(bgf) as z:
            stats = [(z["mu%d" % c], z["S%d" % c], int(z["n%d" % c]) if "n%d" % c in z else 0)
                     for c in range(len(cd.CHANNELS))]
        print("background statistics from %s" % bgf)
    else:
        stats = background_stats(cache, bg, args.per_pano)
        np.savez(bgf, **{"mu%d" % c: stats[c][0] for c in range(len(stats))},
                 **{"S%d" % c: stats[c][1] for c in range(len(stats))}, **{"n%d" % c: stats[c][2] for c in range(len(stats))})
    print("background: %s windows (%d panoramas, %.0fs)" % ([n for _, _, n in stats], len(bg), time.time() - t0),
          flush=True)
    T = [whitening(S, args.k, args.shrink) for _, S, _ in stats]
    # whitened placement vectors of every detector member
    vec, owner, pfold = [], [], []
    for d, ps in enumerate(members):
        c = dets[d]["channel"]
        g = cd.grid(cd.CHANNELS[c])
        for pid, p in ps:
            yaw = (float(p["heading"]) - float(recs[pid]["heading"]) + 180.0) % 360.0 - 180.0
            v = placement_vector(cache.get(pid)[c][0], g, stats[c][0], yaw, float(p.get("pitch", 0.0)))
            vec.append(T[c] @ v)
            owner.append(d)
            pfold.append(fold_of(pid))
    owner, pfold = np.array(owner), np.array(pfold)
    out = {}
    for name, sel in (("all", np.ones(len(owner), bool)), ("A", pfold == 0), ("B", pfold == 1)):
        for c in range(len(cd.CHANNELS)):
            idx = [d for d in range(len(dets)) if dets[d]["channel"] == c]
            U = np.zeros((args.k, len(idx)))
            for j, d in enumerate(idx):
                k = np.flatnonzero(sel & (owner == d))
                if len(k):
                    m = np.mean([vec[i] for i in k], axis=0)
                    U[:, j] = m / max(np.linalg.norm(m), 1e-9)
            G = T[c].T @ U
            out["G%d_%s" % (c, name)] = G.astype(np.float32)
            out["O%d_%s" % (c, name)] = (stats[c][0] @ G).astype(np.float32)
    meta = {"detectors": dets, "channels": list(cd.CHANNELS), "k": args.k, "shrink": args.shrink,
            "bg_panoramas": len(bg)}
    np.savez(os.path.join(CACHE, "detectors_raw.npz"), meta=np.array(json.dumps(meta)), **out)
    print("saved scratch/cards/detectors_raw.npz (%.0fs)" % (time.time() - t0))


def raw_bank(raw, name="all"):
    """ClueDetectors of one detector set of detectors_raw.npz (no calibration)."""
    meta = json.loads(str(raw["meta"]))
    arr = {"meta": json.dumps({"channels": meta["channels"], "detectors": meta["detectors"]})}
    for c in range(len(meta["channels"])):
        arr["G%d" % c], arr["O%d" % c] = raw["G%d_%s" % (c, name)], raw["O%d_%s" % (c, name)]
    return cd.ClueDetectors(arr)


# ----------------------------------------------------------------------------- score
_W = {}


def _score_init(raw_path):
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    raw = np.load(raw_path)
    _W["banks"] = {n: raw_bank(raw, n) for n in ("all", "A", "B")}
    _W["cache"] = CellCache()


def _score_job(job):
    pid, name = job
    bank = _W["banks"][name]
    z, row, col = bank.scores(_W["cache"].get(pid))
    if name != "all":  # a detector without placements in the fold has no template there: missing, not 0
        f = 0 if name == "A" else 1
        z[[d.get("n_fold", [1, 1])[f] == 0 for d in bank.dets]] = np.nan
    return pid, z.astype(np.float32), row.astype(np.int16), col.astype(np.int16)


def score_set(pid, rounds):
    """Detector set scoring a panorama: training panoramas with placements get the other fold's."""
    r = rounds.get(pid)
    if r is not None and r["split"] == "train":
        return "B" if fold_of(pid) == 0 else "A"
    return "all"


def cmd_score(args):
    _, rounds = frozen_labels()
    cache = CellCache()
    raw = os.path.join(CACHE, "detectors_raw.npz")
    ids = cache.ids()
    t0 = time.time()
    res = {}
    with Pool(args.workers, initializer=_score_init, initargs=(raw,)) as pool:
        for i, (pid, z, row, col) in enumerate(pool.imap_unordered(_score_job, [(p, score_set(p, rounds)) for p in ids],
                                                                   chunksize=16)):
            res[pid] = (z, row, col)
            if (i + 1) % 2000 == 0:
                print("  %d/%d  %.1f pano/s" % (i + 1, len(ids), (i + 1) / (time.time() - t0)), flush=True)
    np.savez(os.path.join(CACHE, "scores.npz"), ids=np.array(ids), Z=np.array([res[p][0] for p in ids], np.float16),
             row=np.array([res[p][1] for p in ids], np.uint8), col=np.array([res[p][2] for p in ids], np.uint8),
             sets=np.array([score_set(p, rounds) for p in ids]), det_hash=_file_hash(raw))
    print("saved scratch/cards/scores.npz (%d panoramas, %.0fs)" % (len(ids), time.time() - t0))


def cached_loglik(pids, classes, bank=None):
    """(n, C) card log-likelihoods of panoramas from the score cache with the saved bank; None when the
    bank or the cache is missing or they come from other detectors.  Panoramas missing from the cache
    get zeros (no evidence)."""
    bank = bank or cd.ClueDetectors.load()
    path = os.path.join(CACHE, "scores.npz")
    if bank is None or not os.path.exists(path):
        sys.stderr.write("no %s - card evidence left out (tools/build_clue_detectors.py all)\n" % (
            "detector bank" if bank is None else "scratch/cards/scores.npz"))
        return None
    z = np.load(path)
    if str(z["det_hash"]) != bank.meta.get("det_hash"):
        sys.stderr.write("scratch/cards/scores.npz is from other detectors - card evidence left out\n")
        return None
    row = {str(p): i for i, p in enumerate(z["ids"])}
    out = np.zeros((len(pids), len(classes)))
    have = [k for k, p in enumerate(pids) if p in row]
    if have:
        out[have] = bank.country_loglik(bank.zt(z["Z"][[row[pids[k]] for k in have]]), classes)
    return out


# ----------------------------------------------------------------------------- calibration
def logistic_fit(X, y, w=None, lam=1e-3, iters=50):
    """Weighted ridge logistic regression by Newton's method."""
    w = np.ones(len(y)) if w is None else w
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = cd._sigmoid(X @ b)
        g = X.T @ (w * (p - y)) + lam * b
        H = (X * (w * p * (1 - p))[:, None]).T @ X + lam * np.eye(len(b))
        step = np.linalg.solve(H, g)
        b -= step
        if np.abs(step).max() < 1e-8:
            break
    return b


def auc(pos, neg):
    """Mann-Whitney AUC (ties count one half)."""
    if not len(pos) or not len(neg):
        return None
    x = np.concatenate([pos, neg])
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[o] = np.arange(1, len(x) + 1)
    xs = x[o]
    _, start, cnt = np.unique(xs, return_index=True, return_counts=True)
    for s0, c in zip(start, cnt):
        if c > 1:
            r[o[s0:s0 + c]] = s0 + (c + 1) / 2.0
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def topn_hits(order_ids, truth, n=3):
    return float(len(set(order_ids[:n]) & truth))


def load_scores(D):
    z = np.load(os.path.join(CACHE, "scores.npz"))
    ids = [str(p) for p in z["ids"]]
    return ids, z["Z"].astype(np.float64)[:, :D], z["row"], z["col"], str(z["det_hash"])


def cmd_calibrate(args):
    from engine.hints import ClueBase, observations, region_probs_for
    from engine.model import GeoModel, GroupGaussian
    clues, rounds = frozen_labels()
    pl, cards = placements(clues, rounds)
    raw = np.load(os.path.join(CACHE, "detectors_raw.npz"))
    meta = json.loads(str(raw["meta"]))
    dets = meta["detectors"]
    D = len(dets)
    ids, Z, Zrow, Zcol, det_hash = load_scores(D)
    if det_hash != _file_hash(os.path.join(CACHE, "detectors_raw.npz")):
        raise SystemExit("scores.npz is from other detectors - run `score` first")
    row = {p: i for i, p in enumerate(ids)}
    recs = {r["pano_id"]: r for r in load_records()}
    split = np.array([split_of(recs[p]) for p in ids])
    label = np.array([recs[p]["label"] for p in ids])
    lab_train = np.array([rounds.get(p, {}).get("split") == "train" for p in ids])
    trn = split == "train"
    fin = np.isfinite(Z)
    zmu = np.array([Z[lab_train & fin[:, d], d].mean() if (lab_train & fin[:, d]).any() else 0.0 for d in range(D)])
    zsd = np.array([Z[lab_train & fin[:, d], d].std() if (lab_train & fin[:, d]).sum() > 1 else 1.0 for d in range(D)])
    zsd = np.maximum(zsd, 1e-3)
    ZT = np.where(fin, (Z - zmu) / zsd, 0.0)
    print("scores: %d panoramas (%s), %d detectors" % (len(ids), dict(Counter(split)), D))

    # ---- card catalogue: frequencies over the labelled train + calib panoramas (as the clue index)
    legal = [p for p in cards if rounds[p]["split"] in ("train", "calib")]
    n_cc = Counter(rounds[p]["label"] for p in legal)
    n_card = Counter((rounds[p]["label"], s) for p in legal for s in cards[p])
    kb = ClueBase()
    det_of = defaultdict(list)
    for d, x in enumerate(dets):
        for s in x["cards"]:
            det_of[(x["country"], s)].append(d)
    cat = defaultdict(dict)
    seterra = defaultdict(set)
    ctype = {}
    for pid, s, cc, stem, p in pl:
        if s in ("train", "calib"):
            seterra[(cc, stem)].update(p.get("seterraRegionIds") or [])
            ctype[(cc, stem)] = p.get("type")
    for (cc, stem), n in n_card.items():
        regs = kb._region_codes(cc, sorted(seterra[(cc, stem)])) if seterra[(cc, stem)] else []
        ds = det_of.get((cc, stem)) or []
        # the card's own detector with the most placements (else its group detector)
        best = max(ds, key=lambda d: (not dets[d]["group"], dets[d]["n"])) if ds else None
        cat[cc][stem] = {"freq": n / float(n_cc[cc]), "det": best, "regions": regs, "type": ctype.get((cc, stem))}
    for d in dets:
        d["freq"] = float(sum(n_card[(d["country"], s)] for s in d["cards"]) / max(n_cc[d["country"]], 1))

    def freq_loo(cc, stem, pid):
        n, N = n_card[(cc, stem)], n_cc[cc]
        if rounds.get(pid, {}).get("split") in ("train", "calib") and pid in cards:
            n -= stem in cards[pid]
            N -= 1
        return n / float(max(N, 1))

    def pairs(pids):
        """(features, y, pid, stem, own detector or -1) of every catalogue card of the true country of
        each panorama."""
        X, y, P, S, O = [], [], [], [], []
        for pid in pids:
            cc = rounds[pid]["label"]
            zt = ZT[row[pid]]
            for stem, c in cat[cc].items():
                f = freq_loo(cc, stem, pid)
                d = c["det"]
                own = zt[d] if d is not None and not dets[d]["group"] else 0.0
                grp = zt[d] if d is not None and dets[d]["group"] else 0.0
                X.append([1.0, cd._logit(max(f, 0.5 / n_cc[cc])), own, grp])
                y.append(float(stem in cards[pid]))
                P.append(pid)
                S.append(stem)
                O.append(d if d is not None and not dets[d]["group"] else -1)
        return np.array(X), np.array(y), P, S, np.array(O, int)

    def hits(P, S, O, y):
        """1 where the card is present and its own detector's best window lies within half a window of
        one of its placements; NaN without an own detector."""
        out = np.full(len(P), np.nan)
        for i in np.flatnonzero(O >= 0):
            out[i] = 0.0
            if y[i]:
                ang, fov = window_angle(clues, recs, P[i], S[i], dets[O[i]], Zrow[row[P[i]], O[i]], Zcol[row[P[i]], O[i]])
                out[i] = float(ang <= fov / 2.0)
        return out

    lab_tr = [p for p in cards if rounds[p]["split"] == "train" and p in row]
    lab_ca = [p for p in cards if rounds[p]["split"] == "calib" and p in row]
    lab_te = [p for p in cards if rounds[p]["split"] == "test" and p in row]
    Xp, yp, Pp, Sp, Op = pairs(lab_tr)
    a = logistic_fit(Xp, yp)
    a0 = logistic_fit(Xp[:, :2], yp)
    print("presence model on %d (panorama, card) pairs of %d training panoramas: a = %s" % (
        len(yp), len(lab_tr), np.round(a, 3).tolist()))
    report = {"n_detectors": D, "presence": {"a": a.tolist()}}
    report["detection_train_oof"] = detection_report(Xp, yp, Pp, Sp, cat, dets, rounds, clues, recs, Zrow, Zcol, row)
    print("train (out-of-fold) detection: %s" % json.dumps(report["detection_train_oof"]))
    # direction model: P(present and the best window within half a window of a placement | true country)
    hp = hits(Pp, Sp, Op, yp)
    m_ = np.isfinite(hp)
    Xd = np.column_stack([np.ones(m_.sum()), Xp[m_, 1], Xp[m_, 2]])
    b = logistic_fit(Xd, hp[m_])
    qd = cd._sigmoid(Xd @ b)
    print("direction model on %d training pairs (%.3f hit): b = %s; P(hit) quantiles 50/90/99/max %s" % (
        m_.sum(), hp[m_].mean(), np.round(b, 3).tolist(), np.round(np.percentile(qd, [50, 90, 99, 100]), 3).tolist()))
    report["direction_train_oof"] = {"b": b.tolist(), "n": int(m_.sum()), "hit_rate": float(hp[m_].mean()),
                                     "by_threshold": dir_table(qd, hp[m_])}

    # ---- country model: Gaussian class densities of the standardised max scores (out-of-fold training
    # detections), shared shrunk covariance -> linear discriminants
    model = GeoModel.load()
    classes = model.classes
    cix = {c: i for i, c in enumerate(classes)}
    tr_rows = np.flatnonzero(trn & np.array([c in cix for c in label]))
    yi = np.array([cix[c] for c in label[tr_rows]])
    gg = GroupGaussian(shrink=args.cm_shrink, mean_tau=args.cm_tau).fit(ZT[tr_rows], yi, len(classes))
    Si = np.linalg.inv(gg.S)
    cm_A = Si @ gg.mu.T
    cm_b = 0.5 * np.sum((gg.mu @ Si) * gg.mu, axis=1)
    print("country model on %d training panoramas (%d classes present)" % (len(yi), len(set(yi.tolist()))))
    pres = {"a": a.tolist(), "a_freq_only": a0.tolist(), "direction": {"a": b.tolist(), "min_prob": 1.01}}
    det_meta = {"channels": meta["channels"], "detectors": dets, "cards": cat, "presence": pres,
                "country_model": {"n_train": int(len(yi)), "shrink": args.cm_shrink, "tau": args.cm_tau},
                "region": {"weight": 0.0}, "k": meta["k"], "shrink": meta["shrink"]}
    arrays = {"zmu": zmu, "zsd": zsd, "cm_classes": np.array(classes), "cm_med": gg.med, "cm_sc": gg.sc,
              "cm_A": cm_A.astype(np.float32), "cm_b": cm_b}
    for c in range(len(meta["channels"])):
        arrays["G%d" % c], arrays["O%d" % c] = raw["G%d_all" % c], raw["O%d_all" % c]
    bank = cd.ClueDetectors(dict(arrays, meta=json.dumps(det_meta)))

    def country_eval(rows_):
        ll = bank.country_loglik(ZT[rows_], classes)
        lab = label[rows_]
        ok = np.array([c in cix for c in lab])
        y_ = np.array([cix.get(c, 0) for c in lab])
        lp = ll - np.log(np.exp(ll).sum(1, keepdims=True))
        rank = (lp > lp[np.arange(len(y_)), y_][:, None]).sum(1)
        return {"n": int(ok.sum()), "top1": float(((rank == 0) & ok).mean()), "top5": float(((rank < 5) & ok).mean()),
                "gain_bits": float(np.mean((lp[np.arange(len(y_)), y_] + np.log(len(classes)))[ok]) / np.log(2))}

    names = ("calib", "test") if args.test else ("calib",)
    for name in names:
        report["country_" + name] = country_eval(np.flatnonzero(split == name))
        print("[%s] cards-only country model (uniform prior): %s" % (name, report["country_" + name]))

    # ---- region exponent on CALIB (P(true region | true country))
    regs = region_posteriors_true_country(model, [p for p in ids if split[row[p]] in ("calib", "test")], recs)
    best_w, best = 0.0, None
    for w in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0):
        bank.region["weight"] = w
        s = region_score(bank, regs, [p for p in ids if split[row[p]] == "calib"], recs, ZT, row)
        print("  region exponent %.2f: calib mean log P(true region) %.4f (n=%d)" % (w, s[0], s[1]))
        if best is None or s[0] > best + 1e-4:
            best, best_w = s[0], w
    det_meta["region"] = {"weight": best_w}
    for name in names:
        for w in sorted({0.0, best_w}):
            bank.region["weight"] = w
            s = region_score(bank, regs, [p for p in ids if split[row[p]] == name], recs, ZT, row, full=True)
            report["region_%s_w%s" % (name, w)] = s
            print("[%s] region given true country, exponent %.2f: log P %.4f, top1 %.3f, top3 %.3f (n=%d)" % (
                name, w, s[0], s[2], s[3], s[1]))
    bank.region["weight"] = best_w

    # ---- card ranking given the true country; hint order and direction thresholds on CALIB
    feats = calib_features(lab_ca + lab_te)
    rp_all = {p: [{"code": k, "name": "", "country": rounds[p]["label"], "probability": v}
                  for k, v in regs.get(p, {}).items()] for p in lab_ca + lab_te}
    for name, pids in (("calib", lab_ca), ("test", lab_te))[:len(names)]:
        X, y, P, S, O = pairs(pids)
        if name == "test":
            report["detection_test"] = detection_report(X, y, P, S, cat, dets, rounds, clues, recs, Zrow, Zcol, row)
            print("test detection: %s" % json.dumps(report["detection_test"]))
        h = {"frequency": card_rank(X, P, S, pids, a0, [0, 1], rounds, cards, kb),
             "detection + frequency": card_rank(X, P, S, pids, a, [0, 1, 2, 3], rounds, cards, kb),
             "detection only": card_rank(X, P, S, pids, np.array([0.0, 0.0, 1.0, 0.7]), [0, 1, 2, 3], rounds, cards, kb)}
        base, ntrue = {}, {}
        for pid in pids:
            cc = rounds[pid]["label"]
            F = feats.get(pid, {})
            emb = kb.index.embed(F) if kb.index is not None and F else None
            rp = region_probs_for(rp_all.get(pid), cc)
            ranked = kb.rank_cards(cc, observations(F), rp, emb, exclude=pid if name == "calib" else None)
            base[pid] = [c["id"] for _, c, _, _ in ranked]
            ntrue[pid] = len({kb.card_id(cc, s) for s in cards[pid]})
        truth = {p: {kb.card_id(rounds[p]["label"], s) for s in cards[p]} for p in pids}
        h["hints (clue index)"] = {p: topn_hits(base[p], truth[p]) for p in pids}
        nt = np.array([ntrue[p] for p in pids], float)
        prob = defaultdict(list)
        for x, p, st, q in zip(X, P, S, cd._sigmoid(X @ a)):
            if x[2] != 0 or x[3] != 0:  # cards with a detector
                prob[p].append((q, st))
        if name == "calib":
            # detected cards lead only if the best threshold beats the clue-index order beyond noise
            hb = np.array([h["hints (clue index)"][p] for p in pids])
            sc = []
            for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
                hv = np.array([v[0] for v in hint_hits(prob, pids, rounds, base, truth, kb, t).values()])
                d_ = (hv - hb) / 3.0 + (hv - hb) / np.maximum(nt, 1)
                lo = paired_lower(d_)
                sc.append((float(d_.mean()), lo, t))
                print("  hint threshold %.2f: calib P@3+R@3 %+.4f vs clue index (95%% lower bound %+.4f)" % (t, d_.mean(), lo))
            best = max(sc)
            pres["hint_min_prob"] = best[2] if best[1] > 0 else 1.01
            print("  -> hint threshold %.2f" % pres["hint_min_prob"])
        hh = hint_hits(prob, pids, rounds, base, truth, kb, pres["hint_min_prob"])
        h["hints, detected first"] = {p: v[0] for p, v in hh.items()}
        # directions of the shown cards (top 3 of the hint order) with an own detector
        qd = cd._sigmoid(np.column_stack([np.ones(len(X)), X[:, 1], X[:, 2]]) @ b)
        hd = hits(P, S, O, y)
        cid = [kb.card_id(rounds[p]["label"], st) for p, st in zip(P, S)]
        shown = np.array([O[i] >= 0 and cid[i] in hh[P[i]][1] for i in range(len(P))])
        tab = dir_table(qd[shown], hd[shown])
        print("[%s] directions of the shown cards (%d with an own detector in %d panoramas): %s" % (
            name, shown.sum(), len(pids), "; ".join("P>=%.2f: n %d prec %.3f" % (r["min_prob"], r["n"], r["precision"])
                                                    for r in tab)))
        if name == "calib":
            ok = [r["min_prob"] for r in tab if r["n"] >= DIR_MIN_N and r["precision"] >= DIR_BAR]
            pres["direction"]["min_prob"] = min(ok) if ok else 1.01
            print("  -> direction threshold %.2f (bar: precision >= %.2f on >= %d shown directions)" % (
                pres["direction"]["min_prob"], DIR_BAR, DIR_MIN_N))
        t_ = pres["direction"]["min_prob"]
        sel = shown & (qd >= t_)
        res = {"n": len(pids), "directions": {"by_threshold": tab, "min_prob": t_, "shown": int(sel.sum()),
                                              "precision": float(hd[sel].mean()) if sel.any() else None}}
        hb = np.array([h["frequency"][p] for p in pids])
        print("[%s] cards given the true country, n=%d%s" % (name, len(pids), " (frequencies leave-one-out)"
                                                            if name == "calib" else ""))
        for k, val in h.items():
            hv = np.array([val[p] for p in pids])
            dd = bootstrap(hv, hb, nt)
            res[k] = {"precision@3": float(np.mean(hv / 3.0)), "recall@3": float(np.mean(hv / np.maximum(nt, 1))),
                      "vs_frequency": dd}
            print("   %-24s precision@3 %.3f  recall@3 %.3f   this - frequency: P %+.3f [%+.3f, %+.3f]  "
                  "R %+.3f [%+.3f, %+.3f]" % ((k, res[k]["precision@3"], res[k]["recall@3"]) + tuple(dd[0]) + tuple(dd[1])))
        report["cards_" + name] = res
    det_meta["presence"] = pres
    det_meta["eval"] = report
    det_meta["built"] = time.strftime("%Y-%m-%d %H:%M")
    det_meta["n_train_placements"] = int(sum(d["n"] for d in dets))
    det_meta["det_hash"] = det_hash
    np.savez_compressed(cd.DETECTORS_PATH, meta=np.array(json.dumps(det_meta, ensure_ascii=False)), **arrays)
    print("saved %s (%d detectors, %.1f MB)" % (cd.DETECTORS_PATH, D, os.path.getsize(cd.DETECTORS_PATH) / 1e6))


def card_rank(X, P, S, pids, coef, cols, rounds, cards, kb):
    """Hits@3 of the cards ranked by the presence model with the given columns."""
    s = X[:, cols] @ coef
    at = defaultdict(list)
    for i, p in enumerate(P):
        at[p].append(i)
    out = {}
    for pid in pids:
        o = sorted(at[pid], key=lambda i: -s[i])
        truth = {kb.card_id(rounds[pid]["label"], st) for st in cards[pid]}
        out[pid] = topn_hits(list(dict.fromkeys(kb.card_id(rounds[pid]["label"], S[i]) for i in o)), truth)
    return out


def window_angle(clues, recs, pid, stem, det, wrow, wcol):
    """(angle (deg) between a detector's best window and the nearest placement of the card, window fov)."""
    from engine.hints import clue_stem
    g = cd.grid(cd.CHANNELS[det["channel"]])
    a1 = np.radians([float(cd.window_yaw(g, wcol)), float(cd.window_pitch(g, wrow))])
    best = 180.0
    for p in clues[pid]:
        if (clue_stem(p.get("title")) or p["id"].lower()) != stem:
            continue
        py = (float(p["heading"]) - float(recs[pid]["heading"]) + 180.0) % 360.0 - 180.0
        a2 = np.radians([py, float(p.get("pitch", 0.0))])
        c = np.sin(a1[1]) * np.sin(a2[1]) + np.cos(a1[1]) * np.cos(a2[1]) * np.cos(a1[0] - a2[0])
        best = min(best, float(np.degrees(np.arccos(np.clip(c, -1, 1)))))
    return best, g["fov"]


def dir_table(q, hit, grid=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)):
    """Shown directions and their precision (share pointing at the card) per threshold on P(hit)."""
    return [{"min_prob": t, "n": int((q >= t).sum()), "precision": float(hit[q >= t].mean()) if (q >= t).any() else 0.0}
            for t in grid]


def paired_lower(d, reps=4000, seed=0):
    """2.5th percentile of the bootstrap mean of per-round differences d."""
    idx = np.random.RandomState(seed).randint(0, len(d), (reps, len(d)))
    return float(np.percentile(d[idx].mean(1), 2.5))


def detection_report(X, y, P, S, cat, dets, rounds, clues, recs, Zrow, Zcol, row):
    """Detection quality given the true country: AUC of the card's standardised max score (present vs
    absent cards of the same country, own detectors only) and, for present cards, the angle between the
    detected window and the nearest placement of the card (median, share within half the window fov)."""
    own = X[:, 2] != 0
    pos, neg = X[own & (y == 1), 2], X[own & (y == 0), 2]
    ang, near = [], []
    for i in np.flatnonzero(own & (y == 1)):
        pid, stem = P[i], S[i]
        d = cat[rounds[pid]["label"]][stem]["det"]
        best, fov = window_angle(clues, recs, pid, stem, dets[d], Zrow[row[pid], d], Zcol[row[pid], d])
        ang.append(best)
        near.append(best <= fov / 2.0)
    return {"auc_present_vs_absent": auc(pos, neg), "n_present": int(len(pos)), "n_absent": int(len(neg)),
            "direction_median_deg": float(np.median(ang)) if ang else None,
            "direction_within_half_fov": float(np.mean(near)) if near else None}


def hint_hits(prob, pids, rounds, base, truth, kb, thr):
    """{pid: (hits@3, shown card ids)} of the hint order: cards with a detector and P(present) >= thr first
    (prob: {pid: [(P, stem)]}), then the clue-index ranking."""
    out = {}
    for pid in pids:
        cc = rounds[pid]["label"]
        det = [kb.card_id(cc, st) for q, st in sorted(prob.get(pid, []), key=lambda r: -r[0]) if q >= thr]
        det = [c for c in det if (cc, c) in kb._cards]
        order = list(dict.fromkeys(det + base[pid]))
        out[pid] = (topn_hits(order, truth[pid]), set(order[:3]))
    return out


def calib_features(pids):
    """{pano_id: {'module.feature': value}} from the feature caches (for the clue-index ranking)."""
    from calibrate import load_features
    from build_clue_index import GROUPS
    recs = [r for r in load_records() if r["pano_id"] in set(pids)]
    X, fnames, recs = load_features(GROUPS.split(","), recs)
    return {r["pano_id"]: dict(zip(fnames, X[i].astype(float))) for i, r in enumerate(recs)}


def region_posteriors_true_country(model, pids, recs):
    """{pano_id: {region code: P(region | true country)}} from the saved country model's region kernel."""
    from train_model import group_matrix
    from engine.geo import region_info
    sub = [recs[p] for p in pids]
    mats = {g: group_matrix(g, sub)[0] for g in model.groups}
    ev = model.evidence(mats)
    cls = {c: i for i, c in enumerate(model.classes)}
    out = {}
    for k, p in enumerate(pids):
        cc = recs[p]["label"]
        if cc not in cls:
            continue
        lp = np.full(len(model.classes), -1e9)
        lp[cls[cc]] = 0.0
        pr = model.region_posterior(lp, ev["_d2"][k])
        out[p] = {region_info(int(i))[0]: float(pr[i]) for i in np.flatnonzero(pr > 0)
                  if region_info(int(i))[2] == cc}
    return out


def region_score(bank, regs, pids, recs, ZT, row, full=False):
    """Mean log P(true region | true country) after the card update (floor 1e-4); with full also
    top1 / top3."""
    from engine.geo import region_at
    ll, t1, t3 = [], [], []
    for p in pids:
        rp = regs.get(p)
        reg = region_at(recs[p]["lat"], recs[p]["lng"])
        if not rp or not reg:
            continue
        s = sum(rp.values())
        rp = {k: v / s for k, v in rp.items()}
        up = bank.update_regions(recs[p]["label"], ZT[row[p]], rp)
        v = up.get(reg[0], 0.0)
        ll.append(np.log(max(v, 1e-4)))
        r = sum(1 for x in up.values() if x > v)
        t1.append(r == 0 and v > 0)
        t3.append(r < 3 and v > 0)
    out = [float(np.mean(ll)) if ll else 0.0, len(ll)]
    if full:
        out += [float(np.mean(t1)) if t1 else 0.0, float(np.mean(t3)) if t3 else 0.0]
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("cells")
    p.add_argument("--clues", default=CLUES)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--extra", type=int, default=20, help="unlabelled training panoramas per country")
    p = sub.add_parser("fit")
    p.add_argument("--k", type=int, default=200, help="whitened dimensions")
    p.add_argument("--shrink", type=float, default=0.1)
    p.add_argument("--bg", type=int, default=3000, help="panoramas for the background statistics")
    p.add_argument("--per-pano", type=int, default=400, help="background windows per panorama and channel")
    p.add_argument("--reuse-bg", action="store_true", help="background statistics of the previous fit")
    p = sub.add_parser("score")
    p.add_argument("--workers", type=int, default=4)
    p = sub.add_parser("calibrate")
    p.add_argument("--cm-shrink", type=float, default=0.1, help="covariance shrinkage of the country model")
    p.add_argument("--cm-tau", type=float, default=4.0, help="class-mean shrinkage (pseudo-panoramas)")
    p.add_argument("--test", action="store_true", help="also report TEST (only with the final settings)")
    p = sub.add_parser("all")
    p.add_argument("--clues", default=CLUES)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--extra", type=int, default=20)
    p.add_argument("--k", type=int, default=200)
    p.add_argument("--shrink", type=float, default=0.1)
    p.add_argument("--bg", type=int, default=3000)
    p.add_argument("--per-pano", type=int, default=400)
    p.add_argument("--reuse-bg", action="store_true")
    p.add_argument("--cm-shrink", type=float, default=0.1)
    p.add_argument("--cm-tau", type=float, default=4.0)
    p.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if args.cmd == "all":
        for f in (cmd_cells, cmd_fit, cmd_score, cmd_calibrate):
            f(args)
    elif args.cmd:
        {"cells": cmd_cells, "fit": cmd_fit, "score": cmd_score, "calibrate": cmd_calibrate}[args.cmd](args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
