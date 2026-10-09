#!/usr/bin/env python3
"""
Stream the panoramas of crawled public duel rounds into the feature caches without keeping the images.

  python3 tools/stream_duels.py [--threads 12] [--workers 5] [--commit 500] [--limit N]
  python3 tools/stream_duels.py --refresh          streamed panoramas whose features are missing from a cache
                                                   (a feature module changed and `calibrate.py features` could
                                                   not recompute them): download again, recompute, delete
  python3 tools/stream_duels.py --verify-compat    a cache labelled with another module hash: re-extract --n
                                                   panoramas (both car-axis regimes) with the current module; if the
                                                   cache is reproduced exactly, record the hash pair in
                                                   scratch/features/compat.json and relabel the cache with the
                                                   current module hash (no dependence on any compat file)
  python3 tools/stream_duels.py --status           counts; streamed panoramas missing rows of a cache

Rounds: data/calibration/duel_rounds.json (tools/crawl_duels.py) that pass build_dataset.duel_candidates (not a
round of the user nor within 1 km of one, one round per panorama, panorama not yet in the dataset), in a fixed
pseudo-random order (md5 of the pano id) so that an interrupted run is a uniform sample.  For each:
  1. Street View metadata + tiles (engine/streetview.py, <= 12 threads) -> scratch/dataset/stream_tmp/<id>.jpg
     (JPEG quality 90, exactly as build_dataset.fetch_and_store) + <id>.json (the index record)
  2. the feature modules in <= 5 processes with calibrate._init / calibrate._extract_one: the code path and the
     module hashes of `calibrate.py features`, incl. the hide_car_axis rule (by pano id)
  3. every --commit panoramas (or --commit-s seconds), under the cache lock: rows upserted into
     scratch/features/<module>.npz (temp file + os.replace), then the index records (mode "duel",
     "pano_deleted": true) appended to scratch/dataset/index.jsonl; then the images are deleted.
Crash-safe and resumable: a record is in the index only after its features are in every cache; images left in
stream_tmp/ are extracted on the next run (or deleted if already committed); cache rows of a crashed commit are
replaced on the next one; temp files of a cache write killed half-way are removed at the start.  Rounds without a panorama are listed in scratch/dataset/stream_skipped.jsonl and not
tried again (--retry-skipped).  Disk: ~2.5 KB of features + ~0.6 KB of index per panorama; the staging
directory holds <= --commit + 60 images (~0.45 MB each).
"""
import argparse
import hashlib
import json
import os
import queue
import shutil
import sys
import threading
import time

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one BLAS thread per worker process (results identical, no oversubscription)

import multiprocessing  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import calibrate  # noqa: E402
from build_dataset import (DUELS_FILE, OWN_FILE, duel_candidates, duel_extra, duel_meta, load_index, retry,  # noqa: E402
                           round_key)
from engine.geo import country_at  # noqa: E402

MAX_THREADS, MAX_WORKERS = 12, 5
MIN_FREE_GB = 3.0                     # stop when the disk has less free space than this
STAGE_NAME = "stream_tmp"
SKIP_NAME = "stream_skipped.jsonl"
COMPAT_FILE = calibrate.COMPAT_FILES[0]


def stage_dir(dataset):
    return os.path.join(dataset, STAGE_NAME)


def dir_bytes(path):
    total = 0
    for dp, _, fs in os.walk(path):
        for f in fs:
            try:
                total += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return total


def path_bytes(p):
    return dir_bytes(p) if os.path.isdir(p) else (os.path.getsize(p) if os.path.exists(p) else 0)


def module_info(names):
    """{module: (hash, FEATURE_NAMES)} of the current module sources."""
    from engine import features
    out = {}
    for n in names:
        m = features.load(n)
        out[n] = (calibrate.module_hash(m), list(m.FEATURE_NAMES))
    return out


def cache_problems(info, feat_dir=calibrate.FEAT_DIR, compat_files=None):
    """[(module, cache hash, module hash)] of caches the current modules may not append to."""
    out = []
    for n, (h, names) in info.items():
        c = calibrate.read_cache(n, feat_dir)
        if c is None:
            continue
        if c["names"] != names or not calibrate.hash_ok(n, c["hash"], h, compat_files):
            out.append((n, c["hash"], h))
    return out


def append_index(recs, dataset):
    """Append index records (one fsync'ed write); a partial last line of a crashed writer is terminated first
    so that it cannot swallow the first new record."""
    path = os.path.join(dataset, "index.jsonl")
    lead = ""
    if os.path.exists(path) and os.path.getsize(path):
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            lead = "" if f.read(1) == b"\n" else "\n"
    with open(path, "a") as f:
        f.write(lead + "".join(json.dumps(r) + "\n" for r in recs))
        f.flush()
        os.fsync(f.fileno())


def read_skips(dataset):
    """build_dataset.round_key of the rounds given up before."""
    out = set()
    path = os.path.join(dataset, SKIP_NAME)
    if os.path.exists(path):
        with open(path) as f:
            lines = f.readlines()
        for line in lines:
            try:
                out.add(round_key(json.loads(line)))
            except Exception:
                pass
    return out


def add_skip(dataset, r, reason, lock):
    with lock:
        with open(os.path.join(dataset, SKIP_NAME), "a") as f:
            f.write(json.dumps({"pano_id": r.get("pano_id"), "game": r.get("game"), "round": r.get("round"),
                                "reason": reason}) + "\n")


def make_record(meta, r):
    """Index record of a streamed duel panorama (as build_dataset.fetch_and_store + "pano_deleted")."""
    rec = dict(meta)
    rec["mode"] = "duel"
    rec["raster_country"] = country_at(meta["lat"], meta["lng"])
    rec.update(duel_extra(r))
    rec["pano_deleted"] = True
    return rec


def store_image(img, path):
    """JPEG quality 90 as build_dataset.fetch_and_store, written to a temp name first (no partial .jpg)."""
    img.save(path + ".part", format="JPEG", quality=90)
    os.replace(path + ".part", path)


def write_json(obj, path):
    with open(path + ".part", "w") as f:
        json.dump(obj, f)
    os.replace(path + ".part", path)


# ------------------------------------------------------------------ worker processes
_HASHES = None


def _init(names):
    global _HASHES
    calibrate._init(names)
    _HASHES = tuple(calibrate.module_hash(m) for m in calibrate._WORK_MODS)


def _job(item):
    """(pano id, [feature vector per module] or None, error, module hashes of this worker)."""
    pid, path, heading = item
    try:
        return pid, calibrate._extract_one((path, heading)), None, _HASHES
    except Exception as e:
        return pid, None, repr(e), _HASHES


# ------------------------------------------------------------------ the stream
class Stop(Exception):
    pass


def stream(items, args, info, dataset=calibrate.DATASET, feat_dir=calibrate.FEAT_DIR, refresh=False,
           fetch_meta=duel_meta, download=None, compat_files=None, log=print):
    """Download -> features -> commit for `items` (duel rounds, or with refresh=True index records).  Returns
    a summary dict."""
    if download is None:
        from engine.streetview import download_panorama as download
    names = list(info)
    hashes = {n: info[n][0] for n in names}
    fnames = {n: info[n][1] for n in names}
    stage = stage_dir(dataset)
    os.makedirs(stage, exist_ok=True)
    with calibrate.cache_lock(feat_dir):
        calibrate.clean_tmp(feat_dir, log=log)
    lock = threading.Lock()
    seen = set(load_index(dataset))
    stats = {"committed": 0, "skipped": 0, "failed_download": 0, "failed_extract": 0, "commits": 0}
    t0 = time.time()

    # images of an interrupted run: extract the uncommitted ones again, drop the committed ones
    recovered = []
    for f in sorted(os.listdir(stage)):
        p = os.path.join(stage, f)
        if f.endswith(".part"):
            os.remove(p)
            continue
        if not f.endswith(".json"):
            continue
        pid, jpg = f[:-5], p[:-5] + ".jpg"
        try:
            with open(p) as fh:
                rec = json.load(fh)
        except ValueError:
            rec = None
        done = rec is None or (pid in seen and not refresh) or not os.path.exists(jpg)
        if done or bool(rec.get("_refresh")) != refresh:
            if done:
                for q in (p, jpg):
                    if os.path.exists(q):
                        os.remove(q)
            continue
        recovered.append((rec, jpg))
    for f in os.listdir(stage):  # images without a record
        if f.endswith(".jpg") and not os.path.exists(os.path.join(stage, f[:-4] + ".json")):
            os.remove(os.path.join(stage, f))
    if recovered:
        log("recovered %d downloaded panoramas of an interrupted run" % len(recovered))
    rec_ids = {rec["pano_id"] for rec, _ in recovered}
    items = [it for it in items if it.get("pano_id") not in rec_ids]

    max_pending = args.commit + 8 * args.workers + 2 * args.threads
    slots = threading.Semaphore(max_pending)
    dl_q = queue.Queue()
    stop = threading.Event()
    claimed = set()

    def fetch(it):
        try:
            if stop.is_set():
                return False
            if refresh:
                meta, r = it, None
            else:
                meta = fetch_meta(it)
                if not meta:
                    add_skip(dataset, it, "no panorama", lock)
                    with lock:
                        stats["skipped"] += 1
                    return False
                with lock:
                    if meta["pano_id"] in seen or meta["pano_id"] in claimed or meta["pano_id"] in rec_ids:
                        dup = True
                    else:
                        dup = False
                        claimed.add(meta["pano_id"])
                if dup:
                    add_skip(dataset, it, "duplicate panorama %s" % meta["pano_id"], lock)
                    with lock:
                        stats["skipped"] += 1
                    return False
                r = it
            rec = dict(meta, _refresh=True) if refresh else make_record(meta, r)
            jpg = os.path.join(stage, meta["pano_id"] + ".jpg")
            store_image(retry(download, meta), jpg)
            write_json(rec, jpg[:-4] + ".json")
            dl_q.put((rec, jpg))
            return True
        except Exception as e:
            with lock:
                stats["failed_download"] += 1
                if stats["failed_download"] <= 20 or stats["failed_download"] % 100 == 0:
                    log("download failed (%d): %s %r" % (stats["failed_download"], it.get("pano_id"), e))
            return False

    def producer():
        try:
            for rec, jpg in recovered:
                slots.acquire()
                dl_q.put((rec, jpg))
            with ThreadPoolExecutor(args.threads) as ex:
                for it in items:
                    if stop.is_set():
                        break
                    while not slots.acquire(timeout=1.0):
                        if stop.is_set():
                            break
                    if stop.is_set():
                        break
                    fut = ex.submit(fetch, it)
                    fut.add_done_callback(lambda f: None if f.result() else slots.release())
        finally:
            dl_q.put(None)

    pending = {}

    def gen():
        while True:
            x = dl_q.get()
            if x is None:
                return
            rec, jpg = x
            pending[rec["pano_id"]] = x
            yield rec["pano_id"], jpg, rec.get("heading")

    def release(entries):
        for rec, jpg in entries:
            for q in (jpg, jpg[:-4] + ".json"):
                if os.path.exists(q):
                    os.remove(q)
            slots.release()

    def commit(buf):
        if not buf:
            return
        rows = {n: {rec["pano_id"]: feats[j] for rec, feats, _ in buf} for j, n in enumerate(names)}
        with calibrate.cache_lock(feat_dir):
            sizes = calibrate.upsert_features(rows, hashes, fnames, feat_dir, compat_files, locked=True)
            if not refresh:
                append_index([rec for rec, _, _ in buf], dataset)
        with lock:
            seen.update(rec["pano_id"] for rec, _, _ in buf)
        release([(rec, jpg) for rec, _, jpg in buf])
        stats["committed"] += len(buf)
        stats["commits"] += 1
        dt = time.time() - t0
        rate = stats["committed"] / max(dt, 1e-9)
        left = len(items) + len(recovered) - stats["committed"] - stats["skipped"] - stats["failed_download"] \
            - stats["failed_extract"]
        log("commit %d: +%d -> %d panoramas streamed (cache %d rows), %.2f pano/s, ~%.1f h left, skipped %d, "
            "failed %d/%d, staging %.0f MB, features %.0f MB, free %.1f GB"
            % (stats["commits"], len(buf), stats["committed"], max(sizes.values()), rate,
               max(left, 0) / max(rate, 1e-9) / 3600, stats["skipped"], stats["failed_download"],
               stats["failed_extract"], dir_bytes(stage) / 1e6, dir_bytes(feat_dir) / 1e6,
               shutil.disk_usage(dataset).free / 1e9))

    th = threading.Thread(target=producer, daemon=True)
    ctx = multiprocessing.get_context("spawn")  # the download threads run in this process
    buf, last, clean = [], time.time(), False
    pool = ctx.Pool(args.workers, initializer=_init, initargs=(names,))
    try:
        th.start()
        it = pool.imap_unordered(_job, gen(), chunksize=1)
        while True:
            try:
                pid, feats, err, whash = it.next(timeout=20)
            except multiprocessing.TimeoutError:
                if buf and time.time() - last > args.commit_s:
                    commit(buf)
                    buf, last = [], time.time()
                continue
            except StopIteration:
                break
            rec, jpg = pending.pop(pid)
            if whash is not None and tuple(hashes[n] for n in names) != whash:
                raise Stop("a feature module changed while streaming (worker hashes %s, expected %s): restart"
                           % (whash, [hashes[n] for n in names]))
            if feats is None:
                stats["failed_extract"] += 1
                log("features failed: %s %s" % (pid, err))
                if not refresh:
                    add_skip(dataset, rec, "extract: %s" % err, lock)
                release([(rec, jpg)])
                continue
            buf.append((rec, feats, jpg))
            if len(buf) >= args.commit or time.time() - last > args.commit_s:
                if shutil.disk_usage(dataset).free < MIN_FREE_GB * 1e9:
                    raise Stop("less than %.0f GB free on the disk" % MIN_FREE_GB)
                commit(buf)
                buf, last = [], time.time()
        commit(buf)
        buf, clean = [], True
    except (Stop, calibrate.CacheMismatch, KeyboardInterrupt) as e:
        log("stopped: %s - %d uncommitted panoramas stay in %s for the next run" % (e if str(e) else repr(e),
                                                                                  len(buf), stage))
        stats["stopped"] = str(e) or repr(e)
    finally:
        stop.set()  # the producer leaves within ~1 s (after the downloads in flight)
        if clean:
            pool.close()
        else:
            pool.terminate()
        pool.join()
        th.join(timeout=120)
    stats["seconds"] = round(time.time() - t0)
    return stats


# ------------------------------------------------------------------ selection
def new_items(dataset=calibrate.DATASET, retry_skipped=False, duels_file=DUELS_FILE, own_file=OWN_FILE):
    with open(duels_file) as f:
        rounds = json.load(f)["rounds"]
    with open(own_file) as f:
        own = json.load(f)
    index = load_index(dataset)
    done = {(r.get("game"), r.get("round")) for r in index.values() if r.get("mode") == "duel"}
    skip = set() if retry_skipped else read_skips(dataset)
    keep = [r for r in duel_candidates(rounds, own, seen=index, skip=skip) if (r["game"], r["round"]) not in done]
    keep.sort(key=lambda r: hashlib.md5(round_key(r).encode()).hexdigest())
    return keep, len(rounds)


def refresh_items(info, dataset=calibrate.DATASET, feat_dir=calibrate.FEAT_DIR, compat_files=None):
    """Streamed (image-less) records that miss a row in one of the caches -> (records, modules to compute)."""
    recs = [r for r in calibrate.load_records(dataset, exclude_own=False) if r.get("pano_deleted")]
    have, mods = {}, set()
    for n, (h, names) in info.items():
        c = calibrate.read_cache(n, feat_dir)
        ok = c is not None and c["names"] == names and calibrate.hash_ok(n, c["hash"], h, compat_files)
        have[n] = set(c["ids"]) if ok else set()
    out = []
    for r in recs:
        miss = [n for n in info if r["pano_id"] not in have[n]]
        if miss:
            out.append(r)
            mods.update(miss)
    return out, [n for n in info if n in mods]


def relabeled_caches(info, feat_dir=calibrate.FEAT_DIR):
    """[(module, cache hash, module hash)] of caches labelled with another hash than the current module's (whether
    or not a compat file accepts the pair)."""
    out = []
    for n, (h, names) in info.items():
        c = calibrate.read_cache(n, feat_dir)
        if c is not None and c["hash"] != h and c["names"] == names:
            out.append((n, c["hash"], h))
    return out


def relabel(name, old, new, feat_dir=calibrate.FEAT_DIR):
    """Relabel the cache of `name` from hash `old` to `new` (atomic, under the cache lock; refuses if the cache
    changed its label meanwhile)."""
    with calibrate.cache_lock(feat_dir):
        c = calibrate.read_cache(name, feat_dir)
        if c is None or c["hash"] != old:
            return False
        calibrate.write_cache(name, c["ids"], c["X"], c["names"], new, feat_dir)
    return True


def verify_compat(name, n, workers, dataset=calibrate.DATASET, feat_dir=calibrate.FEAT_DIR, compat_file=COMPAT_FILE,
                  log=print):
    """Re-extract `n` train panoramas of the cache of `name` (smallest md5, both car-axis regimes) with the current
    module; equal features and NaN pattern -> the (module, cache) hash pair is recorded as equivalent in
    compat_file and the cache is relabelled with the module hash."""
    info = module_info([name])
    hc = info[name][0]
    c = calibrate.read_cache(name, feat_dir)
    if c is None or c["hash"] == hc:
        log("%s: nothing to verify (cache %s, module %s)" % (name, c and c["hash"], hc))
        return True
    idx = {p: i for i, p in enumerate(c["ids"])}
    recs = sorted((r for r in calibrate.load_records(dataset, pixels=True)
                   if calibrate.split_of(r) == "train" and r["pano_id"] in idx),
                  key=lambda r: hashlib.md5(r["pano_id"].encode()).hexdigest())[:n]
    jobs = [(r["pano_id"], calibrate.pano_path(r["pano_id"], dataset), r.get("heading")) for r in recs]
    t0 = time.time()
    with multiprocessing.get_context("spawn").Pool(min(workers, MAX_WORKERS), initializer=_init,
                                                   initargs=([name],)) as pool:
        res = pool.map(_job, jobs, chunksize=2)
    failed = [x[0] for x in res if x[1] is None]
    results = {x[0]: x[1] for x in res if x[1] is not None}
    ok, n_cmp, n_bad, d = calibrate.reproduces(results, 0, c, min_rows=min(n, calibrate.MIN_EQUIV_ROWS))
    eq = bool(ok and not failed)
    try:
        compat = json.load(open(compat_file))
    except (OSError, ValueError):
        compat = {}
    e = {"module": hc, "cache": c["hash"], "n": n_cmp, "rows_differing": n_bad, "max_abs_diff": d,
         "regimes": sorted({int(calibrate.hide_car_axis(p)) for p in results}), "failed": len(failed),
         "equivalent": eq, "checked": time.strftime("%Y-%m-%d %H:%M")}
    compat[name] = [x for x in compat.get(name, []) if (x.get("module"), x.get("cache")) != (hc, c["hash"])] + [e]
    write_json(compat, compat_file)
    log("%s: module %s vs cache %s on %d panoramas (%.0f s): %d rows differ, max |diff| %.3g, %d failed -> %s"
        % (name, hc, c["hash"], n_cmp, time.time() - t0, n_bad, d, len(failed),
           "equivalent" if eq else "NOT equivalent"))
    if eq and relabel(name, c["hash"], hc, feat_dir):
        log("%s: cache relabelled %s -> %s (%d rows)" % (name, c["hash"], hc, len(c["ids"])))
    return eq


def missing_rows(info, recs, feat_dir=calibrate.FEAT_DIR, compat_files=None):
    """{module: number of streamed (image-less) records of recs without a row in its cache under the current
    module}."""
    recs = [r for r in recs if r.get("pano_deleted")]
    out = {}
    for n, (h, names) in info.items():
        c = calibrate.read_cache(n, feat_dir)
        ok = c is not None and c["names"] == names and calibrate.hash_ok(n, c["hash"], h, compat_files)
        have = set(c["ids"]) if ok else set()
        out[n] = sum(1 for r in recs if r["pano_id"] not in have)
    return out


def label_counts(dataset=calibrate.DATASET):
    from collections import Counter
    return Counter(r["label"] for r in calibrate.load_records(dataset) if calibrate.split_of(r) == "train")


def main():
    from engine import features
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=MAX_THREADS, help="download threads (<= %d)" % MAX_THREADS)
    ap.add_argument("--workers", type=int, default=MAX_WORKERS, help="feature processes (<= %d)" % MAX_WORKERS)
    ap.add_argument("--commit", type=int, default=500, help="panoramas per commit")
    ap.add_argument("--commit-s", type=float, default=900.0, help="commit at least every this many seconds")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many panoramas")
    ap.add_argument("--refresh", action="store_true", help="recompute missing features of streamed panoramas")
    ap.add_argument("--retry-skipped", action="store_true", help="try the rounds of stream_skipped.jsonl again")
    ap.add_argument("--verify-compat", action="store_true", help="verify caches of another module version")
    ap.add_argument("--n", type=int, default=100, help="--verify-compat: panoramas to re-extract")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    args.threads = max(1, min(args.threads, MAX_THREADS))
    args.workers = max(1, min(args.workers, MAX_WORKERS))
    names = list(features.MODULES)
    info = module_info(names)
    bad = cache_problems(info)
    if args.verify_compat:
        todo = relabeled_caches(info)
        if not todo:
            print("every cache is labelled with its module's current hash")
        ok = [verify_compat(n, args.n, args.workers) for n, _, _ in todo]
        if not all(ok):
            sys.exit(1)
        return
    if args.status:
        items, n_rounds = new_items(retry_skipped=args.retry_skipped)
        recs = calibrate.load_records()
        print("crawled rounds %d, to stream %d, streamed %d, skipped %d, records %d; caches %s; labels %s; "
              "problems %s; streamed records without a row %s"
              % (n_rounds, len(items), sum(1 for r in recs if r.get("pano_deleted")),
                 len(read_skips(calibrate.DATASET)), len(recs),
                 {n: len((calibrate.read_cache(n) or {"ids": []})["ids"]) for n in names},
                 relabeled_caches(info) or "current", bad, missing_rows(info, recs)))
        return
    if args.refresh:
        items, mods = refresh_items(info)
        if not items:
            print("every streamed panorama has features in every cache")
            return
        bad = [b for b in bad if b[0] in mods]
        info = {n: info[n] for n in mods}
        print("refresh: %d streamed panoramas miss features of %s" % (len(items), mods), flush=True)
    else:
        items, n_rounds = new_items(retry_skipped=args.retry_skipped)
        print("%d crawled rounds -> %d panoramas to stream" % (n_rounds, len(items)), flush=True)
    if bad:
        sys.exit("feature caches of another module version: %s.\nRun `python3 tools/calibrate.py features` first "
                 "(recomputes the panoramas with images; streamed ones then need `stream_duels.py --refresh`), or "
                 "`python3 tools/stream_duels.py --verify-compat` if the change cannot alter full-panorama features."
                 % ", ".join("%s (cache %s, module %s)" % b for b in bad))
    if args.limit:
        items = items[: args.limit] if not args.refresh else items
    if shutil.disk_usage(calibrate.DATASET).free < MIN_FREE_GB * 1e9:
        sys.exit("less than %.0f GB free on the disk" % MIN_FREE_GB)
    before = label_counts() if not args.refresh else None
    paths = [calibrate.FEAT_DIR, os.path.join(calibrate.DATASET, "index.jsonl"),
             os.path.join(calibrate.DATASET, SKIP_NAME)]
    disk0 = {p: path_bytes(p) for p in paths}
    stats = stream(items, args, info, refresh=args.refresh, log=lambda m: print(m, flush=True))
    disk1 = {p: path_bytes(p) for p in paths}
    print("summary", json.dumps(stats))
    print("disk growth: " + ", ".join("%s %+.1f MB" % (os.path.relpath(p, ROOT), (disk1[p] - disk0[p]) / 1e6)
                                       for p in disk0))
    if before is not None:
        after = label_counts()
        gain = sorted(((after[c] - before.get(c, 0), c) for c in after), reverse=True)
        print("train panoramas per country (+new): " + ", ".join("%s %d (+%d)" % (c, after[c], d)
                                                                   for d, c in gain[:40] if d > 0))
    if stats.get("stopped"):
        sys.exit(1)


if __name__ == "__main__":
    main()
