#!/usr/bin/env python3
"""
Tests for streaming duel panoramas into the feature caches without keeping the images
(tools/stream_duels.py, the image-less records / cache plumbing of tools/calibrate.py,
tools/build_dataset.duel_candidates, tools/crawl_duels.py):
    python3 -m unittest test_stream -v
"""
import argparse
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import build_dataset  # noqa: E402
import calibrate  # noqa: E402
import crawl_duels  # noqa: E402
import stream_duels  # noqa: E402
from build_dataset import duel_candidates  # noqa: E402


_OWN = calibrate.OWN_FILE


def setUpModule():
    calibrate.OWN_FILE = os.devnull  # no own rounds unless a test passes own_file (the real ones would exclude)


def tearDownModule():
    calibrate.OWN_FILE = _OWN


def write_index(dataset, recs, tail=""):
    os.makedirs(os.path.join(dataset, "panos"), exist_ok=True)
    with open(os.path.join(dataset, "index.jsonl"), "w") as f:
        f.write("".join(json.dumps(r) + "\n" for r in recs) + tail)


def synthetic_pano(seed):
    """2048 x 1024 panorama: blue sky, grey road band, green verge, a little noise."""
    rng = np.random.RandomState(seed)
    img = np.zeros((1024, 2048, 3), np.uint8)
    img[:480] = (90, 140, 220)
    img[480:700] = (60, 130, 50)
    img[700:] = (110, 110, 105)
    img[700:, 1000:1048] = (230, 230, 230)
    img = np.clip(img.astype(int) + rng.randint(-12, 13, img.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def meta_of(pid, lat=50.45, lng=30.52, cc="UA"):
    return {"pano_id": pid, "size": [832, 1664], "levels": [[416, 832], [832, 1664]], "tile_size": [512, 512],
            "lat": lat, "lng": lng, "heading": 37.5, "tilt": 90.0, "roll": 0.0, "country_code": cc,
            "date": [2021, 6], "camera": "gen4"}


class TestRecords(unittest.TestCase):
    def setUp(self):
        self.ds = tempfile.mkdtemp()
        write_index(self.ds, [dict(meta_of("a"), mode="duel"), dict(meta_of("b"), mode="duel"),
                              dict(meta_of("c"), mode="duel", pano_deleted=True)])
        Image.new("RGB", (8, 4)).save(os.path.join(self.ds, "panos", "a.jpg"))

    def tearDown(self):
        shutil.rmtree(self.ds)

    def test_image_less_records(self):
        self.assertEqual([r["pano_id"] for r in calibrate.load_records(self.ds)], ["a", "c"])
        self.assertEqual([r["pano_id"] for r in calibrate.load_records(self.ds, pixels=True)], ["a"])
        recs = {r["pano_id"]: r for r in calibrate.load_records(self.ds)}
        self.assertTrue(calibrate.has_pixels(recs["a"], self.ds))
        self.assertFalse(calibrate.has_pixels(recs["c"], self.ds))
        self.assertEqual(recs["c"]["label"], "UA")

    def test_own_rounds_exclude_training_records(self):
        own = os.path.join(self.ds, "own.json")
        json.dump([{"game": "G", "round": 1, "pano_id": "ownpano", "lat": 10.0, "lng": 20.0}], open(own, "w"))
        recs = [dict(meta_of("near", 10.005, 20.0), mode="duel", pano_deleted=True),        # ~0.56 km
                dict(meta_of("far", 10.02, 20.0), mode="duel", pano_deleted=True),          # ~2.2 km
                dict(meta_of("rnd", 0.0, 0.0), mode="duel", pano_deleted=True, round_lat=10.0, round_lng=20.004),
                dict(meta_of("wld", 10.0, 20.009), mode="world"),                           # ~0.99 km
                dict(meta_of("hist", -5.0, 30.0), mode="history", game="H", round=2),
                dict(meta_of("nearhist", -5.0, 30.003), mode="duel", pano_deleted=True),    # own index record
                dict(meta_of("both", 40.0, 40.0), mode="duel", pano_deleted=True),
                dict(meta_of("both", 40.0, 40.0), mode="history", game="H", round=3)]
        write_index(self.ds, recs)
        for p in ("wld", "hist", "both"):
            Image.new("RGB", (8, 4)).save(calibrate.pano_path(p, self.ds))
        got = {r["pano_id"]: r for r in calibrate.load_records(self.ds, own_file=own)}
        self.assertEqual(sorted(got), ["both", "far", "hist"])
        self.assertEqual(got["both"]["mode"], "history")  # the user's round wins over a training record
        every = calibrate.load_records(self.ds, own_file=own, exclude_own=False)
        self.assertEqual(calibrate.near_own(every, own), {"near", "rnd", "wld", "nearhist"})

    def test_extract_one_names_the_missing_image(self):
        with self.assertRaises(FileNotFoundError) as cm:
            calibrate._extract_one((calibrate.pano_path("c", self.ds), 0.0))
        self.assertIn("pano_deleted", str(cm.exception))

    def test_index_append_after_partial_line(self):
        write_index(self.ds, [dict(meta_of("a"), mode="duel")], tail='{"pano_id": "broken", "lat"')
        stream_duels.append_index([dict(meta_of("d"), mode="duel", pano_deleted=True)], self.ds)
        ids = [r["pano_id"] for r in calibrate.load_records(self.ds)]
        self.assertEqual(ids, ["a", "d"])


class TestCaches(unittest.TestCase):
    def setUp(self):
        self.fd = tempfile.mkdtemp()
        self.names = {"road": ["x0", "x1"], "solar": ["s0"]}
        calibrate.write_cache("road", ["a", "b"], np.array([[1, 2], [3, 4]]), self.names["road"], "h1", self.fd)
        calibrate.write_cache("solar", ["a", "b"], np.array([[5], [6]]), self.names["solar"], "s1", self.fd)
        self.compat = os.path.join(self.fd, "compat.json")

    def tearDown(self):
        shutil.rmtree(self.fd)

    def test_upsert_appends_and_replaces(self):
        rows = {"road": {"c": [7, 8], "a": [9, 9]}, "solar": {"c": [1], "a": [2]}}
        sizes = calibrate.upsert_features(rows, {"road": "h1", "solar": "s1"}, self.names, self.fd, [self.compat])
        self.assertEqual(sizes, {"road": 3, "solar": 3})
        c = calibrate.read_cache("road", self.fd)
        self.assertEqual(c["ids"], ["a", "b", "c"])
        np.testing.assert_array_equal(c["X"], [[9, 9], [3, 4], [7, 8]])
        calibrate.upsert_features(rows, {"road": "h1", "solar": "s1"}, self.names, self.fd, [self.compat])
        self.assertEqual(calibrate.read_cache("road", self.fd)["ids"], ["a", "b", "c"])  # idempotent
        self.assertEqual(sorted(f for f in os.listdir(self.fd) if "tmp" in f), [])

    def test_mismatch_writes_nothing(self):
        rows = {"road": {"c": [7, 8]}, "solar": {"c": [1]}}
        with self.assertRaises(calibrate.CacheMismatch):
            calibrate.upsert_features(rows, {"road": "h1", "solar": "OTHER"}, self.names, self.fd, [self.compat])
        self.assertEqual(calibrate.read_cache("road", self.fd)["ids"], ["a", "b"])
        with self.assertRaises(calibrate.CacheMismatch):  # feature names differ
            calibrate.upsert_features({"road": {"c": [7]}}, {"road": "h1"}, {"road": ["x0"]}, self.fd, [self.compat])

    def test_verified_equivalent_hash_keeps_the_label(self):
        json.dump({"road": [{"module": "h2", "cache": "h1", "equivalent": True}],
                   "solar": [{"extra": "s2", "main": "s1", "equivalent": False}]}, open(self.compat, "w"))
        self.assertTrue(calibrate.hash_ok("road", "h1", "h2", [self.compat]))
        self.assertFalse(calibrate.hash_ok("road", "h2", "h1", [self.compat]))
        self.assertFalse(calibrate.hash_ok("solar", "s1", "s2", [self.compat]))
        calibrate.upsert_features({"road": {"c": [7, 8]}}, {"road": "h2"}, self.names, self.fd, [self.compat])
        self.assertEqual(calibrate.read_cache("road", self.fd)["hash"], "h1")

    def test_clean_tmp(self):
        dead = os.path.join(self.fd, "road.npz.tmp999999999")  # no such process
        live = os.path.join(self.fd, "solar.npz.tmp%d" % os.getppid())
        other = os.path.join(self.fd, "notes.tmp1")
        for f in (dead, live, other):
            open(f, "w").write("x")
        with calibrate.cache_lock(self.fd):
            calibrate.clean_tmp(self.fd, log=lambda m: None)
        self.assertEqual([os.path.exists(f) for f in (dead, live, other)], [False, True, True])

    def test_keep_stale(self):
        c = calibrate.read_cache("road", self.fd)
        path = calibrate.keep_stale("road", c, ["b", "zz"], self.fd)
        s = calibrate.read_cache("stale/road.h1", self.fd)
        self.assertTrue(os.path.exists(path))
        self.assertEqual(s["ids"], ["b"])
        np.testing.assert_array_equal(s["X"], [[3, 4]])


class TestCmdFeatures(unittest.TestCase):
    """`calibrate.py features` keeps the rows of image-less records and never recomputes them."""

    def setUp(self):
        self.ds, self.fd = tempfile.mkdtemp(), tempfile.mkdtemp()
        # img1 / img2: both car-axis regimes (calibrate.hide_car_axis)
        write_index(self.ds, [dict(meta_of("img1"), mode="duel"), dict(meta_of("img2"), mode="duel"),
                              dict(meta_of("gone"), mode="duel", pano_deleted=True)])
        self.assertNotEqual(calibrate.hide_car_axis("img1"), calibrate.hide_car_axis("img2"))
        for k, p in enumerate(("img1", "img2")):
            synthetic_pano(k + 1).save(calibrate.pano_path(p, self.ds), quality=90)
        from engine.features import vehicle
        self.mod = vehicle
        self.h = calibrate.module_hash(vehicle)
        self.d = len(vehicle.FEATURE_NAMES)
        self.args = argparse.Namespace(modules="vehicle", force=False, sample=0, workers=1)
        self.min_rows = calibrate.MIN_EQUIV_ROWS
        calibrate.MIN_EQUIV_ROWS = 2
        calibrate._init(["vehicle"])
        self.true = {p: calibrate._extract_one((calibrate.pano_path(p, self.ds), 37.5))[0] for p in ("img1", "img2")}

    def tearDown(self):
        calibrate.MIN_EQUIV_ROWS = self.min_rows
        shutil.rmtree(self.ds)
        shutil.rmtree(self.fd)

    def run_features(self):
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            self.status = calibrate.cmd_features(self.args, self.ds, self.fd)
        return out.getvalue()

    def cache_with(self, label, img_rows=None):
        """vehicle cache: img1, img2 (the true features, or img_rows) and the image-less 'gone' (7.0)."""
        X = np.stack([self.true["img1"], self.true["img2"], np.full(self.d, 7.0, np.float32)])
        if img_rows is not None:
            X[:2] = img_rows
        calibrate.write_cache("vehicle", ["img1", "img2", "gone"], X, self.mod.FEATURE_NAMES, label, self.fd)

    def assert_dropped(self, log, label):
        self.assertEqual(self.status, 3)
        self.assertIn("lose their vehicle features", log)
        self.assertIn("stream_duels.py --refresh", log)
        c = calibrate.read_cache("vehicle", self.fd)
        self.assertEqual((c["ids"], c["hash"]), (["img1", "img2"], self.h))
        s = calibrate.read_cache("stale/vehicle." + label, self.fd)
        self.assertEqual(s["ids"], ["gone"])
        np.testing.assert_array_equal(s["X"][0], np.full(self.d, 7.0))
        info = {"vehicle": (self.h, list(self.mod.FEATURE_NAMES))}
        items, mods = stream_duels.refresh_items(info, self.ds, self.fd, compat_files=[])
        self.assertEqual(([r["pano_id"] for r in items], mods), (["gone"], ["vehicle"]))

    def test_force_differs_drops_image_less_rows(self):
        self.cache_with(self.h, img_rows=np.full((2, self.d), 7.0))
        self.args.force = True
        self.assert_dropped(self.run_features(), self.h)

    def test_force_reproduced_keeps_image_less_rows(self):
        self.cache_with(self.h)
        self.args.force = True
        log = self.run_features()
        self.assertEqual(self.status, 0)
        self.assertIn("equal the cached ones on all 2 panoramas", log)
        c = calibrate.read_cache("vehicle", self.fd)
        self.assertEqual((c["ids"], c["hash"]), (["gone", "img1", "img2"], self.h))
        np.testing.assert_array_equal(c["X"][0], np.full(self.d, 7.0))

    def test_changed_module_reproduced_keeps_rows(self):
        self.cache_with("old")  # e.g. a comment edited: other hash, same features
        log = self.run_features()
        self.assertEqual(self.status, 0, log)
        c = calibrate.read_cache("vehicle", self.fd)
        self.assertEqual((c["ids"], c["hash"]), (["gone", "img1", "img2"], self.h))
        self.assertFalse(os.path.exists(os.path.join(self.fd, "stale")))

    def test_too_few_compared_rows(self):
        calibrate.MIN_EQUIV_ROWS = 3
        self.cache_with("old")
        self.assert_dropped(self.run_features(), "old")

    def test_current_cache(self):
        calibrate.write_cache("vehicle", ["gone"], np.ones((1, self.d)), self.mod.FEATURE_NAMES, self.h, self.fd)
        log = self.run_features()
        self.assertIn("2 to process", log)
        c = calibrate.read_cache("vehicle", self.fd)
        self.assertEqual(c["ids"], ["gone", "img1", "img2"])  # load_records order (pano id)
        np.testing.assert_array_equal(c["X"][0], np.ones(self.d))
        np.testing.assert_array_equal(c["X"][1], self.true["img1"])

    def test_changed_module(self):
        self.cache_with("old", img_rows=np.ones((2, self.d)))
        self.assert_dropped(self.run_features(), "old")


class TestCandidates(unittest.TestCase):
    def test_duel_candidates(self):
        own = [{"pano_id": "own", "lat": 50.0, "lng": 30.0}]
        rounds = [{"pano_id": "own", "lat": 10.0, "lng": 10.0},      # the user's own panorama
                  {"pano_id": "near", "lat": 50.005, "lng": 30.0},   # ~0.56 km from an own round
                  {"pano_id": "far", "lat": 50.02, "lng": 30.0},     # ~2.2 km
                  {"pano_id": "far", "lat": 50.02, "lng": 30.0},     # same panorama again
                  {"pano_id": "seen", "lat": 0.0, "lng": 0.0},
                  {"pano_id": "skip", "lat": 0.0, "lng": 1.0},
                  {"pano_id": "new", "lat": -30.0, "lng": 20.0}]
        keep = duel_candidates(rounds, own, seen={"seen"}, skip={"skip"})
        self.assertEqual([r["pano_id"] for r in keep], ["far", "new"])

    def test_rounds_without_pano_id(self):
        rounds = [{"pano_id": None, "game": "g", "round": 1, "lat": 1.0, "lng": 1.0},
                  {"pano_id": None, "game": "g", "round": 2, "lat": 2.0, "lng": 2.0},
                  {"pano_id": None, "game": "g", "round": 3, "lat": 3.0, "lng": 3.0},
                  {"pano_id": None, "game": "g", "round": 4, "lat": 50.0, "lng": 30.0}]  # an own location
        own = [{"pano_id": None, "lat": 50.0, "lng": 30.0}]
        keep = duel_candidates(rounds, own, skip={"g/3"})
        self.assertEqual([r["round"] for r in keep], [1, 2])
        ds = tempfile.mkdtemp()
        try:
            write_index(ds, [])
            dump = lambda o, n: (json.dump(o, open(os.path.join(ds, n), "w")), os.path.join(ds, n))[1]
            items, n = stream_duels.new_items(ds, duels_file=dump({"rounds": rounds}, "d.json"),
                                              own_file=dump(own, "o.json"))
            self.assertEqual((sorted(r["round"] for r in items), n), ([1, 2, 3], 4))
            stream_duels.add_skip(ds, rounds[0], "no panorama", __import__("threading").Lock())
            self.assertEqual(stream_duels.read_skips(ds), {"g/1"})
        finally:
            shutil.rmtree(ds)

    def test_disk_guard_any_n(self):
        ds = tempfile.mkdtemp()
        old = build_dataset.DUELS_FILE, build_dataset.OWN_FILE
        try:
            rounds = [{"pano_id": "p%d" % i, "game": "g%d" % i, "round": 1, "lat": i * 0.05 - 60, "lng": 0.0}
                      for i in range(1500)]
            build_dataset.DUELS_FILE = os.path.join(ds, "d.json")
            build_dataset.OWN_FILE = os.path.join(ds, "o.json")
            json.dump({"rounds": rounds}, open(build_dataset.DUELS_FILE, "w"))
            json.dump([], open(build_dataset.OWN_FILE, "w"))
            for n in (0, 2000, 1200):
                args = argparse.Namespace(n=n, keep_images=False, threads=1, out=ds)
                with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                    build_dataset.run_duels(args, {})
        finally:
            build_dataset.DUELS_FILE, build_dataset.OWN_FILE = old
            shutil.rmtree(ds)

    def test_merge_own_rounds(self):
        ds = tempfile.mkdtemp()
        try:
            path = os.path.join(ds, "own.json")
            json.dump([{"game": "a", "round": 1, "lat": 1}, {"game": "a", "round": 2, "lat": 2}], open(path, "w"))
            n = build_dataset.merge_own_rounds([{"game": "a", "round": 2, "lat": 9}, {"game": "b", "round": 1}], path)
            self.assertEqual(n, 1)
            got = json.load(open(path))
            self.assertEqual([(r["game"], r["round"], r.get("lat")) for r in got],
                             [("a", 1, 1), ("a", 2, 9), ("b", 1, None)])
        finally:
            shutil.rmtree(ds)


class TestCrawl(unittest.TestCase):
    def history(self, k):
        return {"entries": [{"gameId": "g%d" % k, "players": [{"id": "p%d" % k}],
                             "duel": {"gameMode": "NoMoveDuels",
                                      "rounds": [{"roundNumber": 1, "correctLat": 1.0, "correctLng": 2.0,
                                                  "panoId": "x%d" % k, "correctCountryCode": "ua"}]}}]}

    def test_stop_at_first_429_and_throttle(self):
        calls, t, sleeps, saved = [], [0.0], [], []

        def get(url):
            calls.append(url)
            t[0] += 0.2
            return (429, None) if len(calls) == 3 else (200, self.history(len(calls)))
        th = crawl_duels.Throttle(1.5, clock=lambda: t[0], sleep=lambda d: (sleeps.append(d), t.__setitem__(0, t[0] + d)))
        state = {"rounds": [], "games": {}, "visited": ["me"], "queue": [["a", "a"]]}
        reason = crawl_duels.crawl(state, 10, throttle=th, get=get, save=lambda s: saved.append(1), log=lambda m: None)
        self.assertEqual(reason, "HTTP 429")
        self.assertEqual(len(calls), 3)  # nothing after the 429
        self.assertTrue(all(abs(d - 1.3) < 1e-9 for d in sleeps))  # >= 1.5 s start to start
        self.assertEqual(state["queue"][0], ["p2", "p2"])  # the rate-limited player is read next time
        self.assertEqual([r["gg_country"] for r in state["rounds"]], ["UA", "UA"])
        self.assertTrue(saved)

    def run_crawl(self, statuses, queue=("a", "b", "c", "d", "e", "f", "g", "h")):
        calls = []

        def get(url):
            calls.append(url.rsplit("/", 1)[1])
            st = statuses[len(calls) - 1] if len(calls) <= len(statuses) else 200
            return st, (self.history(len(calls)) if st == 200 else None)
        th = crawl_duels.Throttle(1.5, clock=lambda: 0.0, sleep=lambda d: None)
        state = {"rounds": [], "games": {}, "visited": ["me"], "queue": [[q, q] for q in queue]}
        reason = crawl_duels.crawl(state, 3, throttle=th, get=get, save=lambda s: None, log=lambda m: None)
        return reason, calls, state

    def test_stop_at_401(self):
        reason, calls, state = self.run_crawl([200, 401])
        self.assertEqual((reason, calls), ("HTTP 401", ["a", "b"]))
        self.assertEqual(state["queue"][0], ["b", "b"])
        self.assertNotIn("b", state["visited"])

    def test_failing_statuses(self):
        # 503: player to the back of the queue, not visited; 404: visited; 5 failures in a row: stop
        reason, calls, state = self.run_crawl([503, 404, 200, 403, 502, 500, 503, 500])
        self.assertEqual(reason, "HTTP 500 x5")
        self.assertEqual(calls, ["a", "b", "c", "d", "e", "f", "g", "h"])
        self.assertIn("b", state["visited"])
        self.assertNotIn("a", state["visited"])
        self.assertIn(["a", "a"], state["queue"])
        reason, calls, state = self.run_crawl([503] + [200] * 7, queue=("a",))
        self.assertEqual((reason, calls[:2]), ("done", ["a", "a"]))  # a is read again at the back of the queue
        self.assertIn("a", state["visited"])
        reason, calls, state = self.run_crawl([503, 503], queue=("a",))
        self.assertEqual((reason, calls), ("queue empty", ["a", "a"]))  # second failure: given up
        self.assertIn("a", state["visited"])

    def test_sleep_floor(self):
        self.assertGreaterEqual(crawl_duels.Throttle(max(0.1, crawl_duels.MIN_SLEEP)).gap, 1.5)


class TestStream(unittest.TestCase):
    """Download -> features (real modules, worker process) -> commit, with a fake Street View."""
    MODS = ["road", "vehicle"]

    def setUp(self):
        self.ds, self.fd = tempfile.mkdtemp(), tempfile.mkdtemp()
        write_index(self.ds, [dict(meta_of("old1"), mode="duel")])
        Image.new("RGB", (8, 4)).save(calibrate.pano_path("old1", self.ds))
        self.info = stream_duels.module_info(self.MODS)
        for n, (h, names) in self.info.items():
            calibrate.write_cache(n, ["old1"], np.zeros((1, len(names))), names, h, self.fd)
        self.args = argparse.Namespace(threads=2, workers=1, commit=2, commit_s=900.0)
        self.metas = {"r1": meta_of("p1"), "r2": None, "r3": meta_of("p3", -33.9, 18.4, "ZA"), "r4": meta_of("p1")}

    def tearDown(self):
        shutil.rmtree(self.ds)
        shutil.rmtree(self.fd)

    def expected(self, pid, seed):
        """Features of the same image through calibrate._extract_one in this process."""
        d = tempfile.mkdtemp()
        try:
            path = os.path.join(d, pid + ".jpg")
            synthetic_pano(seed).save(path, quality=90)
            calibrate._init(self.MODS)
            return calibrate._extract_one((path, 37.5))
        finally:
            shutil.rmtree(d)

    def test_stream_commits_features_and_deletes_images(self):
        rounds = [{"pano_id": k, "game": "g", "round": i + 1, "lat": 1.0, "lng": 2.0, "gg_country": "UA",
                   "gg_heading": 10, "mode": "NoMoveDuels", "start": "2026-10-01"} for i, k in enumerate(self.metas)]
        # an image of an interrupted run (downloaded, not committed) and one that was committed already
        stage = stream_duels.stage_dir(self.ds)
        os.makedirs(stage)
        rec5 = stream_duels.make_record(meta_of("p5"), dict(rounds[0], pano_id="r5"))
        synthetic_pano(5).save(os.path.join(stage, "p5.jpg"), quality=90)
        json.dump(rec5, open(os.path.join(stage, "p5.json"), "w"))
        synthetic_pano(6).save(os.path.join(stage, "old1.jpg"), quality=90)
        json.dump(dict(meta_of("old1"), mode="duel"), open(os.path.join(stage, "old1.json"), "w"))
        seeds = {"p1": 1, "p3": 3}
        logs = []
        stats = stream_duels.stream(rounds, self.args, self.info, self.ds, self.fd,
                                    fetch_meta=lambda r: self.metas[r["pano_id"]],
                                    download=lambda m: synthetic_pano(seeds[m["pano_id"]]),
                                    compat_files=[], log=logs.append)
        self.assertEqual(stats["committed"], 3, logs)
        self.assertEqual(stats["skipped"], 2)  # r2: no panorama, r4: duplicate of p1
        self.assertEqual(os.listdir(stage), [])
        recs = {r["pano_id"]: r for r in calibrate.load_records(self.ds)}
        self.assertEqual(sorted(recs), ["old1", "p1", "p3", "p5"])
        self.assertTrue(recs["p3"]["pano_deleted"])
        self.assertEqual((recs["p3"]["label"], recs["p3"]["mode"], recs["p3"]["round"]), ("ZA", "duel", 3))
        skips = stream_duels.read_skips(self.ds)
        self.assertEqual(skips, {"r2", "r4"})
        for j, n in enumerate(self.MODS):
            c = calibrate.read_cache(n, self.fd)
            self.assertEqual(sorted(c["ids"]), ["old1", "p1", "p3", "p5"])
            self.assertEqual(c["hash"], self.info[n][0])
        for pid, seed in (("p1", 1), ("p3", 3), ("p5", 5)):
            want = self.expected(pid, seed)
            for j, n in enumerate(self.MODS):
                c = calibrate.read_cache(n, self.fd)
                np.testing.assert_array_equal(c["X"][c["ids"].index(pid)], want[j])
        # the next run has nothing left; the records survive a features run
        items, _ = stream_duels.new_items(self.ds, duels_file=self._dump({"rounds": rounds}),
                                          own_file=self._dump([]))
        self.assertEqual(items, [])
        refresh, mods = stream_duels.refresh_items(self.info, self.ds, self.fd, compat_files=[])
        self.assertEqual((refresh, mods), ([], []))

    def test_refresh_after_module_change(self):
        write_index(self.ds, [dict(meta_of("old1"), mode="duel"),
                              dict(meta_of("p9"), mode="duel", pano_deleted=True)])
        items, mods = stream_duels.refresh_items(self.info, self.ds, self.fd, compat_files=[])
        self.assertEqual(([r["pano_id"] for r in items], mods), (["p9"], self.MODS))
        stats = stream_duels.stream(items, self.args, self.info, self.ds, self.fd, refresh=True,
                                    download=lambda m: synthetic_pano(9), compat_files=[], log=lambda m: None)
        self.assertEqual(stats["committed"], 1)
        self.assertEqual(stream_duels.refresh_items(self.info, self.ds, self.fd, compat_files=[])[0], [])
        with open(os.path.join(self.ds, "index.jsonl")) as f:
            self.assertEqual(len(f.readlines()), 2)  # refresh does not append records

    def test_stale_cache_stops_before_writing(self):
        n = self.MODS[0]
        calibrate.write_cache(n, ["old1"], np.zeros((1, len(self.info[n][1]))), self.info[n][1], "other", self.fd)
        self.assertEqual(stream_duels.cache_problems(self.info, self.fd, compat_files=[]),
                         [(n, "other", self.info[n][0])])
        rounds = [{"pano_id": "r1", "game": "g", "round": 1, "lat": 1.0, "lng": 2.0, "gg_country": "UA"}]
        stats = stream_duels.stream(rounds, self.args, self.info, self.ds, self.fd,
                                    fetch_meta=lambda r: meta_of("p1"), download=lambda m: synthetic_pano(1),
                                    compat_files=[], log=lambda m: None)
        self.assertIn("stopped", stats)
        self.assertEqual([r["pano_id"] for r in calibrate.load_records(self.ds)], ["old1"])
        self.assertTrue(os.path.exists(os.path.join(stream_duels.stage_dir(self.ds), "p1.jpg")))  # kept for later

    def test_verify_compat_relabels(self):
        write_index(self.ds, [dict(meta_of("img1"), mode="duel"), dict(meta_of("img2"), mode="duel"),
                              dict(meta_of("gone"), mode="duel", pano_deleted=True)])
        for k, p in enumerate(("img1", "img2")):
            synthetic_pano(k + 1).save(calibrate.pano_path(p, self.ds), quality=90)
        calibrate._init(["vehicle"])
        X = np.stack([calibrate._extract_one((calibrate.pano_path(p, self.ds), 37.5))[0] for p in ("img1", "img2")]
                     + [np.zeros(len(self.info["vehicle"][1]), np.float32)])
        names, h = self.info["vehicle"][1], self.info["vehicle"][0]
        compat = os.path.join(self.fd, "compat.json")
        kw = dict(dataset=self.ds, feat_dir=self.fd, compat_file=compat, log=lambda m: None)
        bad = X.copy()
        bad[1, 0] += 1.0
        calibrate.write_cache("vehicle", ["img1", "img2", "gone"], bad, names, "old", self.fd)
        self.assertFalse(stream_duels.verify_compat("vehicle", 2, 1, **kw))
        self.assertEqual(calibrate.read_cache("vehicle", self.fd)["hash"], "old")
        calibrate.write_cache("vehicle", ["img1", "img2", "gone"], X, names, "old", self.fd)
        self.assertEqual(stream_duels.relabeled_caches({"vehicle": self.info["vehicle"]}, self.fd),
                         [("vehicle", "old", h)])
        self.assertTrue(stream_duels.verify_compat("vehicle", 2, 1, **kw))
        c = calibrate.read_cache("vehicle", self.fd)
        self.assertEqual((c["hash"], c["ids"]), (h, ["img1", "img2", "gone"]))
        e = json.load(open(compat))["vehicle"]
        self.assertEqual([(x["module"], x["cache"], x["equivalent"], x["n"], x["regimes"]) for x in e],
                         [(h, "old", True, 2, [0, 1])])  # the failed check of the same pair is replaced

    def _dump(self, obj):
        path = os.path.join(self.ds, "tmp_%d.json" % len(os.listdir(self.ds)))
        with open(path, "w") as f:
            json.dump(obj, f)
        return path


if __name__ == "__main__":
    unittest.main()
