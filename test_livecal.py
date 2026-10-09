#!/usr/bin/env python3
"""
Tests for the live-play ("views") parameter set: python3 -m unittest test_livecal -v

engine.model.GeoModel.for_input / param_sets / input stamp, engine.regions.RegionModel.params_for /
regions_params.json, the input-kind switch of engine.locator.Locator, the live feature cache of
tools/live_features.py and the views calibration / evaluation / save code of tools/train_model.py.
"""
import contextlib
import io
import json
import os
import random
import shutil
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

from engine.geo import region_index
from engine.model import GeoModel, priors_hash
from engine.panorama import SphericalImage
from engine.regions import PARAMS_FILE, REGIONS_FILE, RegionModel, file_hash, region_posterior
from test_regions import PARAMS, synthetic

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "tools"))
PANO = os.path.join(ROOT, "test_japan_pano.jpg")
HAVE_MODEL = os.path.exists(os.path.join(ROOT, "data", "model", "model.json"))
WORLD_MAP = {"id": "w", "name": "A World", "world": True, "bounds": None, "maxErrorDistance": 14916862}


def tiny_geomodel(seed=0):
    rng = np.random.RandomState(seed)
    cls = ["AU", "BR", "JP"]
    ctr = {"AU": (3.0, 0.0), "BR": (0.0, 3.0), "JP": (-3.0, -3.0)}
    site = {"AU": (-25.0, 135.0), "BR": (-10.0, -50.0), "JP": (36.0, 138.0)}
    y = np.repeat(cls, 30)
    X = {"road": np.array([ctr[c] for c in y]) + rng.randn(len(y), 2), "texture": rng.randn(len(y), 3)}
    lat = np.array([site[c][0] for c in y]) + rng.randn(len(y))
    lng = np.array([site[c][1] for c in y]) + rng.randn(len(y))
    m = GeoModel().fit(X, y, lat, lng, np.ones(len(y), bool), min_count=5, knn_k=10)
    m.priors = {"alpha": 0.25, "ranked_world": {"counts": {"AU": 5, "BR": 50, "JP": 10}},
                "params": {"weights": dict(m.weights, glm=0.11), "prior_mix": 0.3,
                           "mix": {"map": 0.5, "map_ranked": 0.0, "ranked": 0.2}, "model": m.fingerprint()}}
    return m, X


def views_set(m):
    return {"model": m.fingerprint(), "weights": dict(m.weights, road=0.9, glm=0.2), "prior_mix": 0.85,
            "loc_params": {"bw": 0.5, "floor": 0.1}, "region_params": {"bw": 2.0, "floor": 0.3},
            "map": {"weights": dict(m.weights, glm=0.33), "prior_mix": 0.1,
                    "mix": {"map": 0.5, "map_ranked": 0.0, "ranked": 0.9}, "model": m.fingerprint()}}


class TestGeoModelSets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m, cls.X = tiny_geomodel()

    def test_no_views_set_is_pano(self):
        self.m.param_sets = {}
        self.assertIs(self.m.for_input("views"), self.m)
        self.assertIs(self.m.for_input("pano"), self.m)
        self.assertEqual(self.m.param_set, "pano")

    def test_views_set(self):
        m = self.m
        m.param_sets = {"views": views_set(m)}
        v = m.for_input("views")
        self.assertIsNot(v, m)
        self.assertIs(m.for_input("pano"), m)
        self.assertEqual(v.param_set, "views")
        self.assertEqual(v.weights["road"], 0.9)
        self.assertEqual(v.prior_mix, 0.85)
        self.assertEqual(v.loc_params, {"bw": 0.5, "floor": 0.1})
        self.assertEqual(v.region_params, {"bw": 2.0, "floor": 0.3})
        self.assertIs(v.ref_emb, m.ref_emb)  # the fitted arrays are shared
        # the model itself keeps the panorama parameters
        self.assertNotEqual(m.weights["road"], 0.9)
        self.assertEqual(m.priors["params"]["prior_mix"], 0.3)
        # no map: the set's exponents and prior; World-type map: the set's map-aware parameters
        np.testing.assert_allclose(v.map_setup(None)["prior"], v.base_prior(0.85))
        self.assertEqual(v.map_setup(None)["weights"]["road"], 0.9)
        sv, sp = v.map_setup(WORLD_MAP), m.map_setup(WORLD_MAP)
        self.assertEqual((sv["kind"], sp["kind"]), ("ranked", "ranked"))
        self.assertEqual(sv["weights"]["glm"], 0.33)
        self.assertEqual(sp["weights"]["glm"], 0.11)
        self.assertFalse(np.allclose(sv["prior"], sp["prior"]))
        # the posterior follows the set
        ev = m.evidence({g: x[:3] for g, x in self.X.items()})
        np.testing.assert_allclose(v.combine(ev), m.combine(ev, views_set(m)["weights"], 0.85))
        m.param_sets = {}

    def test_views_set_without_map_params_keeps_pano_map(self):
        m = self.m
        vs = views_set(m)
        vs.pop("map")
        m.param_sets = {"views": vs}
        self.assertEqual(m.for_input("views").map_setup(WORLD_MAP)["weights"]["glm"], 0.11)
        m.param_sets = {}

    def test_stale_views_set_ignored(self):
        m = self.m
        m.param_sets = {"views": dict(views_set(m), model="000000000000")}
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertIs(m.for_input("views"), m)
        m.param_sets = {}

    def test_input_stamp(self):
        m = self.m
        keys = ["model_fit", "priors_counts"]
        m.param_sets = {"views": dict(views_set(m), inputs=m.input_hashes(keys))}
        self.assertEqual(m.for_input("views").param_set, "views")
        old = m.priors
        try:  # new per-map / ranked counts (tools/build_priors.py): the set is stale
            m.priors = dict(old, ranked_world={"counts": {"AU": 6, "BR": 50, "JP": 10}})
            with contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertIs(m.for_input("views"), m)
            self.assertIn("priors_counts", err.getvalue())
            # the calibrated "params" are not part of the counts
            m.priors = dict(old, params={"prior_mix": 0.9})
            self.assertEqual(priors_hash(m.priors), priors_hash(old))
            self.assertEqual(m.for_input("views").param_set, "views")
        finally:
            m.priors = old
        # another fit (same classes and groups): the set is stale
        m.param_sets = {"views": dict(views_set(m), inputs=dict(m.input_hashes(keys), model_fit="000000000000"))}
        with contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertIs(m.for_input("views"), m)
        self.assertIn("model_fit", err.getvalue())
        m.param_sets = {}

    def test_fit_hash_survives_save_load(self):
        m = self.m
        tmp = tempfile.mkdtemp()
        try:
            h = m.fit_hash()
            m.save(tmp)
            self.assertEqual(json.load(open(os.path.join(tmp, "model.json")))["fit_hash"], h)
            self.assertEqual(GeoModel.load(tmp).fit_hash(), h)
            m2 = GeoModel.load(tmp)
            m2._fit_hash = None  # recomputed from the loaded arrays: the same
            self.assertEqual(m2.fit_hash(), h)
            m3, _ = tiny_geomodel(seed=1)
            self.assertNotEqual(m3.fit_hash(), h)
        finally:
            shutil.rmtree(tmp)

    def test_save_load(self):
        m = self.m
        m.param_sets = {"views": views_set(m)}
        tmp = tempfile.mkdtemp()
        try:
            m.save(tmp)
            json.dump(m.priors, open(os.path.join(tmp, "priors.json"), "w"))
            m2 = GeoModel.load(tmp)
            self.assertEqual(m2.param_sets, json.loads(json.dumps(m.param_sets)))
            self.assertEqual(m2.for_input("views").weights["road"], 0.9)
            m.param_sets = {}
            m.save(tmp)
            self.assertNotIn("param_sets", json.load(open(os.path.join(tmp, "model.json"))))
            self.assertIs(GeoModel.load(tmp).for_input("views").param_set, "pano")
        finally:
            shutil.rmtree(tmp)
            m.param_sets = {}


class TestRegionSets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, cc, lat, lng = synthetic()
        reg = region_index(lat, lng)
        cls.m = RegionModel(PARAMS).fit(X, cc, reg, lat, lng, np.arange(len(cc)) % 2 == 0)
        cls.tmp = tempfile.mkdtemp()
        cls.m.save(cls.tmp)
        cls.x = {"road": np.array([0.0, 0.0]), "texture": np.array([0.0])}
        cls.views = dict(PARAMS, weights={"prior": 0.2, "lda": 2.0, "knn": 0.0, "sun": 0.0}, smooth=0.3)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def write_sets(self, h):
        with open(os.path.join(self.tmp, PARAMS_FILE), "w") as f:
            json.dump({"views": {"params": self.views, "regions_npz": h}}, f)

    def test_without_file(self):
        p = os.path.join(self.tmp, PARAMS_FILE)
        if os.path.exists(p):
            os.remove(p)
        m = RegionModel.load(self.tmp)
        self.assertEqual(m.param_sets, {})
        self.assertIs(m.params_for("views"), m.params)

    def test_views_params(self):
        self.write_sets(file_hash(os.path.join(self.tmp, REGIONS_FILE)))
        m = RegionModel.load(self.tmp)
        pv = m.params_for("views")
        self.assertEqual(pv["weights"]["lda"], 2.0)
        self.assertEqual(pv["smooth"], 0.3)
        self.assertIs(m.params_for("pano"), m.params)
        # region_posterior with the views params = a model whose own parameters are the views set
        ref = RegionModel.load(self.tmp)
        ref.params = pv
        a = region_posterior(self.x, {"AU": 0.6, "NZ": 0.4}, m, params=pv)
        b = region_posterior(self.x, {"AU": 0.6, "NZ": 0.4}, ref)
        c = region_posterior(self.x, {"AU": 0.6, "NZ": 0.4}, m)
        self.assertEqual(sorted(a), sorted(b))
        for k in a:
            self.assertAlmostEqual(a[k], b[k], places=12)
        self.assertGreater(max(abs(a[k] - c[k]) for k in a), 1e-3)

    def test_stale_file_ignored(self):
        self.write_sets("000000000000")
        with contextlib.redirect_stderr(io.StringIO()):
            m = RegionModel.load(self.tmp)
        self.assertEqual(m.param_sets, {})
        os.remove(os.path.join(self.tmp, PARAMS_FILE))


class TestLiveFeatures(unittest.TestCase):
    def test_live_sphere_is_the_eval_live_capture(self):
        import live_features as lf
        from eval_live import render_views
        live = lf.live_sphere(PANO, 90.0, "pano-x")
        sph = SphericalImage.from_equirect(PANO, heading=90.0)
        ref = SphericalImage.from_views(render_views(sph, random.Random("pano-x").uniform(0, 360)), width=2048,
                                        heading=0.0)
        self.assertEqual(live.source, "views")
        self.assertIsNone(live.car_heading)
        self.assertEqual(live.heading, 0.0)
        np.testing.assert_array_equal(live.rgb, ref.rgb)
        self.assertGreater(live.coverage(), 0.9)  # the grid stops at pitch +-85

    def test_frame_sphere(self):
        import live_features as lf
        fr = lf.live_sphere(PANO, 90.0, "pano-x", "frame")
        self.assertEqual(fr.source, "views")
        self.assertTrue(0.1 < fr.coverage() < 0.2)  # one 112.7 x 90 deg view
        self.assertNotEqual(lf.render_hash("frame"), lf.render_hash("grid"))

    def test_cache_alignment_and_staleness(self):
        import live_features as lf
        from calibrate import module_hash
        from engine import features
        mod = features.load("road")
        tmp = tempfile.mkdtemp()
        old = lf.LIVE_DIR
        try:
            lf.LIVE_DIR = tmp
            d = len(mod.FEATURE_NAMES)
            rh = lf.render_hash("grid")
            np.savez_compressed(os.path.join(tmp, "road.npz"), ids=np.array(["b", "a"]),
                                X=np.stack([np.full(d, 2.0), np.full(d, 1.0)]).astype(np.float32),
                                names=np.array(mod.FEATURE_NAMES), hash=module_hash(mod), render=rh)
            M, have = lf.live_matrix("road", [{"pano_id": "a"}, {"pano_id": "c"}, {"pano_id": "b"}])
            self.assertEqual(list(have), [True, False, True])
            self.assertTrue((M[0] == 1.0).all() and np.isnan(M[1]).all() and (M[2] == 2.0).all())
            self.assertIsNone(lf.live_status(["road"]))
            self.assertIsNotNone(lf.live_status(["road", "texture"]))
            with self.assertRaises(FileNotFoundError):  # the frame cache is separate
                lf.live_matrix("road", [{"pano_id": "a"}], "frame")
            for kw in ({"hash": "stale", "render": rh}, {"hash": module_hash(mod), "render": "other"},
                       {"hash": module_hash(mod)}):  # module changed / rendering changed / no render hash
                np.savez_compressed(os.path.join(tmp, "road.npz"), ids=np.array(["a"]), X=np.ones((1, d), np.float32),
                                    names=np.array(mod.FEATURE_NAMES), **kw)
                with self.assertRaises(ValueError):
                    lf.live_matrix("road", [{"pano_id": "a"}])
                self.assertIsNotNone(lf.live_status(["road"]))
        finally:
            lf.LIVE_DIR = old
            shutil.rmtree(tmp)

    def test_live_cards_coverage(self):
        import live_features as lf

        class Bank:
            meta = {"det_hash": "h1"}

            def zt(self, z):
                return z

            def country_loglik(self, ZT, classes):
                return np.tile(ZT[:, :1], (1, len(classes))) * np.arange(len(classes))[None]

        tmp = tempfile.mkdtemp()
        old = lf.LIVE_DIR
        try:
            lf.LIVE_DIR = tmp

            def write(det="h1", render=lf.render_hash("grid")):
                np.savez_compressed(lf.cards_file(), ids=np.array(["a", "b"]), Z=np.array([[1.0], [2.0]], np.float32),
                                    coverage=np.array([1.0, 0.5], np.float32), det_hash=det, render=render)

            self.assertIsNotNone(lf.cards_status(Bank()))  # missing
            write()
            self.assertIsNone(lf.cards_status(Bank()))
            ll = lf.live_cards(["b", "a", "c"], ["X", "Y"], Bank())
            np.testing.assert_allclose(ll, [[0.0, 0.0], [0.0, 1.0], [0.0, 0.0]])  # b: coverage < 0.9; c missing
            for kw in ({"det": "other"}, {"render": "other"}):
                write(**kw)
                self.assertIsNotNone(lf.cards_status(Bank()))
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertIsNone(lf.live_cards(["a"], ["X"], Bank()))
            # the refit path asks for the cards too when a bank exists
            self.assertIsNotNone(lf.live_status([], bank=Bank()))
            self.assertIsNone(lf.live_status([]))
        finally:
            lf.LIVE_DIR = old
            shutil.rmtree(tmp)


def synthetic_rounds(m, n=60, seed=3):
    """Fake CALIB rounds of tiny_geomodel's countries: records (one game per 5 rounds, World-type map) and
    their 'live' features (the tiny model's feature distribution, noisier)."""
    rng = np.random.RandomState(seed)
    ctr = {"AU": (3.0, 0.0), "BR": (0.0, 3.0), "JP": (-3.0, -3.0)}
    site = {"AU": (-25.0, 135.0), "BR": (-10.0, -50.0), "JP": (36.0, 138.0)}
    lab = [m.classes[k % 3] for k in range(n)]
    recs = [{"pano_id": "p%d" % k, "label": c, "lat": site[c][0] + rng.randn(), "lng": site[c][1] + rng.randn(),
             "game": "g%d" % (k // 5), "map": "A Community World", "heading": 0.0} for k, c in enumerate(lab)]
    X = {"road": np.array([ctr[c] for c in lab]) + 2.0 * rng.randn(n, 2), "texture": rng.randn(n, 3)}
    return recs, X


class TestViewsCalibration(unittest.TestCase):
    """tools/train_model.py: calibrate_views leaves the panorama set alone, the save helpers write only what
    they own, the paired bootstrap."""

    def test_calibrate_views_keeps_pano_set(self):
        import copy
        import train_model as tm
        m, _ = tiny_geomodel()
        m.weights["cards"] = 0.05
        recs, X = synthetic_rounds(m)
        rows = np.arange(len(recs))
        ev = m.evidence(X)
        before = copy.deepcopy({k: getattr(m, k) for k in ("weights", "prior_mix", "loc_params", "region_params",
                                                            "priors")})
        with contextlib.redirect_stdout(io.StringIO()):
            ps, rp = tm.calibrate_views(m, ev, X, recs, rows)
        for k, v in before.items():
            self.assertEqual(getattr(m, k), v, k)
        self.assertIsNone(rp)
        self.assertIs(m.param_sets["views"], ps)
        self.assertEqual(ps["weights"]["cards"], 0.0)  # no card evidence: not tuned, so not used
        self.assertFalse(ps["calib"]["cards"])
        self.assertEqual(sorted(ps["inputs"]), ["model_fit", "priors_counts"])
        self.assertEqual(ps["map"]["mix"].keys(), {"map", "map_ranked", "ranked"})
        v = m.for_input("views")
        self.assertEqual(v.param_set, "views")
        self.assertEqual(v.weights, ps["weights"])
        self.assertEqual(v.loc_params, ps["loc_params"])
        # the pipeline evaluation runs with both sets and reports the guesses
        with contextlib.redirect_stdout(io.StringIO()):
            r = tm.evaluate_pipeline(v, ev, X, recs, rows, "t", use_map=True)
        self.assertEqual(r["params"], "views")
        self.assertEqual(len(r["per_round"]["lat"]), len(rows))
        self.assertGreater(r["top1"], 0.5)
        m.param_sets = {}

    def test_paired_diff_clusters(self):
        import train_model as tm
        rng = np.random.RandomState(0)
        a, b = rng.randn(50), rng.randn(50)
        r = tm.paired_diff(a, b, np.arange(50))
        self.assertEqual(r["ci95"], r["ci95_rounds"])  # one round per cluster: the round bootstrap
        self.assertAlmostEqual(r["diff"], float(np.mean(a - b)), places=4)
        # rounds of a game moving together: the game-level interval is wider
        d = np.repeat(rng.randn(10), 5)
        r = tm.paired_diff(d, np.zeros(50), np.repeat(np.arange(10), 5))
        self.assertEqual(r["clusters"], 10)
        self.assertGreater(r["ci95"][1] - r["ci95"][0], 1.5 * (r["ci95_rounds"][1] - r["ci95_rounds"][0]))

    def test_save_param_sets(self):
        import train_model as tm
        m, _ = tiny_geomodel()
        tmp = tempfile.mkdtemp()
        try:
            m.save(tmp)
            json.dump(m.priors, open(os.path.join(tmp, "priors.json"), "w"))
            mj = os.path.join(tmp, "model.json")
            meta0 = json.load(open(mj))
            loaded = GeoModel.load(tmp)
            expect = loaded.input_hashes(["model_fit", "priors_counts"])
            loaded.param_sets = {"views": views_set(loaded)}
            loaded.weights = dict(loaded.weights, road=0.99)  # not written: only the parameter sets are
            self.assertIsNone(tm.save_param_sets(loaded, expect, tmp))
            meta = json.load(open(mj))
            self.assertEqual(meta["param_sets"], json.loads(json.dumps(loaded.param_sets)))
            self.assertEqual({k: v for k, v in meta.items() if k != "param_sets"}, meta0)
            # another workstream refits meanwhile: refused, file untouched
            m2, _ = tiny_geomodel(seed=1)
            m2.save(tmp)
            meta2 = json.load(open(mj))
            why = tm.save_param_sets(loaded, expect, tmp)
            self.assertIn("model_fit", why)
            self.assertEqual(json.load(open(mj)), meta2)
        finally:
            shutil.rmtree(tmp)

    def test_save_region_views(self):
        import train_model as tm
        X, cc, lat, lng = synthetic()
        rm = RegionModel(PARAMS).fit(X, cc, region_index(lat, lng), lat, lng, np.arange(len(cc)) % 2 == 0)
        tmp = tempfile.mkdtemp()
        try:
            rm.save(tmp)
            f = os.path.join(tmp, PARAMS_FILE)
            rm.param_sets = {"views": dict(rm.params, smooth=0.3)}
            with contextlib.redirect_stdout(io.StringIO()):
                tm.save_region_views(rm, 10, tmp)
            self.assertEqual(RegionModel.load(tmp).params_for("views")["smooth"], 0.3)
            rm.param_sets = {"views": json.loads(json.dumps(rm.params))}  # calibration kept the pano parameters
            with contextlib.redirect_stdout(io.StringIO()):
                tm.save_region_views(rm, 10, tmp)
            self.assertFalse(os.path.exists(f))
        finally:
            shutil.rmtree(tmp)


@unittest.skipUnless(HAVE_MODEL, "no trained model (tools/train_model.py --save)")
class TestLocatorSwitch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from engine.locator import Locator
        cls.loc = Locator()

    def test_input_kind_picks_the_set(self):
        from engine.locator import VIEWS_MIN_COVERAGE, input_kind
        m = self.loc.model
        views = SphericalImage.from_screenshot(Image.open(PANO).crop((0, 100, 800, 500)), hfov=100.0,
                                               true_north=True)
        self.assertLess(views.coverage(), VIEWS_MIN_COVERAGE)
        res = self.loc.analyze(views)
        self.assertEqual(res["input"]["params"], m.for_input(input_kind(views)).param_set)
        import live_features as lf
        grid = lf.live_sphere(PANO, 90.0, "pano-x")
        self.assertEqual(input_kind(grid), "views")
        pano = SphericalImage.from_equirect(Image.open(PANO).convert("RGB").resize((1024, 512)), heading=0.0,
                                            width=1024)
        self.assertEqual(input_kind(pano), "pano")
        self.assertEqual(self.loc.analyze(pano)["input"]["params"], "pano")
        self.assertEqual(self.loc.analyze(pano)["input"]["region_params"], "pano")
        self.assertIs(m.for_input("pano"), m)

    def test_pipeline_matches_locator(self):
        """train_model.evaluate_pipeline on the features of a live-grid capture gives the locator's guess and
        posterior (no map and with a World-type map)."""
        import live_features as lf
        import train_model as tm
        from engine.locator import input_kind
        from engine.regions import load_region_model
        sph = lf.live_sphere(PANO, 90.0, "pano-x")
        kind = input_kind(sph)
        vm = self.loc.model.for_input(kind)
        X = {g: np.asarray(mod.extract(sph)["x"], np.float64)[None, :] for g, mod in self.loc.modules.items()}
        cards = None
        if self.loc.cards is not None:
            det = self.loc.cards.detect(sph)
            det["zt"] = self.loc.cards.zt(det["z"])
            cards = self.loc.cards.country_evidence(det, vm.classes)
        ev = vm.evidence(X, cards_ll=cards)
        rmodel = load_region_model()
        rec = {"pano_id": "pano-x", "label": "JP", "lat": 35.0, "lng": 139.0, "game": "g", "map": "A Community World"}
        for use_map in (False, True):
            res = self.loc.analyze(sph, map_info=rec["map"] if use_map else None)
            with contextlib.redirect_stdout(io.StringIO()):
                r = tm.evaluate_pipeline(vm, ev, X, [rec], [0], "t", use_map, rmodel,
                                         rmodel.params_for(kind) if rmodel is not None else None)
            pr = r["per_round"]
            self.assertAlmostEqual(round(float(pr["lat"][0]), 5), res["guess"]["lat"], places=5)
            self.assertAlmostEqual(round(float(pr["lng"][0]), 5), res["guess"]["lng"], places=5)
            self.assertAlmostEqual(float(pr["true_p"][0]), res["posterior"]["JP"], delta=1e-3 * res["posterior"]["JP"])


if __name__ == "__main__":
    unittest.main()
