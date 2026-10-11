#!/usr/bin/env python3
"""
Tests of the country workstream: the official World map's own prior (per-map counts from the locator's
world_calib games, its own mixture weights), the World calibration helpers of tools/train_model.py and the
flag-detector spike (engine/flag_detect.py).

  python3 -m unittest test_country -v
"""
import hashlib
import json
import math
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from engine.geo import load_maps, resolve_map  # noqa: E402
from engine.model import GeoModel, mix_prior  # noqa: E402

WORLD_ID = "52d15ff6850ffb3b847cb3b4"
HAVE_MAPS = bool(load_maps())


def toy_model(classes=("UA", "PL", "US"), prior_world=(0.1, 0.2, 0.7)):
    refs = [(50.4, 30.5, 0), (49.8, 24.0, 0), (52.0, 21.0, 1), (52.4, 19.0, 1), (40.0, -100.0, 2), (35.0, -90.0, 2)]
    m = GeoModel()
    m.classes = list(classes)
    m.ref_lat = np.array([r[0] for r in refs], float)
    m.ref_lng = np.array([r[1] for r in refs], float)
    m.ref_yi = np.array([r[2] for r in refs])
    m.ref_rows = [np.flatnonzero(m.ref_yi == c) for c in range(len(classes))]
    m.prior_world = np.array(prior_world, float)
    m.prior_uniform = np.full(len(classes), 1.0 / len(classes))
    m.prior_mix = 0.4
    m.weights = {"glm": 0.5}
    m.loc_params = {"bw": 1.0, "floor": 0.1}
    m.priors = {
        "alpha": 0.5, "folds": 5,
        "ranked_world": {"counts": {"UA": 10, "PL": 30, "US": 60}},
        "maps": {WORLD_ID: {"name": "World", "counts": {"UA": 20, "PL": 10, "US": 10},
                            "fold_counts": [{"UA": 20}, {"PL": 10}, {"US": 10}, {}, {}]},
                 "tw": {"name": "The World", "counts": {"UA": 5, "PL": 5, "US": 5}}},
        "params": {"weights": {"glm": 0.7}, "prior_mix": 0.0, "mix": {"map": 0.4, "map_ranked": 0.2, "ranked": 0.4}}}
    return m


def game_in(split):
    """A game token whose md5 parity puts it in split (world_calib / calib = even)."""
    for k in range(1000):
        t = "g%d" % k
        if (int(hashlib.md5(t.encode()).hexdigest(), 16) % 2 == 0) == (split in ("calib", "world_calib")):
            return t
    raise AssertionError


class TestWorldOverride(unittest.TestCase):
    def test_per_map_mix(self):
        m = toy_model()
        m._fit_hash = "toyfit000001"
        world = {"id": WORLD_ID, "world": True}
        shared = m.map_setup(world)
        self.assertEqual(shared["kind"], "map")
        stamp = m.override_stamp(m.priors["params"])
        ranked = m.empirical_prior({"UA": 10, "PL": 30, "US": 60})
        m.priors["params"]["maps"] = {WORLD_ID: dict(stamp, mix={"map": 0.9, "map_ranked": 0.0}, model=m.fingerprint())}
        s = m.map_setup(world)
        emp = m.empirical_prior({"UA": 20, "PL": 10, "US": 10})
        np.testing.assert_allclose(s["prior"], 0.9 * emp + 0.1 / 3, rtol=1e-9)
        # another map keeps the shared mixture
        tw = m.map_setup({"id": "tw", "world": True})
        ranked = m.empirical_prior({"UA": 10, "PL": 30, "US": 60})
        np.testing.assert_allclose(tw["prior"], 0.4 * m.empirical_prior({"UA": 5, "PL": 5, "US": 5}) + 0.2 * ranked
                                   + 0.4 / 3, rtol=1e-9)
        # an override calibrated for another model is ignored
        m.priors["params"]["maps"][WORLD_ID]["model"] = "0123456789ab"
        np.testing.assert_allclose(m.map_setup(world)["prior"], shared["prior"])
        m.priors["params"]["maps"][WORLD_ID]["model"] = m.fingerprint()
        np.testing.assert_allclose(m.map_setup(world)["prior"], 0.9 * emp + 0.1 / 3, rtol=1e-9)
        # ... and so is one calibrated on another fit of the same classes / groups (a GLM refit), one without its
        # stamp, and one layered on other shared parameters (--calibrate-maps since)
        m._fit_hash = "toyfit000002"
        np.testing.assert_allclose(m.map_setup(world)["prior"], shared["prior"])
        m._fit_hash = "toyfit000001"
        for k in ("fit", "shared"):
            o = dict(m.priors["params"]["maps"][WORLD_ID])
            del o[k]
            m.priors["params"]["maps"][WORLD_ID] = o
            np.testing.assert_allclose(m.map_setup(world)["prior"], shared["prior"])
            m.priors["params"]["maps"][WORLD_ID] = dict(o, **stamp)
        m.priors["params"]["mix"] = {"map": 0.5, "map_ranked": 0.2, "ranked": 0.3}
        shared2 = m.map_setup(world)
        self.assertEqual(m.map_override(world), {})
        np.testing.assert_allclose(shared2["prior"], 0.5 * emp + 0.2 * ranked + 0.3 / 3, rtol=1e-9)
        m.priors["params"]["mix"] = {"map": 0.4, "map_ranked": 0.2, "ranked": 0.4}
        self.assertEqual(m.map_override(world)["mix"], {"map": 0.9, "map_ranked": 0.0})
        # an override that only records the calibration (family "shared": no mix / prior_mix) keeps the shared prior
        m.priors["params"]["maps"][WORLD_ID] = {"family": "shared", "model": m.fingerprint()}
        np.testing.assert_allclose(m.map_setup(world)["prior"], shared["prior"])
        # the model-prior blend of the World map alone (Street View coverage frequencies vs uniform)
        m.priors["params"]["maps"][WORLD_ID] = dict(stamp, family="prior_mix", mix={"map": 0.4, "map_ranked": 0.2},
                                                    prior_mix=1.0, model=m.fingerprint())
        np.testing.assert_allclose(m.map_setup(world)["prior"], 0.4 * emp + 0.2 * ranked + 0.4 * m.prior_world,
                                   rtol=1e-9)
        # without per-map counts the World map uses the ranked prior, override or not
        m.priors["params"]["maps"][WORLD_ID] = dict(stamp, mix={"map": 0.9, "map_ranked": 0.0}, model=m.fingerprint())
        del m.priors["maps"][WORLD_ID]
        s = m.map_setup(world)
        self.assertEqual(s["kind"], "ranked")
        np.testing.assert_allclose(s["prior"], 0.4 * ranked + 0.6 / 3, rtol=1e-9)


class TestWorldSelection(unittest.TestCase):
    def test_families_are_nested_around_the_shared_mix(self):
        import train_model as tm
        shared = {"map": 0.3, "map_ranked": 0.1, "prior_mix": 0.0}
        fams = dict(tm.world_families(shared))
        self.assertEqual(fams["shared"], [shared])
        self.assertTrue(all(c["map"] == 0.3 and c["map_ranked"] == 0.1 for c in fams["prior_mix"]))
        self.assertEqual(sorted(c["prior_mix"] for c in fams["prior_mix"]), sorted(tm.PM_GRID))
        self.assertTrue(all(c["map_ranked"] == 0.1 and c["prior_mix"] == 0.0 for c in fams["map"]))
        for name, cands in fams.items():
            self.assertTrue(all(c["map"] + c["map_ranked"] <= 0.95 + 1e-9 for c in cands), name)
            self.assertIn(shared, cands, name)
            self.assertEqual(len(cands), len({tuple(sorted(c.items())) for c in cands}), name)

    def test_repeated_cv_scores_out_of_fold(self):
        import train_model as tm
        games = np.array(["g%d" % (k // 3) for k in range(60)])
        folds = tm.world_cv_folds(games, k=5, reps=4)
        self.assertEqual(len(folds), 4)
        for f in folds:  # the rounds of one game share a fold
            for g in set(games):
                self.assertEqual(len(set(f[games == g])), 1)
        np.testing.assert_array_equal(folds[0], tm.world_cv_folds(games, k=5, reps=4)[0])
        # candidate 1 better everywhere: every out-of-fold value comes from it
        L = np.vstack([np.full(60, -3.0), np.full(60, -2.0)])
        np.testing.assert_allclose(tm.world_cv(L, folds), -2.0)
        # a candidate that only wins on one game is chosen in-sample but its CV is no better than its rival
        L = np.vstack([np.full(60, -2.0), np.where(games == "g0", 5.0, -2.2)])
        self.assertGreater(L[1].mean(), L[0].mean())
        self.assertLess(tm.world_cv(L, folds)[games != "g0"].mean(), -2.0 + 1e-9)


@unittest.skipUnless(HAVE_MAPS, "data/maps.json missing")
class TestWorldPriors(unittest.TestCase):
    def test_world_rounds_and_counts(self):
        import build_priors
        cal, test = game_in("world_calib"), game_in("world_test")
        games = {"map": {"id": WORLD_ID}, "games": [
            {"token": cal, "status": "finished", "split": "world_calib", "walk": 0,
             "rounds": [{"round": 1, "lat": 50.0, "lng": 30.0, "pano_id": "p1", "gg_country": "UA"},
                        {"round": 2, "lat": 52.0, "lng": 21.0, "pano_id": "p2", "gg_country": "pl"}]},
            {"token": test, "status": "finished", "split": "world_test", "walk": 3,
             "rounds": [{"round": 1, "lat": 40.0, "lng": -100.0, "pano_id": "p3", "gg_country": "US"}]},
            {"token": "unfinished", "status": "created", "rounds": []}]}
        with tempfile.TemporaryDirectory() as tmp:
            f = os.path.join(tmp, "world_games.json")
            with open(f, "w") as fh:
                json.dump(games, fh)
            rr = build_priors.world_play_rounds(f)
        self.assertEqual(len(rr), 3)
        self.assertTrue(all(r["map"] == "World" and r["source"] == "world_play" for r in rr))
        out = build_priors.per_map(rr, 1, 5)
        self.assertIn(WORLD_ID, out)
        e = out[WORLD_ID]
        # only the calib half is counted (world_test rounds never enter the prior)
        self.assertEqual(e["counts"], {"PL": 1, "UA": 1})
        self.assertEqual(e["world_play_rounds"], 2)
        self.assertEqual(sum(sum(fc.values()) for fc in e["fold_counts"]), 2)

    def test_split_fallback_is_the_parity_rule(self):
        import train_model as tm
        import world_play
        from calibrate import split_of
        cal, test = game_in("world_calib"), game_in("world_test")
        toks = [cal, test] + ["g%d" % i for i in range(200)]
        for t in toks:
            want = split_of({"mode": "world_play", "game": t, "pano_id": ""})
            self.assertEqual(tm.world_split_of(t), want, t)
            self.assertEqual(world_play.split_of_game(t), want, t)
        # a game record without "split" lands in the half of its hash, like the stored ones
        games = {"games": [{"token": t, "status": "finished", "walk": 0,
                            "rounds": [{"round": 1, "lat": 0.0, "lng": 0.0, "gg_country": "US"}]} for t in toks]}
        with tempfile.TemporaryDirectory() as tmp:
            f = os.path.join(tmp, "world_games.json")
            with open(f, "w") as fh:
                json.dump(games, fh)
            rr = tm.world_rounds(f)
        self.assertEqual(len(rr), len(toks))
        for r in rr:
            self.assertEqual(r["split"], split_of({"mode": "world_play", "game": r["game"], "pano_id": ""}))
        self.assertEqual({r["game"]: r["split"] for r in rr}[cal], "world_calib")
        self.assertEqual({r["game"]: r["split"] for r in rr}[test], "world_test")

    def test_out_of_fold_world_prior(self):
        import train_model as tm
        m = toy_model()
        k = int(m.priors["folds"])
        tok = next(t for t in ("g%d" % i for i in range(1000))
                   if tm.fold_of(t, k) == 0 and tm.fold_of(t, 2) == 0)
        r_cal = {"game": tok, "split": "world_calib"}
        r_test = {"game": tok, "split": "world_test"}
        mix = {"map": 1.0, "map_ranked": 0.0}
        P = tm.world_priors(m, [r_cal, r_test], mix)
        # world_calib round of fold 0: its fold's UA rounds are left out of the counts; world_test: all counts
        emp_oof = m.empirical_prior({"UA": 0, "PL": 10, "US": 10})
        emp_all = m.empirical_prior({"UA": 20, "PL": 10, "US": 10})
        np.testing.assert_allclose(P[0], emp_oof / emp_oof.sum(), rtol=1e-9)
        np.testing.assert_allclose(P[1], emp_all / emp_all.sum(), rtol=1e-9)

    def test_response_evidence_swaps_the_prior(self):
        import train_model as tm
        classes = ["IN", "KH", "TH"]
        prior = np.array([0.6, 0.1, 0.3])
        ev = np.array([1.0, 2.5, 0.0])
        post = prior * np.exp(ev)
        post /= post.sum()
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "response.json"), "w") as fh:
                json.dump({"posterior": dict(zip(classes, post.tolist())), "prior": dict(zip(classes, prior.tolist()))},
                          fh)
            e, q = tm.response_evidence(tmp, classes)
        np.testing.assert_allclose(q, prior)
        new = np.array([0.2, 0.4, 0.4])
        tp, lp = tm._lp_true(new[None, :], e[None, :], np.array([1]))
        want = new * np.exp(ev)
        want /= want.sum()
        np.testing.assert_allclose(np.exp(lp[0]), want, rtol=1e-6)
        self.assertAlmostEqual(float(tp[0]), math.log(want[1]), places=6)


class TestPrunedMoveCaptures(unittest.TestCase):
    """tools/world_play.py deletes the move captures' view images of old games when its disk budget runs out: the
    re-analysis keeps the start answer of such a round and leaves its fused answer out (NaN) of every Moving row."""

    def test_capture_has_images(self):
        import train_model as tm
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "views.json"), "w") as fh:
                json.dump({"views": [{"file": "view_00.jpg"}, {"file": "view_01.jpg"}]}, fh)
            for f in ("view_00.jpg", "view_01.jpg"):
                open(os.path.join(tmp, f), "wb").close()
            self.assertTrue(tm.capture_has_images(tmp))
            os.remove(os.path.join(tmp, "view_01.jpg"))
            self.assertFalse(tm.capture_has_images(tmp))
            self.assertFalse(tm.capture_has_images(os.path.join(tmp, "missing")))

    def test_report_leaves_out_missing_fused_answers(self):
        import train_model as tm
        C = 3
        post = np.tile(np.array([0.7, 0.2, 0.1], np.float32), (4, 1))
        fin = post.copy()
        fin[3] = np.nan  # Moving round 3: move images pruned
        a = {"game": np.array(["a", "a", "b", "b"]), "y": np.array([0, 1, 0, 0]), "walk": np.array([0, 0, 3, 3]),
             "complete": np.array([True, True, True, False]), "played_points": np.zeros(4)}
        for n in ("new", "old"):
            a.update({"%s_start_post" % n: post, "%s_start_points" % n: np.array([5000.0, 100.0, 4000.0, 3000.0]),
                      "%s_start_km" % n: np.ones(4), "%s_final_post" % n: fin,
                      "%s_final_points" % n: np.array([5000.0, 100.0, 4500.0, np.nan]),
                      "%s_final_km" % n: np.array([1.0, 1.0, 1.0, np.nan])})
        rep = tm.world_points_report(a, "test")
        self.assertEqual(rep["moving_rounds_without_move_images"], 1)
        fused = rep["Moving games (fused)"]
        self.assertEqual(fused["n"], 1)
        self.assertEqual(fused["new"]["points"], 4500.0)
        self.assertEqual(rep["Moving games, start capture"]["n"], 2)
        self.assertEqual(rep["No Move answer (start capture), all rounds"]["n"], 4)
        self.assertEqual(C, post.shape[1])


def flag_sphere(paint=None, w=2048):
    """Synthetic equirectangular panorama: sky, buildings, grey road; paint(rgb, col_of, row_of) draws on it."""
    from engine.panorama import SphericalImage
    h = w // 2
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[: h // 2] = (150, 185, 225)                       # sky
    rgb[h // 2 - 60: h // 2] = (205, 195, 175)            # beige buildings at the horizon
    rgb[h // 2:] = (110, 110, 105)                        # road / ground

    def col(az):
        return int((az + 180.0) / 360.0 * w)

    def row(el):
        return int((90.0 - el) / 180.0 * h)
    if paint:
        paint(rgb, col, row)
    return SphericalImage(rgb, heading=0.0, car_heading=0.0)


def kh_flag(rgb, x0, y0, wd, ht, vertical=False):
    """Cambodian layout blue | red (x2) | blue: horizontal stripes, or vertical ones for a hanging banner."""
    blue, red = (3, 46, 161), (224, 0, 37)
    n = wd if vertical else ht
    for k in range(n):
        c = blue if (k < n / 4 or k >= 3 * n / 4) else red
        if vertical:
            rgb[y0:y0 + ht, x0 + k] = c
        else:
            rgb[y0 + k, x0:x0 + wd] = c


class TestFlagDetector(unittest.TestCase):
    def test_kh_flag_found_where_painted(self):
        from engine import flag_detect as F
        sph = flag_sphere(lambda rgb, col, row: kh_flag(rgb, col(40.0), row(10.0), 30, 20))
        x, where = F.scan(sph, return_where=True)
        s, az, el = where["KH"]
        self.assertLess(abs(az - 40.0 - 30 * 360.0 / 2048 / 2), 2.0)
        self.assertLess(abs(el - (10.0 - 20 * 180.0 / 1024 / 2)), 2.0)
        x0 = F.scan(flag_sphere())
        i = F.FEATURE_NAMES.index("flag_KH")
        self.assertGreater(x[i], 0.4)
        self.assertGreater(x[i] - x0[i], 0.3)

    def test_vertical_banner(self):
        from engine import flag_detect as F
        sph = flag_sphere(lambda rgb, col, row: kh_flag(rgb, col(-60.0), row(12.0), 16, 40, vertical=True))
        x, where = F.scan(sph, return_where=True)
        i = F.FEATURE_NAMES.index("flag_KH_banner")
        self.assertGreater(x[i], 0.4)
        self.assertLess(abs(where["KH_banner"][1] - (-60.0 + 8 * 360.0 / 2048)), 2.0)

    def test_red_roof_is_not_a_red_flag(self):
        """A large red area scores below a small isolated red patch (the ring term)."""
        from engine import flag_detect as F

        def roof(rgb, col, row):
            rgb[row(8.0):row(-2.0), col(100.0):col(140.0)] = (200, 20, 30)

        def patch(rgb, col, row):
            rgb[row(9.0):row(6.0), col(100.0):col(104.0)] = (200, 20, 30)
        i = F.FEATURE_NAMES.index("flag_RED_FIELD")
        self.assertGreater(F.scan(flag_sphere(patch))[i], F.scan(flag_sphere(roof))[i] + 0.2)

    def test_masked_windows_are_skipped(self):
        from engine import flag_detect as F
        sph = flag_sphere(lambda rgb, col, row: kh_flag(rgb, col(40.0), row(10.0), 30, 20))
        sph.mask[:, :] = False
        x = F.scan(sph)
        self.assertTrue(np.all(np.isnan(x[:len(F.TEMPLATES)])))


if __name__ == "__main__":
    unittest.main()
