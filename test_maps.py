#!/usr/bin/env python3
"""
Tests for the map-aware locator (score scale, bounds, prior):  python3 -m unittest test_maps -v
"""
import math
import os
import sys
import tempfile
import unittest

import numpy as np

from engine.geo import (WORLD_SCORE_SCALE_KM, clip_to_bounds, geoguessr_points, geoguessr_score, in_bounds,
                        is_world_map, load_maps, parse_bounds, positive_float, resolve_map, score_scale_km)
from engine.model import GeoModel

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "web"))
HAVE_MODEL = os.path.exists(os.path.join(ROOT, "data", "model", "model.json"))
HAVE_MAPS = bool(load_maps())
UKRAINE = {"min": {"lat": 46.3336, "lng": 22.2097}, "max": {"lat": 52.2407, "lng": 37.9413}}
UK = {"min": {"lat": 49.9609, "lng": -8.1376}, "max": {"lat": 60.8248, "lng": 1.7582}}
NYC = {"min": {"lat": 40.5, "lng": -74.3}, "max": {"lat": 40.95, "lng": -73.7}}


def client_points(d_km, max_error_m):
    """The game client: Math.round(d > 25 m ? 5000 / exp(10 * d / D) : 5000), d and D in metres."""
    d = d_km * 1000.0
    return 5000 if d <= 25 else int(math.floor(5000.0 / math.exp(10.0 * d / max_error_m) + 0.5))


class TestScore(unittest.TestCase):
    def test_formula_per_map(self):
        for D in (14916862, 13316891, 14999250, 18499075, 1312553):
            for d in (0.01, 0.5, 12.0, 150.0, 1000.0, 4321.0, 12000.0):
                self.assertEqual(int(geoguessr_points(d, D)), client_points(d, D), (D, d))
        self.assertAlmostEqual(score_scale_km(14916862), WORLD_SCORE_SCALE_KM)
        self.assertAlmostEqual(float(geoguessr_score(500.0)), float(geoguessr_score(500.0, score_scale_km(None))))
        self.assertEqual(score_scale_km(0), WORLD_SCORE_SCALE_KM)
        for bad in (True, float("nan"), float("inf"), -1, "abc", None):
            self.assertIsNone(positive_float(bad), bad)
            self.assertEqual(score_scale_km(bad), WORLD_SCORE_SCALE_KM)
        self.assertEqual(positive_float("1312553"), 1312553.0)
        # Ukraine: 11x tighter than the World map
        self.assertLess(int(geoguessr_points(100.0, 1312553)), 2400)
        self.assertGreater(int(geoguessr_points(100.0, 14916862)), 4600)

    @unittest.skipUnless(HAVE_MAPS, "data/maps.json missing (tools/fetch_maps.py)")
    def test_registry(self):
        tw = resolve_map("The World")
        self.assertEqual(tw["id"], "66014417ff2366aa9a7504df")
        self.assertEqual(tw["maxErrorDistance"], 13316891)
        self.assertTrue(tw["world"] and tw["known"])
        self.assertEqual(resolve_map("ukraine")["maxErrorDistance"], 1312553)
        self.assertFalse(resolve_map("Ukraine")["world"])
        self.assertEqual(resolve_map({"id": "696fe47c5b07bed052077a95"})["name"], "A Moving World")
        # in-game bounds win over the registry
        b = {"min": {"lat": -50, "lng": -170}, "max": {"lat": 71.09, "lng": 178}}
        self.assertEqual(resolve_map({"id": "698f47ed7f653e99dffa51bb", "bounds": b})["bounds"]["max"]["lat"], 71.09)
        self.assertIsNone(resolve_map(None))
        # invalid in-game values fall back to the registry
        self.assertEqual(resolve_map({"id": "ukraine", "maxErrorDistance": float("nan")})["maxErrorDistance"], 1312553)
        self.assertEqual(resolve_map({"name": "World", "maxErrorDistance": "abc"})["maxErrorDistance"], 14916862)
        self.assertEqual(resolve_map({"id": "ukraine", "maxErrorDistance": True})["maxErrorDistance"], 1312553)
        self.assertEqual(resolve_map({"id": "ukraine", "bounds": {"min": {"lat": 0, "lng": 170},
                                                                  "max": {"lat": 1, "lng": -170}}})["bounds"],
                         resolve_map("ukraine")["bounds"])
        self.assertTrue(resolve_map("A Moving World")["updatedAt"].startswith("20"))
        unknown = resolve_map("Some Other World")
        self.assertFalse(unknown["known"])
        self.assertTrue(unknown["world"])


class TestBounds(unittest.TestCase):
    def test_parse_and_clip(self):
        self.assertEqual(parse_bounds(UKRAINE), (46.3336, 22.2097, 52.2407, 37.9413))
        self.assertEqual(parse_bounds([[46.3336, 22.2097], [52.2407, 37.9413]]), parse_bounds(UKRAINE))
        self.assertIsNone(parse_bounds({"min": {"lat": 10, "lng": 0}, "max": {"lat": 0, "lng": 5}}))
        self.assertIsNone(parse_bounds({"min": {}}))
        self.assertEqual(clip_to_bounds(55.0, 30.0, UKRAINE), (52.2407, 30.0))
        self.assertEqual(clip_to_bounds(40.0, 10.0, UKRAINE), (46.3336, 22.2097))
        self.assertEqual(clip_to_bounds(50.0, 30.0, UKRAINE), (50.0, 30.0))
        self.assertEqual(clip_to_bounds(50.0, 30.0, None), (50.0, 30.0))
        self.assertTrue(in_bounds(52.5, 30.0, UKRAINE, margin_deg=0.5))
        self.assertFalse(in_bounds(52.5, 30.0, UKRAINE))
        self.assertTrue(is_world_map({"bounds": {"min": {"lat": -43.6, "lng": -123.2}, "max": {"lat": 66.5, "lng": 174.9}}}))
        self.assertFalse(is_world_map({"name": "A World", "bounds": UKRAINE}))

    def test_locate_inside_bounds(self):
        m = toy_model()
        lp = np.log([0.45, 0.45, 0.10])
        d2 = np.zeros(len(m.ref_yi))
        w = m.ref_weights(lp, d2, bounds=UKRAINE)
        outside = ~in_bounds(m.ref_lat, m.ref_lng, UKRAINE, margin_deg=0.5)
        self.assertTrue(np.all(w[outside] == 0))
        self.assertAlmostEqual(float(w.sum()), 1.0)
        g = m.locate(lp, d2, bounds=UKRAINE, score_scale_km=score_scale_km(1312553))
        self.assertTrue(in_bounds(g["lat"], g["lng"], UKRAINE))
        # a reference just outside the box (inside the margin) is clipped onto the boundary
        g = m.locate(np.log([1e-6, 1.0 - 2e-6, 1e-6]), d2, bounds={"min": {"lat": 46.5, "lng": 22.8},
                                                                    "max": {"lat": 51.8, "lng": 37.5}})
        self.assertEqual((g["lat"], g["lng"]), (50.1, 22.8))
        # without bounds the guess may leave the box
        g = m.locate(np.log([0.05, 0.05, 0.9]), d2)
        self.assertFalse(in_bounds(g["lat"], g["lng"], UKRAINE))


def toy_model(classes=("UA", "PL", "US"), refs=None, prior_world=(0.1, 0.2, 0.7)):
    """Toy classes with reference panoramas (default: UA inside the Ukraine box, PL straddling it,
    US far away).  refs: [(lat, lng, class index)]."""
    if refs is None:
        refs = [(50.4, 30.5, 0), (49.8, 24.0, 0), (48.5, 35.0, 0), (46.9, 30.7, 0), (52.0, 21.0, 1), (52.4, 19.0, 1),
                (50.1, 22.6, 1), (40.0, -100.0, 2), (35.0, -90.0, 2), (45.0, -110.0, 2)]
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
    return m


class TestPrior(unittest.TestCase):
    def setUp(self):
        self.m = toy_model()
        self.m.priors = {
            "alpha": 0.5,
            "ranked_world": {"counts": {"UA": 10, "PL": 30, "US": 60}},
            "maps": {"ukr": {"name": "Ukraine", "counts": {"UA": 70}}},
            "params": {"weights": {"glm": 0.7}, "prior_mix": 0.0,
                       "mix": {"map": 0.7, "map_ranked": 0.3, "ranked": 0.9}}}

    def test_fallbacks(self):
        m = self.m
        s = m.map_setup(None)
        self.assertEqual(s["kind"], "none")
        np.testing.assert_allclose(s["prior"], m.base_prior())
        self.assertIs(s["weights"], m.weights)
        self.assertAlmostEqual(s["scale_km"], WORLD_SCORE_SCALE_KM)
        # an unknown World-type map: ranked-duel mixture with the map-calibrated exponents
        s = m.map_setup({"name": "Some World", "world": True})
        self.assertEqual(s["kind"], "ranked")
        np.testing.assert_allclose(s["prior"], 0.9 * m.empirical_prior({"UA": 10, "PL": 30, "US": 60}) + 0.1 / 3)
        self.assertEqual(s["weights"], {"glm": 0.7})
        # an unknown single-country map: the model prior
        s = m.map_setup({"name": "Somewhere", "world": False, "maxErrorDistance": 2000000})
        self.assertEqual(s["kind"], "base")
        np.testing.assert_allclose(s["prior"], m.base_prior())
        self.assertAlmostEqual(s["scale_km"], 200.0)
        # a map with its own frequencies
        s = m.map_setup({"id": "ukr", "world": False})
        self.assertEqual(s["kind"], "map")
        self.assertGreater(s["prior"][0], 0.6)
        # no calibrated parameters: the model prior even for World-type maps
        del m.priors["params"]
        self.assertEqual(m.map_setup({"name": "Some World", "world": True})["kind"], "base")
        m.priors = {}
        self.assertEqual(m.map_setup({"id": "ukr", "world": True})["kind"], "base")

    def test_bounds_mask(self):
        s = self.m.map_setup({"id": "ukr", "bounds": UKRAINE, "world": False})
        self.assertAlmostEqual(float(s["prior"].sum()), 1.0)
        self.assertLess(s["prior"][2], 1e-6)          # US: nothing inside the box
        self.assertGreater(s["prior"][1], 1e-3)       # PL: part of the country is inside
        self.assertEqual(s["bounds"], parse_bounds(UKRAINE))
        # World-type map: classes inside are kept as they are, the rest removed
        wide = {"min": {"lat": 20.0, "lng": 10.0}, "max": {"lat": 60.0, "lng": 179.0}}
        s = self.m.map_setup({"name": "Some World", "bounds": wide, "world": True})
        ranked = self.m.empirical_prior({"UA": 10, "PL": 30, "US": 60})[:2]
        np.testing.assert_allclose(s["prior"][:2], 0.9 * ranked / ranked.sum() + 0.1 * 0.5, rtol=1e-6)
        self.assertLess(s["prior"][2], 1e-6)

    def test_neighbour_touching_the_box(self):
        """A large neighbour with a sliver inside a single-country box (France and the UK map)
        keeps only that sliver of its prior, also with references just outside the box."""
        refs = [(51.5, -0.1, 0), (53.5, -2.2, 0), (55.9, -3.2, 0), (57.1, -2.1, 0), (49.6, -1.6, 1), (48.85, 2.35, 1),
                (45.76, 4.84, 1), (43.3, 5.4, 1), (53.35, -6.26, 2), (51.9, -8.47, 2)]
        m = toy_model(("GB", "FR", "IE"), refs, (0.25, 0.6, 0.15))
        m.prior_mix = 1.0
        s = m.map_setup({"name": "AI Generated United Kingdom", "bounds": UK, "world": False})
        self.assertEqual(s["kind"], "base")
        self.assertEqual(int(np.argmax(s["prior"])), 0)
        self.assertLess(s["prior"][1], 0.01)
        self.assertGreater(s["prior"][2], 0.1)
        # the location kernel still counts the reference just outside the box (margin)
        self.assertTrue(m.refs_inside(UK)[4])

    def test_small_box(self):
        """A city box inside a large country keeps (only) that country."""
        refs = [(40.7, -74.0, 0)] + [(30.0 + 0.1 * k, -120.0 + 0.25 * k, 0) for k in range(149)] + \
               [(45.5, -73.6, 1), (43.65, -79.4, 1)]
        m = toy_model(("US", "CA"), refs, (0.3, 0.7))
        support, share = m.bounds_factors(NYC)
        self.assertEqual(support.tolist(), [1.0, 1e-9])
        self.assertGreater(share[0], 1e-3)
        s = m.map_setup({"name": "New York City", "bounds": NYC, "world": False})
        self.assertGreater(s["prior"][0], 0.999)

    def test_counts_dates_and_model_check(self):
        m = self.m
        m.priors["maps"]["ukr"].update(dates=["2026-02-16", "2026-03-12"], map_updated="2026-10-03T13:55:40Z", n=70)
        c = m.map_setup({"id": "ukr", "world": False})["counts"]
        self.assertEqual((c["rounds"], c["stale"], c["map_updated"]), (70, True, "2026-10-03"))
        self.assertFalse(m.map_setup({"id": "ukr", "world": False, "updatedAt": "2025-08-01T00:00:00Z"})["counts"]["stale"])
        self.assertIsNone(m.map_setup({"name": "Some World", "world": True})["counts"])
        # parameters calibrated for another model (classes / groups) are not used
        m.priors["params"]["model"] = m.fingerprint()
        self.assertEqual(m.map_setup({"name": "Some World", "world": True})["kind"], "ranked")
        m.priors["params"]["model"] = "0123456789ab"
        self.assertEqual(m.map_setup({"name": "Some World", "world": True})["kind"], "base")


class TestServer(unittest.TestCase):
    def test_map_request(self):
        from server import debug_dir_from_request, map_from_request
        self.assertIsNone(map_from_request({}))
        self.assertIsNone(map_from_request({"map": None}))
        self.assertEqual(map_from_request({"map": "The World"}), "The World")
        req = {"map": {"id": "66014417ff2366aa9a7504df", "name": "The World", "bounds": UKRAINE,
                       "maxErrorDistance": "13316891"}}
        self.assertEqual(map_from_request(req), {"id": "66014417ff2366aa9a7504df", "name": "The World",
                                                 "bounds": UKRAINE, "maxErrorDistance": 13316891.0})
        self.assertEqual(map_from_request({"map": {"name": "X", "maxErrorDistance": None}}), {"name": "X"})
        # bad fields are dropped with a warning (the round keeps its prediction)
        crossing = {"min": {"lat": -20, "lng": 170}, "max": {"lat": -10, "lng": -170}}
        for bad in ({"bounds": {"min": 1}}, {"bounds": crossing}, {"maxErrorDistance": -5},
                    {"maxErrorDistance": "abc"}, {"maxErrorDistance": True}, {"maxErrorDistance": float("nan")}):
            warnings = []
            self.assertEqual(map_from_request({"map": dict(bad, id="ukraine")}, warnings), {"id": "ukraine"}, bad)
            self.assertEqual(len(warnings), 1, bad)
        self.assertIsNone(map_from_request({"map": {"bounds": crossing}}))
        with self.assertRaises(ValueError):
            map_from_request({"map": [1, 2]})
        self.assertIsNone(debug_dir_from_request({}))
        self.assertEqual(debug_dir_from_request({"debug_dir": "scratch/live/r1"}),
                         os.path.realpath(os.path.join(ROOT, "scratch", "live", "r1")))
        t = os.path.join(tempfile.gettempdir(), "gg_debug")
        self.assertEqual(debug_dir_from_request({"debug_dir": t}), os.path.realpath(t))
        with self.assertRaises(ValueError):
            debug_dir_from_request({"debug_dir": "/etc/x"})
        with self.assertRaises(ValueError):
            debug_dir_from_request({"debug_dir": "../outside"})

    @unittest.skipUnless(HAVE_MAPS, "data/maps.json missing")
    def test_evaluate_round_uses_map_scale(self):
        from server import evaluate_round
        from engine.geo import haversine_km
        r = {"lat": 50.45, "lng": 30.52, "guess_lat": 49.84, "guess_lng": 24.03}
        d = float(haversine_km(50.45, 30.52, 49.84, 24.03))
        self.assertEqual(evaluate_round(r)["points"], client_points(d, 14916862))
        self.assertEqual(evaluate_round(dict(r, map={"id": "ukraine"}))["points"], client_points(d, 1312553))
        self.assertEqual(evaluate_round(dict(r, map="The World"))["points"], client_points(d, 13316891))


@unittest.skipUnless(HAVE_MODEL and HAVE_MAPS, "trained model or data/maps.json missing")
class TestLocatorMap(unittest.TestCase):
    def test_map_result(self):
        from engine.locator import get_locator
        pano = os.path.join(ROOT, "test_japan_pano.jpg")
        loc = get_locator()
        res = loc.analyze_image(pano, map_info="Ukraine")
        self.assertEqual(res["map"]["name"], "Ukraine")
        self.assertEqual(res["map"]["maxErrorDistance"], 1312553)
        self.assertTrue(in_bounds(res["guess"]["lat"], res["guess"]["lng"], resolve_map("Ukraine")["bounds"]))
        self.assertNotIn("US", [c["code"] for c in res["countries"]])
        res = loc.analyze_image(pano)
        self.assertIsNone(res["map"])
        with tempfile.TemporaryDirectory() as d:
            res = loc.analyze_image(pano, map_info={"name": "The World"}, debug_dir=d)
            self.assertTrue(os.path.exists(os.path.join(d, "sphere.jpg")))
        self.assertEqual(res["map"]["id"], "66014417ff2366aa9a7504df")

    def test_uk_map_prior(self):
        """On the UK map without its own counts the prior is Great Britain's, not France's."""
        from engine.locator import get_locator
        m = get_locator().model
        s = m.map_setup(resolve_map("United Kingdom (Better Map)"))
        p = dict(zip(m.classes, s["prior"]))
        self.assertEqual(max(p, key=p.get), "GB")
        self.assertLess(p.get("FR", 0.0), 0.02)


if __name__ == "__main__":
    unittest.main()
