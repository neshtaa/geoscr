#!/usr/bin/env python3
"""
Tests for the pure-math locator:  python3 -m unittest test_locator -v
"""
import math
import os
import unittest

import numpy as np
from PIL import Image

from engine import features
from engine.geo import country_at, geoguessr_score, haversine_km
from engine.hints import ClueBase, build_hints, observations
from engine.panorama import SphericalImage

ROOT = os.path.dirname(os.path.abspath(__file__))
PANO = os.path.join(ROOT, "test_japan_pano.jpg")
HAVE_MODEL = os.path.exists(os.path.join(ROOT, "data", "model", "model.json"))


class TestGeometry(unittest.TestCase):
    def test_country_lookup(self):
        self.assertEqual(country_at(50.45, 30.52), "UA")
        self.assertEqual(country_at(35.68, 139.69), "JP")
        self.assertEqual(country_at(-23.55, -46.63), "BR")

    def test_region_lookup(self):
        from engine.geo import region_at
        self.assertEqual(region_at(50.45, 30.52)[0], "UA-30")
        self.assertEqual(region_at(-7.28, 112.75)[0], "ID-JI")

    def test_score_curve(self):
        self.assertAlmostEqual(float(geoguessr_score(0)), 5000.0)
        d = float(haversine_km(50.45, 30.52, 52.23, 21.01))
        self.assertTrue(680 < d < 700)

    def test_views_roundtrip(self):
        img = Image.open(PANO).convert("RGB").resize((1024, 512))
        sph = SphericalImage.from_equirect(img, heading=0.0, width=1024)
        views = [{"image": sph.render_view(y, p, 120.0, (640, 400)), "yaw": y, "pitch": p, "hfov": 120.0}
                 for p in (-55, 0, 55) for y in (0, 90, 180, 270)]
        rec = SphericalImage.from_views(views, width=1024)
        self.assertGreater(rec.coverage(), 0.98)
        band = slice(rec.row_of(40), rec.row_of(-40))
        err = np.abs(rec.rgb[band].astype(float) - sph.rgb[band].astype(float))[rec.mask[band]].mean()
        self.assertLess(err, 12.0)


class TestSolar(unittest.TestCase):
    def test_latitude_likelihood_hemisphere(self):
        from engine.features.solar import latitude_likelihood
        grid = np.arange(-60.0, 75.0, 1.0)
        # sun due north at 40 deg elevation: only possible in the southern hemisphere / tropics
        L = latitude_likelihood(0.0, 40.0, grid)
        self.assertGreater(L[grid < 0].sum(), 0.8)
        L = latitude_likelihood(180.0, 30.0, grid)
        self.assertGreater(L[grid > 0].sum(), 0.8)
        self.assertAlmostEqual(float(L.sum()), 1.0, places=5)


class TestFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sph = SphericalImage.from_equirect(PANO, heading=90.0)

    def test_modules(self):
        for m in features.available():
            r = m.extract(self.sph)
            self.assertEqual(len(r["x"]), len(m.FEATURE_NAMES), m.NAME)
            self.assertIsInstance(r["evidence"], dict)

    def test_unknown_heading_and_partial_view(self):
        sph = SphericalImage.from_screenshot(Image.open(PANO).crop((0, 100, 800, 500)), hfov=100.0)
        for m in features.available():
            r = m.extract(sph)
            self.assertEqual(len(r["x"]), len(m.FEATURE_NAMES), m.NAME)


class TestHints(unittest.TestCase):
    def test_observations_and_cards(self):
        F = {"road.right_hand_traffic": 0.05, "road.yellow_center": 0.0, "road.paved": 0.2, "road.markings": 0.0,
             "landscape.soil_red": 0.3}
        tags = [t for t, _ in observations(F)]
        self.assertIn("drive_left", tags)
        self.assertIn("unpaved", tags)
        self.assertIn("red_soil", tags)
        h = build_hints(F, [("KE", 0.6), ("ZA", 0.2)], 2)
        # keyword + region ranking (no clue index): a regional GeoGuessr card is promoted when its region is
        # likely. With data/model/clue_index.npz the calibrated weights decide (tools/build_clue_index.py).
        hb = build_hints({}, [("BR", 0.9)], 1, [{"code": "BR-RS", "name": "Rio Grande do Sul", "country": "BR",
                                                  "probability": 0.9}], kb=ClueBase(index_path=""))
        self.assertTrue(any("BR-RS" in c["regions"] for c in hb["countries"][0]["geoguessr"]))
        self.assertEqual(h["countries"][0]["country_code"], "KE")
        self.assertTrue(h["countries"][0]["driving_side_consistent"])
        self.assertTrue(h["countries"][0]["geoguessr"] or h["countries"][0]["plonkit"])


@unittest.skipUnless(HAVE_MODEL, "trained model missing (tools/train_model.py --save)")
class TestLocator(unittest.TestCase):
    def test_end_to_end(self):
        from engine.locator import get_locator
        res = get_locator().analyze_image(PANO)
        p = [c["probability"] for c in res["countries"]]
        self.assertEqual(p, sorted(p, reverse=True))
        self.assertTrue(0 < sum(p) <= 1.0001)
        self.assertTrue(-90 <= res["guess"]["lat"] <= 90 and -180 <= res["guess"]["lng"] <= 180)
        self.assertTrue(res["hints"])
        self.assertTrue(res["regions"])
        rp = [r["probability"] for r in res["hints"][0]["regions"]]
        self.assertTrue(all(0 <= p <= 1 for p in rp))
        self.assertFalse(math.isnan(res["guess"]["expected_score"]))


if __name__ == "__main__":
    unittest.main()
