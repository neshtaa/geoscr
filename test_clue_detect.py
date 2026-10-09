#!/usr/bin/env python3
"""
Tests for the GeoGuessr clue-card detectors (engine/clue_detect.py) and their use in the hints:
    python3 -m unittest test_clue_detect -v
"""
import json
import math
import os
import time
import unittest

import numpy as np

from engine import clue_detect as cd
from engine.hints import ClueBase, detected_look_text
from engine.panorama import SphericalImage


def stripes_pano(az=60.0, el=-5.0, w=2048, seed=0):
    """Noisy grey-green panorama with a red-white striped post centred at relative longitude az, elevation el."""
    rng = np.random.RandomState(seed)
    h = w // 2
    img = np.empty((h, w, 3), np.float32)
    img[:] = (90, 110, 80)
    img[: h // 2] = (150, 170, 200)
    img += rng.normal(0, 12, img.shape)
    lon = (np.arange(w) + 0.5) / w * 360 - 180
    lat = 90 - (np.arange(h) + 0.5) / h * 180
    cols = np.abs((lon - az + 180) % 360 - 180) < 1.5
    rows = np.abs(lat - el) < 7.0
    band = ((lat // 1.5) % 2 == 0)[:, None]
    post = rows[:, None] & cols[None, :]
    img[post & band] = (220, 30, 30)
    img[post & ~band] = (245, 245, 245)
    return np.clip(img, 0, 255).astype(np.uint8)


def synthetic_bank(pano, az, el, channel=2, k=60, arrays=False):
    """Bank with one whitened-LDA detector of the striped post for country ZZ (background = all windows)."""
    sph = SphericalImage(pano)
    g = cd.grid(cd.CHANNELS[channel])
    cells = cd.sphere_cells(sph)[channel][0]
    X = cd.windows(cells).astype(np.float64).reshape(g["n_wrows"], g["n_az"], cd.DIM)
    mu = X.mean(1)
    R = (X - mu[:, None]).reshape(-1, cd.DIM)
    S = R.T @ R / len(R) + 1e-4 * np.eye(cd.DIM)
    lam, U = np.linalg.eigh(S)
    o = np.argsort(-lam)[:k]
    T = (U[:, o] / np.sqrt(lam[o])).T
    i, j = cd.window_at(g, az, el)
    u = T @ (X[i, j] - mu[i])
    G = T.T @ (u / np.linalg.norm(u))[:, None]
    r = cd.window_at(g, 0.0, el)[0]
    meta = {"channels": list(cd.CHANNELS),
            "detectors": [{"country": "ZZ", "cards": ["zz-post"], "type": "pole", "group": False, "channel": channel,
                           "rows": [max(r - 3, 0), min(r + 3, g["n_wrows"] - 1)], "n": 1, "freq": 0.3}],
            "cards": {"ZZ": {"zz-post": {"freq": 0.3, "det": 0, "regions": ["ZZ-A"], "type": "pole"},
                             "zz-other": {"freq": 0.3, "det": None, "regions": [], "type": "misc"}}},
            "presence": {"a": [0.0, 1.0, 1.0, 0.5], "hint_min_prob": 0.5,
                         "direction": {"a": [-2.0, 0.0, 1.0], "min_prob": 0.5}},
            "region": {"weight": 1.0}}
    arr = {"meta": json.dumps(meta), "zmu": np.array([2.0]), "zsd": np.array([1.0]),
           "cm_classes": np.array(["YY", "ZZ"]), "cm_med": np.zeros(1), "cm_sc": np.ones(1),
           "cm_A": np.array([[0.0, 1.0]]), "cm_b": np.zeros(2)}
    for c in range(len(cd.CHANNELS)):
        gc = cd.grid(cd.CHANNELS[c])
        arr["G%d" % c] = (G if c == channel else np.zeros((cd.DIM, 0))).astype(np.float32)
        arr["O%d" % c] = (mu @ G if c == channel else np.zeros((gc["n_wrows"], 0))).astype(np.float32)
    return arr if arrays else cd.ClueDetectors(arr)


class TestGeometry(unittest.TestCase):
    def test_fov_rule(self):
        # GeoGuessr: tan(hfov / 2) = 2^(1 - zoom) on the long side; the window is the 9/16 central square
        self.assertAlmostEqual(cd.crop_fov(1.0), math.degrees(2 * math.atan(9 / 16.0)), 6)
        self.assertGreater(cd.crop_fov(1.5), cd.crop_fov(2.0))
        self.assertEqual(cd.channel_of(2.5), 1)
        self.assertEqual(cd.channel_of(3.0), 2)
        self.assertEqual(cd.channel_of(0.0), 0)

    def test_grid_round_trip(self):
        for ch in cd.CHANNELS:
            g = cd.grid(ch)
            self.assertAlmostEqual(g["cell"] * cd.CELLS, g["fov"], delta=0.05 * g["fov"])
            self.assertGreaterEqual(float(cd.window_pitch(g, 0)), ch["pitch"][1] - 1e-9)
            self.assertLessEqual(float(cd.window_pitch(g, g["n_wrows"] - 1)), ch["pitch"][0] + 1e-9)
            for yaw, pitch in ((37.0, -6.0), (-179.0, 3.0), (179.5, 0.0)):
                i, j = cd.window_at(g, yaw, pitch)
                self.assertLessEqual(abs(float(cd.window_pitch(g, i)) - pitch), g["cell"] / 2 + 1e-9)
                self.assertLessEqual(abs((float(cd.window_yaw(g, j)) - yaw + 180) % 360 - 180), g["cell"] / 2 + 1e-9)


class TestCells(unittest.TestCase):
    def test_shapes_and_flat_image(self):
        sph = SphericalImage(np.full((1024, 2048, 3), 128, np.uint8))
        cells = cd.sphere_cells(sph)
        for ch, (q, valid) in zip(cd.CHANNELS, cells):
            g = cd.grid(ch)
            self.assertEqual(q.shape, (g["n_rows"], g["n_az"], cd.CF))
            self.assertEqual(q.dtype, np.uint8)
            self.assertIsNone(valid)
            self.assertEqual(int(q[..., :cd.NBINS].max()), 0)  # no gradients in a flat image
            X = cd.windows(q)
            self.assertEqual(X.shape, (g["n_wrows"] * g["n_az"], cd.DIM))

    def test_wraps_around(self):
        """Rolling the panorama by 180 deg rolls the cell grid by half (x wraps, no seam): a post across the
        seam gives the same cells as in the middle of the image."""
        ch = cd.CHANNELS[2]
        g = cd.grid(ch)
        self.assertEqual(g["n_az"] % 2, 0)
        pano = stripes_pano(179.5, 0.0)
        a = cd.channel_cells(pano, None, ch)[0].astype(int)
        b = cd.channel_cells(np.roll(pano, 1024, axis=1), None, ch)[0].astype(int)
        self.assertGreater(np.mean(np.abs(np.roll(a, g["n_az"] // 2, axis=1) - b) <= 3), 0.98)

    def test_mask(self):
        sph = SphericalImage(stripes_pano())
        sph.mask = np.zeros_like(sph.mask)
        sph.mask[:, : sph.w // 2] = True
        q, valid = cd.sphere_cells(sph)[1]
        self.assertIsNotNone(valid)
        self.assertGreater(valid[:, : q.shape[1] // 2 - 1].min(), 0.95)
        self.assertLess(valid[:, q.shape[1] // 2 + 1:].max(), 0.05)
        wv = cd.window_valid(valid)
        self.assertEqual(wv.shape, (q.shape[0] - cd.CELLS + 1, q.shape[1]))


class TestDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.az, cls.el = 60.0, -5.0
        cls.pano = stripes_pano(cls.az, cls.el)
        cls.bank_arrays = synthetic_bank(cls.pano, cls.az, cls.el, arrays=True)
        cls.bank = cd.ClueDetectors(cls.bank_arrays)

    def test_finds_direction(self):
        other = stripes_pano(-100.0, self.el, seed=3)  # same post elsewhere, panorama facing heading 30
        det = self.bank.detect(SphericalImage(other, heading=30.0))
        g = cd.grid(cd.CHANNELS[2])
        self.assertLessEqual(abs((det["yaw"][0] + 100.0 + 180) % 360 - 180), 1.5 * g["cell"])
        self.assertLessEqual(abs(det["pitch"][0] - self.el), 3 * g["cell"])
        empty = self.bank.detect(SphericalImage(stripes_pano(0.0, 80.0, seed=4)))  # post outside the pitch rows
        self.assertGreater(det["z"][0], empty["z"][0] + 1.0)
        self.assertEqual(det["coverage"], 1.0)
        cards = self.bank.detected_cards("ZZ", det, min_prob=0.0)
        self.assertEqual(cards[0]["stem"], "zz-post")
        self.assertTrue(cards[0]["true_north"] and cards[0]["first"] and cards[0]["direction"])
        self.assertGreater(cards[0]["p_dir"], 0.5)
        self.assertLessEqual(abs((cards[0]["heading"] - 290.0 + 180.0) % 360.0 - 180.0), 1.5 * g["cell"])

    def test_partial_sphere(self):
        """Screenshots that do not cover the post: windows outside the mask never fire."""
        sph = SphericalImage(self.pano)
        sph.mask = np.zeros_like(sph.mask)
        sph.mask[:, : sph.w // 3] = True  # relative longitudes -180..-60 only
        det = self.bank.detect(sph)
        self.assertTrue(not np.isfinite(det["z"][0]) or det["yaw"][0] < -60.0)
        self.assertTrue(np.isfinite(self.bank.zt(det["z"])).all())
        self.assertLess(abs(det["coverage"] - 1 / 3.0), 0.08)
        # maxima over a third of the windows are not comparable with full panoramas: no country evidence
        det["zt"] = self.bank.zt(det["z"])
        self.assertIsNone(self.bank.country_evidence(det, ["YY", "ZZ"]))
        self.assertEqual(self.bank.country_evidence(dict(det, coverage=0.95), ["YY", "ZZ"]).shape, (1, 2))

    def test_presence_country_and_hint_flags(self):
        zt_hi, zt_lo = np.array([3.0]), np.array([-1.0])
        p_hi, p_lo = self.bank.card_presence("ZZ", zt_hi), self.bank.card_presence("ZZ", zt_lo)
        self.assertGreater(p_hi["zz-post"], p_lo["zz-post"])
        self.assertAlmostEqual(p_hi["zz-other"], p_lo["zz-other"])  # no detector: frequency only
        ll = self.bank.country_loglik(np.array([[3.0], [-1.0]]), ["YY", "ZZ", "XX"])
        self.assertEqual(ll.shape, (2, 3))
        self.assertTrue(np.allclose(ll.max(1), 0.0))
        self.assertGreater(ll[0, 1] - ll[0, 0], ll[1, 1] - ll[1, 0])
        det = {"z": np.array([5.0]), "zt": np.array([3.0]), "yaw": np.array([10.0]), "pitch": np.array([-4.0]),
               "heading": None}
        d = self.bank.detected_cards("ZZ", det, min_prob=0.99)  # below the hint threshold, P(direction) 0.73
        self.assertEqual(len(d), 1)
        self.assertFalse(d[0]["first"] or d[0]["true_north"])
        self.assertTrue(d[0]["direction"])
        self.assertAlmostEqual(d[0]["p_dir"], 1 / (1 + math.exp(-1.0)), 6)
        self.assertEqual(self.bank.detected_cards("ZZ", dict(det, zt=np.array([1.5])), min_prob=0.99), [])
        # without a calibrated direction model / thresholds nothing leads and no direction is shown
        bare = cd.ClueDetectors(dict(self.bank_arrays, meta=json.dumps(dict(self.bank.meta, presence={"a": [0, 1, 1, 0]}))))
        self.assertEqual(bare.detected_cards("ZZ", det), [])

    def test_region_update(self):
        rp = {"ZZ-A": 0.2, "ZZ-B": 0.8}
        up = self.bank.update_regions("ZZ", np.array([4.0]), rp)
        self.assertAlmostEqual(sum(up.values()), 1.0)
        self.assertGreater(up["ZZ-A"], 0.2)
        self.assertEqual(self.bank.update_regions("ZZ", np.array([4.0]), {}), {})


class TestHints(unittest.TestCase):
    def test_detected_first_and_directions(self):
        kb = ClueBase(index_path=None)
        cc = next(c for c, cards in kb.gg.items() if len(cards) >= 4)
        for c in kb.gg[cc]:  # placement views (from the clue index otherwise)
            c.update(view={"pitch": -12.0, "zoom": 2.0}, category="road")
        base = kb.for_country(cc, [], n_gg=3)
        target = next(c for c in kb.gg[cc] if c["id"] not in {x["id"] for x in base["geoguessr"]})
        d = {"stem": target["id"], "prob": 0.8, "p_dir": 0.55, "score": 2.5, "heading": 230.0, "pitch": -12.0,
             "true_north": True, "zoom": 2.0, "first": True, "direction": True}
        shown = next(x for x in base["geoguessr"][1:] if x.get("look"))
        d2 = {"stem": shown["id"], "prob": 0.2, "p_dir": 0.6, "score": 1.4, "heading": 15.0, "pitch": 2.0,
              "true_north": True, "zoom": 3.5, "first": False, "direction": True}
        res = kb.for_country(cc, [], n_gg=3, detected=[d, d2])
        first = res["geoguessr"][0]
        self.assertEqual(first["id"], target["id"])
        self.assertEqual(first["detected"]["heading"], 230.0)
        self.assertEqual(first["detected"]["probability"], 0.55)  # P(direction right), not P(present)
        self.assertIn("можливо, азимут 230°, вниз", first["look"])
        self.assertIn("Імовірна за частотою в країні й детектором", first["matched"])
        self.assertFalse(any("Знайдено" in m for c in res["geoguessr"] for m in c["matched"]))
        other = {c["id"]: c for c in res["geoguessr"][1:]}
        self.assertIn(shown["id"], other)  # a direction-only detection does not move the card
        o = other[shown["id"]]
        self.assertEqual(o["detected"]["heading"], 15.0)
        self.assertTrue(o["look"].startswith(shown["look"]))  # the placements' look stays, the detector is a hedge
        self.assertEqual(o["view"], shown["view"])
        self.assertEqual(len({c["id"] for c in res["geoguessr"]}), 3)
        self.assertIn("від центру", detected_look_text(dict(d, true_north=False, heading=350.0)))
        weak = kb.for_country(cc, [], n_gg=3, detected=[dict(d, direction=False)])["geoguessr"][0]
        self.assertEqual(weak["id"], target["id"])  # leads on P(present), but no direction to show
        self.assertNotIn("detected", weak)
        quiet = kb.for_country(cc, [], n_gg=3, detected=[dict(d, first=False, direction=False)])
        self.assertEqual([c["id"] for c in quiet["geoguessr"]], [c["id"] for c in base["geoguessr"]])


@unittest.skipUnless(os.path.exists(cd.DETECTORS_PATH), "no data/model/clue_detectors.npz")
class TestSavedBank(unittest.TestCase):
    def test_bank_runs(self):
        bank = cd.ClueDetectors.load()
        self.assertIsNotNone(bank)
        self.assertEqual(sum(len(i) for i in bank.idx), len(bank.dets))
        for c in range(len(bank.channels)):
            self.assertEqual(bank.G[c].shape, (cd.DIM, len(bank.idx[c])))
            self.assertEqual(bank.O[c].shape, (cd.grid(bank.channels[c])["n_wrows"], len(bank.idx[c])))
        sph = SphericalImage(stripes_pano(), heading=10.0)
        bank.detect(sph)
        t = time.time()
        det = bank.detect(sph)
        dt = time.time() - t
        self.assertEqual(len(det["z"]), len(bank.dets))
        self.assertTrue(np.isfinite(det["z"]).all())
        self.assertLess(dt, 1.5)  # ~0.2-0.4 s on an idle laptop
        ll = bank.country_loglik(bank.zt(det["z"])[None], ["BR", "US", "ZZ"])
        self.assertTrue(np.isfinite(ll).all())
        for cc in ("BR", "US"):
            for d in bank.detected_cards(cc, det, min_prob=0.0):
                self.assertTrue(0 <= d["heading"] < 360 and -30 <= d["pitch"] <= 30)
        # no detector has an empty cross-fitting fold (its out-of-fold scores would be constant)
        self.assertTrue(all(min(d.get("n_fold", [1, 1])) >= 1 for d in bank.dets))


if __name__ == "__main__":
    unittest.main()
