"""Tests of engine/fusion.py (moving games: several captures of one round) and its server path.

  python3 -m unittest test_fusion -v
"""
import base64
import io
import os
import random
import sys
import time
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from engine import fusion  # noqa: E402

CLASSES = ["AR", "BR", "CL", "DE"]


def cap(post, prior=None, classes=CLASSES, emb=None, h2=1.0, top=None):
    post = np.asarray(post, float)
    prior = np.full(len(classes), 1.0 / len(classes)) if prior is None else np.asarray(prior, float)
    res = {"countries": [{"code": top or classes[int(np.argmax(post))], "probability": float(post.max())}],
           "guess": {"lat": 0.0, "lng": 0.0}, "observations": [], "hints": [],
           "posterior": dict(zip(classes, post)), "prior": dict(zip(classes, prior)), "input": {"source": "views"}}
    st = {"classes": list(classes), "post": post, "prior": prior, "X": {}, "flat": {}, "det": None, "kind": "views",
          "result": res, "t": time.time()}
    st["emb"] = None if emb is None else np.asarray(emb, float)
    st["h2"] = h2
    return st


class TestWeights(unittest.TestCase):
    def test_weight(self):
        self.assertEqual(fusion.weight(1, 0.7), 1.0)
        self.assertEqual(fusion.weight(4, 0.0), 1.0)                 # plain product
        self.assertAlmostEqual(fusion.weight(4, 1.0), 0.25)          # geometric mean
        self.assertAlmostEqual(4 * fusion.weight(4, 0.5), 1.6)       # effective number of captures

    def test_params_file(self):
        p = fusion.load_params("/nonexistent/fusion.json")
        self.assertEqual(p["rho"], fusion.DEFAULT_PARAMS["rho"])


class TestCountryFusion(unittest.TestCase):
    def test_single_capture_is_its_posterior(self):
        c = cap([0.1, 0.6, 0.2, 0.1])
        _, p = fusion.fuse_country([c], 0.5)
        np.testing.assert_allclose(p, [0.1, 0.6, 0.2, 0.1], rtol=1e-9)

    def test_evidence_relative_to_prior(self):
        prior = [0.4, 0.4, 0.1, 0.1]
        a = cap([0.4, 0.4, 0.1, 0.1], prior)         # no evidence at all
        b = cap([0.2, 0.2, 0.3, 0.3], prior)         # evidence for CL / DE
        _, p = fusion.fuse_country([a, b], 0.0)
        np.testing.assert_allclose(p, [0.2, 0.2, 0.3, 0.3], rtol=1e-9)

    def test_correlated_captures_are_not_over_counted(self):
        a = cap([0.1, 0.6, 0.2, 0.1])
        _, p_prod = fusion.fuse_country([a, a, a], 0.0)
        _, p_geo = fusion.fuse_country([a, a, a], 1.0)
        _, p_mid = fusion.fuse_country([a, a, a], 0.5)
        np.testing.assert_allclose(p_geo, [0.1, 0.6, 0.2, 0.1], rtol=1e-9)   # identical captures add nothing
        self.assertGreater(p_prod[1], p_mid[1])
        self.assertGreater(p_mid[1], p_geo[1])

    def test_latest_prior_and_class_order(self):
        a = cap([0.25, 0.25, 0.25, 0.25])
        b = cap([0.6, 0.2, 0.1, 0.1][::-1], prior=[0.1, 0.1, 0.4, 0.4], classes=CLASSES[::-1])
        classes, p = fusion.fuse_country([a, b], 0.0, CLASSES)
        self.assertEqual(classes, CLASSES)
        # b's evidence (post / prior) in CLASSES order: AR 0.6/0.4, BR 0.2/0.4, CL 0.1/0.1, DE 0.1/0.1; prior = b's
        ev = np.array([1.5, 0.5, 1.0, 1.0]) * np.array([0.4, 0.4, 0.1, 0.1])
        np.testing.assert_allclose(p, ev / ev.sum(), rtol=1e-9)

    def test_zero_probabilities_stay_finite(self):
        a = cap([0.0, 1.0, 0.0, 0.0])
        b = cap([0.5, 0.5, 0.0, 0.0])
        _, p = fusion.fuse_country([a, b], 0.3)
        self.assertTrue(np.all(np.isfinite(p)))
        self.assertAlmostEqual(float(p.sum()), 1.0)
        self.assertEqual(int(np.argmax(p)), 1)


class TestRegionFusion(unittest.TestCase):
    def test_without_region_model_the_shared_distribution_stays(self):
        per = [{"XX": {"XX-A": 0.7, "XX-B": 0.3}}, {"XX": {"XX-A": 0.7, "XX-B": 0.3}}]
        try:
            out = fusion.fuse_regions(per, None, None, 0.5)
        except Exception as e:  # region raster missing
            self.skipTest(repr(e))
        self.assertEqual(out, {})  # codes unknown to the raster are dropped

    def test_real_codes(self):
        try:
            from engine.regions import _regions
            codes = _regions()[1]
        except Exception as e:
            self.skipTest(repr(e))
        de = [c for c in codes if str(c).startswith("DE-")][:2]
        if len(de) < 2:
            self.skipTest("no DE regions")
        per = [{"DE": {de[0]: 0.8, de[1]: 0.2}}, {"DE": {de[0]: 0.6, de[1]: 0.4}}]
        out = fusion.fuse_regions(per, None, None, 0.5)
        # no region model: the captures' geometric mean
        self.assertAlmostEqual(out["DE"][de[0]] + out["DE"][de[1]], 1.0)
        g = np.sqrt([0.8 * 0.6, 0.2 * 0.4])
        self.assertAlmostEqual(out["DE"][de[0]], g[0] / g.sum())

    def test_fallback_is_symmetric_and_never_reverses_a_capture(self):
        try:
            from engine.regions import _regions
            codes = _regions()[1]
        except Exception as e:
            self.skipTest(repr(e))
        de = [c for c in codes if str(c).startswith("DE-")][:2]
        if len(de) < 2:
            self.skipTest("no DE regions")
        dists = [(0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.9, 0.1)]   # only the newest capture has evidence
        per = [{"DE": {de[0]: a, de[1]: b}} for a, b in dists]
        for rho in (0.0, 0.5, 1.0):
            w = fusion.weight(len(per), rho)
            out = fusion.fuse_regions(per, None, None, w)["DE"]
            rev = fusion.fuse_regions(per[::-1], None, None, w)["DE"]
            self.assertAlmostEqual(out[de[0]], rev[de[0]])          # the order of the captures does not matter
            self.assertGreater(out[de[0]], 0.5)                       # the newest capture's evidence keeps its sign
            g = np.exp(np.mean(np.log(np.array(dists)), axis=0))
            self.assertAlmostEqual(out[de[0]], g[0] / g.sum())

    def test_with_region_prior_n4(self):
        class RM:   # a region model with a known prior over two DE regions
            def __init__(self, codes):
                self.cidx = {"DE": 0}
                self.codes = codes

            def prior(self, cc, beta, alpha):
                return np.array([0.3, 0.7])

            def regions_of(self, cc):
                return self.idx
        try:
            from engine.regions import _regions
            codes = list(_regions()[1])
        except Exception as e:
            self.skipTest(repr(e))
        de = [c for c in codes if str(c).startswith("DE-")][:2]
        if len(de) < 2:
            self.skipTest("no DE regions")
        rm = RM(codes)
        rm.idx = [codes.index(de[0]), codes.index(de[1])]
        dists = [(0.3, 0.7), (0.3, 0.7), (0.6, 0.4), (0.3, 0.7)]   # one capture with evidence for de[0]
        per = [{"DE": {de[0]: a, de[1]: b}} for a, b in dists]
        w = fusion.weight(4, 0.5)
        out = fusion.fuse_regions(per, rm, {"weights": {"prior": 1.0}}, w)["DE"]
        # prior x (evidence of the one informative capture)^w
        lr = np.array([0.6 / 0.3, 0.4 / 0.7]) ** w * np.array([0.3, 0.7])
        self.assertAlmostEqual(out[de[0]], lr[0] / lr.sum())
        rev = fusion.fuse_regions(per[::-1], rm, {"weights": {"prior": 1.0}}, w)["DE"]
        self.assertAlmostEqual(out[de[0]], rev[de[0]])


class TestObservations(unittest.TestCase):
    def test_union_keeps_the_strongest(self):
        a = cap([0.25] * 4)
        b = cap([0.25] * 4)
        a["result"]["observations"] = [{"tag": "red_soil", "text": "r", "strength": 0.4}, {"tag": "snow", "text": "s", "strength": 0.9}]
        b["result"]["observations"] = [{"tag": "red_soil", "text": "r", "strength": 0.8}]
        out = fusion.union_observations([a, b])
        self.assertEqual([o["tag"] for o in out], ["snow", "red_soil"])
        self.assertEqual(out[1]["strength"], 0.8)


class TestStore(unittest.TestCase):
    def setUp(self):
        self._orig = fusion.fuse_result

        def fake(loc, caps, map_info=None, params=None, top_k=5):
            _, p = fusion.fuse_country(caps, (params or {}).get("rho", 0.5), CLASSES)
            i = int(np.argmax(p))
            return {"countries": [{"code": CLASSES[i], "probability": float(p[i])}], "guess": {"lat": 0, "lng": 0}}
        fusion.fuse_result = fake

    def tearDown(self):
        fusion.fuse_result = self._orig

    def test_counts_changes_and_duplicates(self):
        s = fusion.FusionStore(max_rounds=4, ttl_s=3600)
        p = {"rho": 0.5, "dup_ratio": 0.05, "max_captures": 8}
        r1 = s.add(None, "game:t#1", cap([0.1, 0.7, 0.1, 0.1], emb=[0, 0]), params=p)
        self.assertEqual(r1["fusion"]["n"], 1)
        self.assertIsNone(r1["fusion"]["changed_top"])
        r2 = s.add(None, "game:t#1", cap([0.8, 0.1, 0.05, 0.05], emb=[5, 0]), params=p)
        self.assertEqual(r2["fusion"]["n"], 2)
        self.assertEqual(r2["fusion"]["prev_top"], "BR")
        self.assertTrue(r2["fusion"]["changed_top"])
        self.assertEqual([c["countries"][0]["code"] for c in r2["fusion"]["captures"]], ["BR", "AR"])
        self.assertIn("capture", r2)
        # the player returns to the start: a re-capture of the first panorama replaces it
        r3 = s.add(None, "game:t#1", cap([0.1, 0.7, 0.1, 0.1], emb=[0.01, 0]), params=p)
        self.assertEqual(r3["fusion"]["n"], 2)
        self.assertEqual(r3["fusion"]["replaced"]["seq"], 1)
        # another round is separate
        r4 = s.add(None, "game:t#2", cap([0.1, 0.1, 0.7, 0.1], emb=[0, 0]), params=p)
        self.assertEqual(r4["fusion"]["n"], 1)

    def test_bounds(self):
        s = fusion.FusionStore(max_rounds=2, ttl_s=3600)
        p = {"rho": 0.5, "dup_ratio": 0.05, "max_captures": 3}
        for k in range(5):
            r = s.add(None, "game:t#1", cap([0.25] * 4, emb=[10 * k, 0]), params=p)
        self.assertEqual(r["fusion"]["n"], 3)                       # oldest captures dropped
        s.add(None, "game:t#2", cap([0.25] * 4, emb=[0, 0]), params=p)
        s.add(None, "game:t#3", cap([0.25] * 4, emb=[0, 0]), params=p)
        self.assertEqual(s.get("game:t#1"), [])                     # least recently used round dropped
        s.ttl_s = 0.0
        time.sleep(0.01)
        self.assertEqual(s.get("game:t#3"), [])                     # expired

    def test_reset(self):
        s = fusion.FusionStore()
        p = {"rho": 0.5, "dup_ratio": 0.05, "max_captures": 8}
        s.add(None, "k", cap([0.25] * 4, emb=[0, 0]), params=p)
        r = s.add(None, "k", cap([0.25] * 4, emb=[9, 0]), params=p, reset=True)
        self.assertEqual(r["fusion"]["n"], 1)


class TestServerRequest(unittest.TestCase):
    def test_round_key_validation(self):
        sys.path.insert(0, os.path.join(ROOT, "web"))
        try:
            import server
        except Exception as e:  # model missing
            self.skipTest(repr(e))
        self.assertIsNone(server.fusion_request({}))
        self.assertEqual(server.fusion_request({"fusion": {"round": "game:abc#2"}}), ("game:abc#2", False))
        self.assertEqual(server.fusion_request({"fusion": {"round": "replay:abc#1@k3x9"}})[0], "replay:abc#1@k3x9")
        for bad in ({"round": 5}, {"round": "a b"}, "x", {"round": "x" * 200}, {"round": "game:abc#null"},
                    {"round": "game:abc#undefined"}, {"round": "game:abc#0"}, {"round": "challenge:AbC#"},
                    {"round": "page:abc#1"}, {"round": "game:abc#2\n"}, {"round": "game:a.b#2"}):
            with self.assertRaises(ValueError):
                server.fusion_request({"fusion": bad})


@unittest.skipUnless(os.environ.get("FUSION_SLOW", "1") == "1", "slow integration test switched off")
class TestLivePath(unittest.TestCase):
    """The server's predict() with a round key on live-grid renderings of two dataset panoramas."""

    @classmethod
    def setUpClass(cls):
        try:
            from calibrate import DATASET, load_records, split_of
            sys.path.insert(0, os.path.join(ROOT, "web"))
            import server
            recs = [r for r in load_records(pixels=True) if split_of(r) == "calib" and r.get("heading") is not None]
        except Exception as e:
            raise unittest.SkipTest(repr(e))
        if len(recs) < 2:
            raise unittest.SkipTest("no dataset panoramas")
        cls.server, cls.recs, cls.dataset = server, recs[:2], DATASET

    def views(self, r, seed):
        from eval_live import render_views
        from engine.panorama import SphericalImage
        sph = SphericalImage.from_equirect(os.path.join(self.dataset, "panos", r["pano_id"] + ".jpg"), heading=r["heading"])
        out = []
        for v in render_views(sph, random.Random(seed).uniform(0, 360)):
            buf = io.BytesIO()
            v["image"].save(buf, "JPEG", quality=92)   # as the page sends them
            out.append({"image_b64": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(),
                        "yaw": v["yaw"], "pitch": v["pitch"], "hfov": v["hfov"]})
        return out

    def test_round_fusion(self):
        srv = self.server
        key = "replay:fusion%d#1" % int(time.time() * 1000)
        a = srv.predict({"views": self.views(self.recs[0], "a"), "fusion": {"round": key}})
        self.assertEqual(a["fusion"]["n"], 1)
        for k in ("countries", "guess", "hints", "observations", "regions"):
            self.assertIn(k, a)
        self.assertAlmostEqual(sum(a["posterior"].values()), 1.0, places=2)
        b = srv.predict({"views": self.views(self.recs[1], "b"), "fusion": {"round": key}})
        self.assertEqual(b["fusion"]["n"], 2, b["fusion"])
        self.assertIsNotNone(b["fusion"]["changed_top"])
        self.assertEqual(len(b["fusion"]["captures"]), 2)
        self.assertEqual(b["capture"]["countries"][0]["code"], b["fusion"]["captures"][1]["countries"][0]["code"])
        self.assertIn("fusion", b["timing_ms"])
        # the first panorama again from another start yaw (the player returned): replaces, not added
        c = srv.predict({"views": self.views(self.recs[0], "c"), "fusion": {"round": key}})
        self.assertEqual(c["fusion"]["n"], 2, c["fusion"])
        self.assertIsNotNone(c["fusion"]["replaced"], c["fusion"])
        # no key: the plain single-capture answer
        d = srv.predict({"views": self.views(self.recs[1], "b")})
        self.assertNotIn("fusion", d)


if __name__ == "__main__":
    unittest.main()
