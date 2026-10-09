#!/usr/bin/env python3
"""
Tests for the admin-1 region model (engine/regions.py):
    python3 -m unittest test_regions -v
"""
import os
import shutil
import tempfile
import unittest

import numpy as np

from engine.geo import _regions, region_index
from engine.model import GeoModel
from engine.regions import (MODEL_DIR, REGIONS_FILE, SUN_LAT_GRID, RegionModel, location_weights, region_list,
                            region_posterior, region_vector, sun_mixture)

PARAMS = {"K": 2, "K_big": 2, "tau": 1.0, "knn_k": 5, "weights": {"prior": 1.0, "lda": 1.0, "knn": 0.3, "sun": 0.0}}
# two regions of Australia, one of New Zealand: (code, lat, lng, feature centre)
SITES = [("AU-QLD", -23.0, 145.0, (4.0, 0.0)), ("AU-VIC", -37.0, 144.5, (-4.0, 0.0)),
         ("NZ-CAN", -43.5, 171.5, (0.0, 4.0))]
# + mainland France and Reunion, a GeoModel class that the raster files under France (FR-RE): (label, code, ...)
SITES_FR = [("AU",) + s for s in SITES] + [("FR", "FR-IDF", 48.85, 2.35, (2.0, 2.0)),
                                           ("RE", "FR-RE", -21.12, 55.53, (0.0, -4.0))]
QLD_BOX = {"min": {"lat": -29.0, "lng": 139.0}, "max": {"lat": -10.0, "lng": 154.0}}
NZ_BOX = {"min": {"lat": -47.0, "lng": 166.0}, "max": {"lat": -34.0, "lng": 179.0}}


def synthetic(n=40, seed=0, sites=None):
    rng = np.random.RandomState(seed)
    sites = sites or [(code[:2], code, la, ln, ctr) for code, la, ln, ctr in SITES]
    X, cc, lat, lng = [], [], [], []
    for label, code, la, ln, ctr in sites:
        sd = 0.3 if code[:2] != "FR" else 0.03  # Reunion is ~50 km across
        X.append(rng.randn(n, 2) + ctr)
        cc += [label] * n
        lat.append(la + sd * rng.randn(n))
        lng.append(ln + sd * rng.randn(n))
    X, lat, lng = np.vstack(X), np.concatenate(lat), np.concatenate(lng)
    noise = rng.randn(len(X), 1)
    return {"road": X, "texture": noise}, np.array(cc), lat, lng


class TestRegionModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, cc, lat, lng = synthetic()
        cls.reg = region_index(lat, lng)
        codes = _regions()[1]
        cls.code = np.array([codes[r] for r in cls.reg])
        duel = np.arange(len(cc)) % 2 == 0
        cards = [("AU", [codes.index("AU-QLD"), codes.index("AU-NSW")], 3)]
        cls.m = RegionModel(PARAMS).fit(X, cc, cls.reg, lat, lng, duel, cards)
        cls.tmp = tempfile.mkdtemp()
        cls.m.save(cls.tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_sites_resolved(self):
        self.assertTrue(all(c in ("AU-QLD", "AU-VIC", "NZ-CAN") for c in self.code))

    def test_given_country(self):
        x = {"road": np.array([4.0, 0.0]), "texture": np.array([0.0])}
        p = region_posterior(x, {"AU": 1.0}, self.m)
        self.assertAlmostEqual(sum(p.values()), 1.0, places=6)
        self.assertEqual(max(p, key=p.get), "AU-QLD")
        self.assertTrue(all(k.startswith("AU-") for k in p))
        self.assertGreater(p["AU-QLD"], 0.8)
        x["road"] = np.array([-4.0, 0.0])
        p = region_posterior(x, {"AU": 1.0}, self.m)
        self.assertEqual(max(p, key=p.get), "AU-VIC")
        # regions without references keep a little mass (area-smoothed prior)
        self.assertGreater(p.get("AU-WA", 0.0), 0.0)

    def test_mixture_over_countries(self):
        x = {"road": np.array([0.0, 4.0]), "texture": np.array([0.0])}
        p = region_posterior(x, {"AU": 0.3, "NZ": 0.7}, self.m)
        self.assertAlmostEqual(sum(p.values()), 1.0, places=6)
        self.assertAlmostEqual(sum(v for k, v in p.items() if k.startswith("NZ-")), 0.7, places=6)
        self.assertEqual(max(p, key=p.get), "NZ-CAN")
        per = region_posterior(x, {"AU": 0.3, "NZ": 0.7}, self.m, by_country=True)
        self.assertEqual(set(per), {"AU", "NZ"})
        self.assertAlmostEqual(sum(per["AU"].values()), 1.0, places=6)
        mix, per2 = region_posterior(x, {"AU": 0.3, "NZ": 0.7}, self.m, by_country="both")
        self.assertEqual(mix, p)
        self.assertEqual(per2, per)
        # vector + classes form
        q = region_posterior(x, np.array([0.3, 0.7]), self.m, classes=["AU", "NZ"])
        self.assertAlmostEqual(q["NZ-CAN"], p["NZ-CAN"], places=9)

    def test_country_without_model(self):
        x = {"road": np.array([0.0, 0.0]), "texture": np.array([0.0])}
        p = region_posterior(x, {"CL": 1.0}, self.m)
        self.assertAlmostEqual(sum(p.values()), 1.0, places=6)
        self.assertTrue(all(k.startswith("CL-") for k in p))
        self.assertGreater(len(p), 5)

    def test_missing_features(self):
        x = {"road": np.array([np.nan, np.nan]), "texture": np.array([np.nan])}
        p = region_posterior(x, {"AU": 1.0}, self.m)
        self.assertAlmostEqual(sum(p.values()), 1.0, places=6)
        self.assertTrue(np.isfinite(list(p.values())).all())
        q = region_posterior({"road": np.array([np.nan, np.nan])}, {"AU": 1.0}, self.m)  # a group left out
        for k in p:
            self.assertAlmostEqual(p[k], q[k], places=9)

    def test_save_load(self):
        m2 = RegionModel.load(self.tmp)
        x = {"road": np.array([1.0, 0.5]), "texture": np.array([0.2])}
        a = region_posterior(x, {"AU": 0.6, "NZ": 0.4}, self.m)
        b = region_posterior(x, {"AU": 0.6, "NZ": 0.4}, m2)
        self.assertEqual(set(a), set(b))
        for k in a:
            self.assertAlmostEqual(a[k], b[k], places=5)

    def test_smoothing(self):
        x = {"road": np.array([4.0, 0.0]), "texture": np.array([0.0])}
        e = self.m.embed(x, "AU")
        comp = self.m.components(e, "AU")
        regs = list(self.m.regions_of("AU"))
        nsw = regs.index(_regions()[1].index("AU-NSW"))
        p0 = self.m.combine(comp, "AU", {"smooth": 0.0, "smooth_card": 0.0})[0]
        p1 = self.m.combine(comp, "AU", {"smooth_card": 0.3})[0]
        p2 = self.m.combine(comp, "AU", {"smooth": 0.3, "smooth_km": 800.0})[0]
        for p in (p0, p1, p2):
            self.assertAlmostEqual(p.sum(), 1.0, places=9)
        self.assertGreater(p1[nsw], p0[nsw])  # the card groups QLD with NSW
        self.assertGreater(p2[nsw], p0[nsw])  # NSW borders QLD
        self.assertIsNone(self.m.card_kernel("NZ", self.m.prior("NZ")))

    def test_quantile_transform(self):
        self.assertIsNotNone(self.m.qn)
        x = np.linspace(-6, 6, 50)
        X = np.zeros((50, 3))
        X[:, 0] = x
        z = self.m._pre(X)[:, 0]
        self.assertTrue((np.diff(z) >= 0).all())
        self.assertLess(z[0], -1.5)
        self.assertGreater(z[-1], 1.5)
        X[3, 0] = np.nan
        self.assertTrue(np.isnan(self.m._pre(X)[3, 0]))
        self.assertTrue(np.isfinite(self.m.transform(X)).all())
        m0 = RegionModel(dict(PARAMS, pre=""))
        Xg, cc, lat, lng = synthetic()
        m0.fit(Xg, cc, self.reg, lat, lng, np.ones(len(cc), bool))
        self.assertIsNone(m0.qn)

    def test_prior_mask(self):
        X, cc, lat, lng = synthetic()
        duel = np.ones(len(cc), bool)
        m1 = RegionModel(PARAMS).fit(X, cc, self.reg, lat, lng, duel)
        mask = self.code != "AU-QLD"
        m2 = RegionModel(PARAMS).fit(X, cc, self.reg, lat, lng, duel, prior_mask=mask)
        regs = list(m1.regions_of("AU"))
        qld = regs.index(_regions()[1].index("AU-QLD"))
        self.assertLess(m2.prior("AU")[qld], 0.5 * m1.prior("AU")[qld])
        # the appearance of QLD is still learnt from its (prior-excluded) references
        np.testing.assert_allclose(m1.mu, m2.mu)
        x = {"road": np.array([4.0, 0.0]), "texture": np.array([0.0])}
        self.assertEqual(max(region_posterior(x, {"AU": 1.0}, m2).items(), key=lambda t: t[1])[0], "AU-QLD")

    def test_vector_and_list(self):
        p = {"AU-QLD": 0.6, "AU-VIC": 0.3, "NZ-CAN": 0.1}
        v = region_vector(p)
        codes = _regions()[1]
        self.assertEqual(len(v), len(codes))
        self.assertAlmostEqual(v[codes.index("AU-VIC")], 0.3)
        self.assertAlmostEqual(v.sum(), 1.0)
        lst = region_list(p)
        self.assertEqual([r["code"] for r in lst], ["AU-QLD", "AU-VIC", "NZ-CAN"])
        self.assertEqual(lst[2]["country"], "NZ")

    def test_location_weights(self):
        geo = GeoModel()
        geo.classes = ["AU", "NZ"]
        X, cc, lat, lng = synthetic(seed=1)
        geo.ref_yi = np.array([geo.classes.index(c) for c in cc])
        geo.ref_lat, geo.ref_lng = lat, lng
        geo.loc_params = {"bw": 1.0, "floor": 0.0}
        geo._index_refs()
        lp = np.log(np.array([0.8, 0.2]))
        d2 = np.ones(len(cc))
        rp = {"AU": {"AU-QLD": 0.9, "AU-VIC": 0.1}, "NZ": {"NZ-CAN": 1.0}}
        w = location_weights(geo, lp, d2, rp)
        code = np.array([_regions()[1][r] for r in geo.ref_region])
        self.assertAlmostEqual(w.sum(), 1.0, places=9)
        self.assertAlmostEqual(w[code == "AU-QLD"].sum(), 0.72, places=6)
        self.assertAlmostEqual(w[code == "AU-VIC"].sum(), 0.08, places=6)
        self.assertAlmostEqual(w[code == "NZ-CAN"].sum(), 0.2, places=6)
        g = geo.locate(lp, d2, w=w)
        self.assertLess(abs(g["lat"] + 23.0), 2.0)
        # without region probabilities: the GeoModel's own weights
        np.testing.assert_allclose(location_weights(geo, lp, d2, {}), geo.ref_weights(lp, d2, 1.0, 0.0))

    def test_bounds(self):
        x = {"road": np.array([-4.0, 0.0]), "texture": np.array([0.0])}  # looks like VIC
        p = region_posterior(x, {"AU": 1.0}, self.m, bounds=QLD_BOX)
        self.assertAlmostEqual(sum(p.values()), 1.0, places=6)
        self.assertAlmostEqual(p["AU-QLD"], 1.0, places=6)  # the only region with its centroid / references inside
        self.assertEqual(p.get("AU-VIC", 0.0), 0.0)
        # a box without any region of the country: the bounds are ignored for it
        q = region_posterior(x, {"AU": 1.0}, self.m, bounds=NZ_BOX)
        p0 = region_posterior(x, {"AU": 1.0}, self.m)
        self.assertEqual(set(q), set(p0))
        for k in p0:
            self.assertAlmostEqual(q[k], p0[k], places=9)
        # a reference panorama inside the box keeps its region even when the centroid is outside
        box = {"min": {"lat": -38.0, "lng": 144.0}, "max": {"lat": -36.0, "lng": 145.0}}
        self.assertGreater(region_posterior(x, {"AU": 1.0}, self.m, bounds=box).get("AU-VIC", 0.0), 0.99)
        # countries without a region model: the centroid rule
        c = region_posterior(x, {"CL": 1.0}, self.m, bounds={"min": {"lat": -35.0, "lng": -75.0},
                                                             "max": {"lat": -32.0, "lng": -69.0}})
        self.assertAlmostEqual(sum(c.values()), 1.0, places=6)
        self.assertEqual(sorted(k for k, v in c.items() if v > 0), ["CL-LI", "CL-RM", "CL-VS"])

    def test_location_weights_bounds(self):
        geo = GeoModel()
        geo.classes = ["AU", "NZ"]
        X, cc, lat, lng = synthetic(seed=1)
        geo.ref_yi = np.array([geo.classes.index(c) for c in cc])
        geo.ref_lat, geo.ref_lng = lat, lng
        geo.loc_params = {"bw": 1.0, "floor": 0.0}
        geo._index_refs()
        lp = np.log(np.array([0.8, 0.2]))
        d2 = np.ones(len(cc))
        rp = {"AU": {"AU-QLD": 0.5, "AU-VIC": 0.5}, "NZ": {"NZ-CAN": 1.0}}
        w = location_weights(geo, lp, d2, rp, bounds=QLD_BOX)
        inside = geo.refs_inside(QLD_BOX)
        code = np.array([_regions()[1][r] for r in geo.ref_region])
        self.assertAlmostEqual(w[inside].sum(), 1.0, places=9)
        self.assertEqual(w[~inside].sum(), 0.0)
        self.assertAlmostEqual(w[code == "AU-QLD"].sum(), 1.0, places=9)
        g = geo.locate(lp, d2, w=w, bounds=QLD_BOX)
        self.assertLess(abs(g["lat"] + 23.0), 2.0)

    def test_sun_term(self):
        x = {"road": np.array([0.0, 0.0]), "texture": np.array([0.0])}
        e = self.m.embed(x, "AU")
        regs, codes = list(self.m.regions_of("AU")), _regions()[1]
        qld, vic = regs.index(codes.index("AU-QLD")), regs.index(codes.index("AU-VIC"))
        w = {"weights": {"prior": 1.0, "lda": 0.0, "knn": 0.0, "sun": 1.0}}
        for lat0, hi, lo in ((-23.0, qld, vic), (-37.0, vic, qld)):
            L = np.exp(-0.5 * ((SUN_LAT_GRID - lat0) / 3.0) ** 2)[None]
            P = self.m.combine(self.m.components(e, "AU", L), "AU", w)[0]
            self.assertGreater(P[hi], 5 * P[lo])
        # a flat latitude likelihood (no sun) changes nothing
        P0 = self.m.combine(self.m.components(e, "AU"), "AU", w)[0]
        P1 = self.m.combine(self.m.components(e, "AU", np.ones((1, len(SUN_LAT_GRID)))), "AU", w)[0]
        np.testing.assert_allclose(P0, P1, atol=1e-9)

    def test_sun_mixture(self):
        from engine.features import solar
        n = solar.FEATURE_NAMES
        row = np.full(len(n), np.nan)
        np.testing.assert_allclose(sun_mixture({"solar": row}), 1.0)  # no sun
        cols = [n.index(k) for k in ("sun_az_cos", "sun_az_sin", "sun_el", "sun_conf")]
        row[cols] = [-1.0, 0.0, 60.0, 1.0]  # high sun due south: northern hemisphere
        S = sun_mixture({"solar": row})[0]
        self.assertAlmostEqual(S.mean(), 1.0, places=6)
        self.assertGreater(S[SUN_LAT_GRID == 30.0][0], 10 * S[SUN_LAT_GRID == -30.0][0])
        row[cols[3]] = 0.0  # zero confidence = no information
        np.testing.assert_allclose(sun_mixture({"solar": row}), 1.0)
        # region_posterior feeds the solar features through the sun term
        m = RegionModel.load(self.tmp)
        m.params["weights"] = {"prior": 1.0, "lda": 0.0, "knn": 0.0, "sun": 1.0}
        row[cols] = [1.0, 0.0, 40.0, 1.0]  # sun due north
        x = {"road": np.array([0.0, 0.0]), "texture": np.array([0.0]), "solar": row}
        p = region_posterior(x, {"AU": 1.0}, m)
        regs = m.regions_of("AU")
        P = m.combine(m.components(m.embed(x, "AU"), "AU", sun_mixture(x)), "AU")[0]
        codes = _regions()[1]
        for r, q in zip(regs, P):
            self.assertAlmostEqual(p[codes[r]], q, places=9)
        P0 = m.combine(m.components(m.embed(x, "AU"), "AU"), "AU")[0]
        self.assertGreater(np.abs(P - P0).max(), 1e-3)

    def test_truncation(self):
        x = {"road": np.array([0.0, 4.0]), "texture": np.array([0.0])}
        p = region_posterior(x, {"AU": 0.6, "NZ": 0.4}, self.m, top_countries=1)
        self.assertTrue(all(k.startswith("AU-") for k in p))
        self.assertAlmostEqual(sum(p.values()), 1.0, places=6)
        p = region_posterior(x, {"AU": 1 - 5e-5, "NZ": 5e-5}, self.m, min_prob=1e-4)
        self.assertFalse(any(k.startswith("NZ-") for k in p))
        self.assertAlmostEqual(sum(p.values()), 1.0, places=6)
        p = region_posterior(x, {"AU": 2e-5, "NZ": 5e-5}, self.m, min_prob=1e-4)  # none above: the top country
        self.assertTrue(all(k.startswith("NZ-") for k in p))

    def test_alias_country(self):
        X, cc, lat, lng = synthetic(sites=SITES_FR)
        reg = region_index(lat, lng)
        codes = _regions()[1]
        self.assertTrue((np.array([codes[r] for r in reg[cc == "RE"]]) == "FR-RE").mean() > 0.9)
        m = RegionModel(PARAMS).fit(X, cc, reg, lat, lng, np.ones(len(cc), bool))
        re_ = codes.index("FR-RE")
        self.assertEqual(list(m.regions_of("RE")), [re_])
        self.assertNotIn(re_, list(m.regions_of("FR")))
        self.assertGreater(m.count[m.reg_off[m.cidx["RE"]]], 30)
        x = {"road": np.array([0.0, -4.0]), "texture": np.array([0.0])}
        self.assertEqual(region_posterior(x, {"RE": 1.0}, m), {"FR-RE": 1.0})
        p = region_posterior(x, {"RE": 0.97, "AU": 0.03}, m)
        self.assertAlmostEqual(p["FR-RE"], 0.97, places=6)
        self.assertAlmostEqual(sum(v for k, v in p.items() if k.startswith("AU-")), 0.03, places=6)
        # a class without a region model: its alias unit (Martinique) or, without any region, no mass
        p = region_posterior(x, {"MQ": 0.97, "AU": 0.03}, m)
        self.assertAlmostEqual(p["FR-MQ"], 0.97, places=6)
        p = region_posterior(x, {"ZZ": 0.5, "AU": 0.5}, m)
        self.assertAlmostEqual(sum(p.values()), 0.5, places=6)
        self.assertEqual(region_posterior(x, {"ZZ": 1.0}, m), {})
        v = region_vector(region_posterior(x, {"RE": 1.0}, m))
        self.assertAlmostEqual(v[re_], 1.0)


@unittest.skipUnless(os.path.exists(os.path.join(MODEL_DIR, REGIONS_FILE)), "data/model/regions.npz not built")
class TestSavedModel(unittest.TestCase):
    def test_saved(self):
        m = RegionModel.load()
        rng = np.random.RandomState(0)
        x = {g: rng.randn(d) for g, d in zip(m.groups, m.dims)}
        p = region_posterior(x, {"US": 0.5, "CA": 0.3, "BR": 0.2}, m)
        self.assertAlmostEqual(sum(p.values()), 1.0, places=6)
        self.assertAlmostEqual(sum(v for k, v in p.items() if k.startswith("US-")), 0.5, places=6)
        self.assertGreater(len(p), 60)
        # GeoModel classes that the raster files under France keep their own region
        for cc, code in (("RE", "FR-RE"), ("MQ", "FR-MQ")):
            self.assertIn(cc, m.cidx)
            q = region_posterior(x, {cc: 0.97, "US": 0.03}, m)
            self.assertAlmostEqual(q[code], 0.97, places=6)


if __name__ == "__main__":
    unittest.main()
