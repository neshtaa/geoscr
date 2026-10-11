#!/usr/bin/env python3
"""
Tests for the GeoGuessr-style areas (engine/areas.py, data/areas.json), their use in the hints and the HUD, and the
region-model additions (pool prior, extra log-likelihood terms):
    python3 -m unittest test_regions_areas -v
"""
import json
import os
import shutil
import subprocess
import unittest

import numpy as np

from engine import areas as A
from engine.geo import _regions, region_index

ROOT = os.path.dirname(os.path.abspath(__file__))


def _read(path):
    with open(path, "rb") as f:
        return f.read()


class TestAreasFile(unittest.TestCase):
    def test_curated_cover_every_unit_once(self):
        with open(A.AREAS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        _, codes, _, rcc, _ = _regions()
        own = {}
        for i in range(1, len(codes)):
            own.setdefault(rcc[i], set()).add(codes[i])
        self.assertGreater(len(data["countries"]), 40)
        for cc, v in data["countries"].items():
            self.assertTrue(v.get("source"), cc)
            seen = []
            for a in v["areas"]:
                self.assertTrue(a["name"] and a["en"] and a["id"].startswith(cc + ":"), (cc, a))
                seen += a["regions"]
            hidden = v.get("hidden", [])
            self.assertEqual(len(seen), len(set(seen)), "%s: a unit in two areas" % cc)
            self.assertFalse(set(seen) & set(hidden), "%s: a hidden unit in an area" % cc)
            self.assertEqual(set(seen) | set(hidden), own.get(cc, set()),
                             "%s: areas + hidden must cover the raster units exactly" % cc)
            # a one-to-one "grouping" only repeats the region list in the HUD
            multi = sum(1 for a in v["areas"] if len(a["regions"]) > 1)
            self.assertGreaterEqual(2 * multi, len(v["areas"]), "%s: mostly single-unit areas" % cc)
        ids = [a["id"] for v in data["countries"].values() for a in v["areas"]]
        self.assertEqual(len(ids), len(set(ids)))
        for code, pt in data["unit_points"].items():
            self.assertTrue(-90 <= pt[0] <= 90 and -180 <= pt[1] <= 180, code)

    def test_known_groupings(self):
        def area_of(cc, code):
            return next(a["en"] for a in A.country_areas(cc) if code in a["regions"])
        self.assertEqual(area_of("US", "US-IA"), "West North Central")
        self.assertEqual(area_of("US", "US-TX"), "West South Central")
        self.assertEqual(area_of("BR", "BR-BA"), "Northeast")
        self.assertEqual(area_of("RU", "RU-NVS"), "Siberian")
        self.assertEqual(area_of("RU", "RU-RUS99"), "Ural")
        self.assertEqual(area_of("IN", "IN-KL"), "South")
        self.assertEqual(area_of("ID", "ID-BA"), "Bali and Nusa Tenggara")
        self.assertEqual(area_of("MX", "MX-YUC"), "Yucatan Peninsula")
        self.assertEqual(area_of("KH", "KH-24"), "Tonle Sap")   # Pailin
        self.assertEqual(area_of("GB", "GB-EDH"), "Scotland")
        self.assertEqual(area_of("TR", "TR-34"), "Marmara")
        # from the Natural Earth "region" attribute (tools/make_areas.py)
        self.assertEqual(area_of("SI", "SI-050"), "Coastal-Karst")      # Koper
        self.assertEqual(area_of("SI", "SI-070"), "Drava")              # Maribor (a unit without raster pixels)
        self.assertEqual(area_of("SI", "SI-061"), "Central Slovenia")   # Ljubljana
        self.assertEqual(area_of("LV", "LV-RIX"), "Riga region")
        self.assertEqual(area_of("UG", "UG-102"), "Central")            # Kampala
        self.assertEqual(area_of("HK", "HK-KG5154"), "Hong Kong Island")   # Wan Chai
        self.assertEqual(area_of("MK", "MK-85"), "Skopje")
        self.assertEqual(area_of("XK", "XK-OS5908"), "Prizren")
        self.assertEqual(area_of("ME", "ME-10"), "Coastal")             # Kotor
        self.assertEqual(area_of("DO", "DO-01"), "Santo Domingo (Ozama)")

    def test_no_one_to_one_layers(self):
        for cc in ("ZA", "CA", "AU", "AE"):   # provinces / states as they are: the region list says it all
            self.assertIsNone(A.country_areas(cc), cc)

    def test_crimea_not_russia(self):
        self.assertEqual(A.hidden_units("RU"), {"UA-43", "UA-40"})
        self.assertFalse({"UA-43", "UA-40"} & {c for a in A.country_areas("RU") for c in a["regions"]})
        self.assertNotIn("Crimea", [a["en"] for a in A.country_areas("RU")])
        g = A.group_regions({"UA-43": 0.5, "RU-KDA": 0.5}, "RU")
        self.assertAlmostEqual(sum(a["probability"] for a in g), 0.5)
        from engine.hints import build_hints
        from engine.regions import region_list
        regs = [dict(r, country="RU") for r in region_list({"UA-43": 0.5, "RU-KDA": 0.3, "RU-ROS": 0.2})]
        h = build_hints({}, [("RU", 1.0)], 1, regs)
        self.assertEqual([r["code"] for r in h["countries"][0]["regions"]], ["RU-KDA", "RU-ROS"])

    def test_region_names(self):
        self.assertEqual(A.region_name("US-DC"), "District of Columbia")
        self.assertEqual(A.region_name("BR-DF"), "Distrito Federal")
        self.assertEqual(A.region_name("US-TX"), "Texas")
        self.assertEqual(A.region_name("XK-OS5908"), "Prizren")   # the raster's name_en says "Plužine" (Montenegro)
        self.assertEqual(A.region_name("XX-NOPE"), "XX-NOPE")


class TestAutoAreas(unittest.TestCase):
    def test_auto_cover_and_deterministic(self):
        for cc in ("MD", "TZ", "AZ", "BF"):   # not curated, >= AUTO_MIN_UNITS units
            self.assertNotIn(cc, A.load()["countries"])
            a1 = A.auto_areas(cc)
            self.assertIsNotNone(a1, cc)
            self.assertTrue(3 <= len(a1) <= A.AUTO_MAX_AREAS)
            units = {_regions()[1][i] for i in A.units_of(cc)}
            got = [c for a in a1 for c in a["regions"]]
            self.assertEqual(sorted(got), sorted(units))
            self.assertEqual(len({a["name"] for a in a1}), len(a1), "auto area names must be distinct")
            self.assertFalse(any("(" in a["name"] for a in a1), "%s: plain compass names expected" % cc)
            self.assertEqual(a1, A.auto_areas(cc))
        self.assertIsNone(A.country_areas("EE"))   # 15 units, not curated: no areas
        self.assertIsNone(A.country_areas("ZZ"))

    def test_auto_directions(self):
        a = A.auto_areas("MD")
        where = {c: x["en"] for x in a for c in x["regions"]}
        self.assertTrue(where["MD-OC"].startswith(("North", "Northwest", "Northeast")), where["MD-OC"])   # Ocnita
        self.assertTrue(where["MD-TA"].startswith(("South", "Southwest", "Southeast")), where["MD-TA"])   # Taraclia

    def test_units_without_pixels(self):
        # Seychelles: 12 of 26 units have no raster pixel; each joins the area nearest to its label point
        from engine.regions import region_centroids
        km2 = region_centroids()[2]
        codes = _regions()[1]
        pts = A.load()["unit_points"]
        a = A.auto_areas("SC")
        where = {c: x for x in a for c in x["regions"]}
        zero = [codes[i] for i in A.units_of("SC") if km2[i] == 0]
        self.assertGreater(len(zero), 5)
        for c in zero:
            self.assertEqual(c in where, c in pts, c)


class TestGrouping(unittest.TestCase):
    def test_group_regions(self):
        probs = {"US-TX": 0.3, "US-OK": 0.1, "US-CA": 0.2, "US-WA": 0.1, "US-DC": 0.05, "US-IA": 0.25}
        g = A.group_regions(probs, "US", top_regions=2)
        self.assertEqual(g[0]["en"], "West South Central")
        self.assertAlmostEqual(g[0]["probability"], 0.4)
        self.assertEqual(g[0]["size"], 4)
        self.assertEqual([r["code"] for r in g[0]["regions"]], ["US-TX", "US-OK"])
        self.assertAlmostEqual(sum(a["probability"] for a in g), 1.0)
        self.assertEqual(g[-1]["regions"][0]["name"], "District of Columbia")
        self.assertIsNone(A.group_regions(probs, "EE"))
        lab = A.area_labels("US", [_regions()[1].index(c) for c in ("US-TX", "US-OK", "US-CA")])
        self.assertEqual(lab[0], lab[1])
        self.assertNotEqual(lab[0], lab[2])

    def test_hints_carry_areas(self):
        from engine.hints import area_hint, build_hints
        from engine.regions import region_list
        regs = region_list({"US-TX": 0.3, "US-OK": 0.1, "US-CA": 0.2, "BR-SP": 0.2, "BR-MG": 0.2})
        h = build_hints({}, [("US", 0.6), ("BR", 0.4)], 2, regs)
        us = h["countries"][0]
        self.assertEqual(us["country_code"], "US")
        self.assertEqual(us["areas"][0]["en"], "West South Central")
        self.assertAlmostEqual(us["areas"][0]["probability"], 0.667, places=3)
        self.assertEqual(us["regions"][0]["name"], "Texas")
        self.assertEqual(h["countries"][1]["areas"][0]["en"], "Southeast")
        self.assertIsNone(area_hint("EE", {"EE-37": 1.0}))
        self.assertIsNone(area_hint("US", {}))

    @unittest.skipUnless(shutil.which("node"), "node not installed")
    def test_hud_helpers(self):
        js = ("const H=require(%s);" % json.dumps(os.path.join(ROOT, "web", "hud", "hud.js")) +
              "const hint={areas:[{name:'Великі рівнини',en:'West North Central',probability:0.41,size:7,"
              "regions:[{code:'US-IA',name:'Iowa',probability:0.2}]},{name:'<b>x</b>',probability:0.2,size:2}],"
              "regions:[{code:'US-IA',name:'Iowa',probability:0.2},{code:'US-KS',name:'Kansas',probability:0.09},"
              "{code:'US-TX',name:'Texas',probability:0.08}]};"
              "const es={areas:[{name:'Мадрид',probability:0.5,size:1,regions:[{code:'ES-M',name:'Madrid',"
              "probability:0.5}]},{name:'Каталонія',probability:0.3,size:4,regions:[{code:'ES-B',name:'Barcelona',"
              "probability:0.2}]}],regions:[{code:'ES-M',name:'Madrid',probability:0.5},{code:'ES-B',"
              "name:'Barcelona',probability:0.2}]};"
              "const out={full:H.regionLinesHtml(hint,false),compact:H.regionLinesHtml(hint,true),"
              "tight:H.regionLinesHtml(hint,false,true),es:H.regionLinesHtml(es,false),"
              "tight_plain:H.regionLinesHtml({regions:hint.regions},false,true),"
              "plain:H.regionLinesHtml({regions:hint.regions},false),"
              "mix:H.areaMixture([{country_code:'US',probability:0.5,areas:hint.areas},{country_code:'AU',"
              "probability:0.3,regions:[]}]),"
              "mix2:H.areaMixture([{country_code:'US',probability:0.5,areas:hint.areas},{country_code:'EE',"
              "probability:0.4,regions:[{name:'Harju',probability:0.9}]}]),"
              "none:H.areaMixture([{country_code:'EE',probability:0.4,regions:[{name:'Harju',probability:0.9}]}]),"
              "iso:[H.isoCountry({code:'UA-43',country:'RU'}),H.isoCountry({code:'x',country:'RU'})]};"
              "console.log(JSON.stringify(out));")
        out = json.loads(subprocess.check_output(["node", "-e", js], timeout=60).decode())
        self.assertIn('class="chip area"', out["full"])
        self.assertIn("Великі рівнини <b>41%</b>", out["full"])
        self.assertIn("regions sub", out["full"])
        self.assertIn("регіони:", out["full"])                 # the sub-region row says what it is
        self.assertNotIn("<b>x</b>", out["full"])             # escaped
        self.assertEqual(out["compact"].count('class="chip area"'), 2)
        self.assertEqual(out["tight"].count('class="chip area"'), 2)   # a short panel: one row, 2 areas
        self.assertEqual(out["tight"].count('class="chip reg"'), 0)
        self.assertIn("Iowa", out["tight"])                                # (their regions in the tooltip)
        self.assertEqual(out["tight_plain"].count('class="chip reg"'), 3)   # no areas: the regions
        self.assertEqual(out["full"].count('class="chip reg"'), 3)
        # an area of one unit (Madrid) is not repeated as a region chip
        self.assertEqual(out["es"].count("Madrid"), 1)        # only in the area chip's tooltip
        self.assertNotIn('chip reg">Madrid', out["es"])
        self.assertIn('chip reg">Barcelona', out["es"])
        self.assertNotIn("chip area", out["plain"])
        self.assertNotIn("регіони:", out["plain"])
        self.assertEqual([m["name"] for m in out["mix"]], ["Великі рівнини", "<b>x</b>"])   # AU: no regions given
        self.assertAlmostEqual(out["mix"][0]["p"], 0.205)
        self.assertEqual([m["name"] for m in out["mix2"]][:2], ["Harju", "Великі рівнини"])
        self.assertEqual(out["none"], [])   # no country with areas: the HUD keeps its region list
        self.assertEqual(out["iso"], ["UA", "RU"])
        with open(os.path.join(ROOT, "web", "hud", "hud.js"), encoding="utf-8") as f:
            src = f.read()
        self.assertIn('<div class="lbl">Частини країн</div>', src)   # groups of oblasts are not "Області"
        self.assertNotIn('<div class="lbl">Області</div>', src)
        for f in ("hud.js", "hud.css"):
            self.assertEqual(_read(os.path.join(ROOT, "web", "hud", f)), _read(os.path.join(ROOT, "extension", "hud", f)),
                             "extension/hud/%s is stale: cp web/hud/%s extension/hud/" % (f, f))


class TestPoolPrior(unittest.TestCase):
    def test_frozen_pool_reproduces_counts(self):
        import sys
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        import eval_regions as ER
        with open(os.path.join(ROOT, "data", "calibration", "duel_rounds.json")) as f:
            r0 = json.load(f)["rounds"][0]
        lat = np.array([r0["lat"], 10.0, 48.5, 50.45])
        lng = np.array([r0["lng"], 20.0, 31.2, 30.52])
        D = {"split": np.array(["calib", "train", "world_calib", "train"]), "lat": lat, "lng": lng,
             "recs": [{"pano_id": "p%d" % i} for i in range(4)], "region": region_index(lat, lng)}
        c0, i0 = ER.pool_counts(D)                     # default: no World-map rounds
        self.assertEqual(i0["world_rounds"], 0)
        self.assertIn(0, i0["excluded_rounds"])        # the duel round at the calib round's position
        c1, i1 = ER.pool_counts(D, world_weight=2.0)
        self.assertEqual(i1["world_round_ids"], ["p2"])
        self.assertAlmostEqual(c1.sum() - c0.sum(), 2.0)
        # frozen: the same counts from the recorded exclusions / ids, whatever the records are now
        D2 = dict(D, split=np.array(["train", "train", "world_calib", "world_test"]))
        c2, i2 = ER.pool_counts(D2, frozen=json.loads(json.dumps(i1)))
        np.testing.assert_array_equal(c1, c2)
        self.assertTrue(i2["frozen"])


class TestRegionModelAdditions(unittest.TestCase):
    def setUp(self):
        from test_regions import PARAMS, synthetic
        self.X, self.cc, self.lat, self.lng = synthetic()
        self.reg = region_index(self.lat, self.lng)
        self.PARAMS = PARAMS

    def test_pool_prior(self):
        from engine.regions import RegionModel
        codes = _regions()[1]
        duel = np.ones(len(self.cc), bool)
        m0 = RegionModel(self.PARAMS).fit(self.X, self.cc, self.reg, self.lat, self.lng, duel)
        pool = np.zeros(len(codes))
        pool[codes.index("AU-WA")] = 500.0   # the pool puts most rounds where no reference panorama is
        pool[codes.index("AU-QLD")] = 50.0
        m1 = RegionModel(self.PARAMS).fit(self.X, self.cc, self.reg, self.lat, self.lng, duel, pool=pool)
        regs = list(m1.regions_of("AU"))
        wa = regs.index(codes.index("AU-WA"))
        self.assertGreater(m1.prior("AU")[wa], 0.6)
        self.assertLess(m0.prior("AU")[wa], 0.2)
        np.testing.assert_allclose(m0.mu, m1.mu)   # appearance unchanged
        # random (non-duel) references still count with beta
        m2 = RegionModel(dict(self.PARAMS, beta=1.0)).fit(self.X, self.cc, self.reg, self.lat, self.lng,
                                                          np.zeros(len(self.cc), bool), pool=pool)
        self.assertGreater(m2.count_rand.sum(), 0)

    def test_extra_terms(self):
        from engine.regions import RegionModel
        m = RegionModel(self.PARAMS).fit(self.X, self.cc, self.reg, self.lat, self.lng, np.ones(len(self.cc), bool))
        x = {"road": np.array([[0.0, 0.0]]), "texture": np.array([[0.0]])}
        comp = m.components(m.embed(x, "AU"), "AU")
        R = len(m.regions_of("AU"))
        t = np.zeros((1, R))
        t[0, 0] = 5.0
        comp["cards"] = t
        p0 = m.combine(comp, "AU", {"weights": dict(m.params["weights"], cards=0.0)})[0]
        p1 = m.combine(comp, "AU", {"weights": dict(m.params["weights"], cards=1.0)})[0]
        self.assertGreater(p1[0], p0[0])
        self.assertAlmostEqual(p1.sum(), 1.0)

    def test_glm_term(self):
        import tempfile
        from engine.regions import RegionModel, region_posterior
        codes = _regions()[1]
        duel = np.ones(len(self.cc), bool)
        m = RegionModel(dict(self.PARAMS, glm_iters=200)).fit(self.X, self.cc, self.reg, self.lat, self.lng, duel)
        regs = list(m.regions_of("AU"))
        qld, vic = regs.index(codes.index("AU-QLD")), regs.index(codes.index("AU-VIC"))
        x = {"road": np.array([[4.0, 0.0]]), "texture": np.array([[0.0]])}
        T = m.glm_term(m.embed_all(x), "AU")
        self.assertEqual(T.shape, (1, len(regs)))
        self.assertGreater(T[0, qld], T[0, vic] + 1.0)
        self.assertIsNone(m.glm_term(m.embed_all(x), "NZ"))   # one region with references: no GLM
        w = dict(m.params["weights"], glm=1.0, lda=0.0, knn=0.0)
        p = region_posterior(x, {"AU": 1.0}, m, params=dict(m.params, weights=w))
        self.assertEqual(max(p, key=p.get), "AU-QLD")
        tmp = tempfile.mkdtemp()
        try:
            m.save(tmp)
            m2 = RegionModel.load(tmp)
            np.testing.assert_allclose(m2.glm_term(m2.embed_all(x), "AU"), T, atol=1e-4)
            q = region_posterior(x, {"AU": 1.0}, m2, params=dict(m2.params, weights=w))
            for k in p:
                self.assertAlmostEqual(p[k], q[k], places=4)
        finally:
            shutil.rmtree(tmp)

    def test_float32_fit_matches_float64_inputs(self):
        from engine.regions import RegionModel
        duel = np.ones(len(self.cc), bool)
        a = RegionModel(self.PARAMS).fit(self.X, self.cc, self.reg, self.lat, self.lng, duel)
        b = RegionModel(self.PARAMS).fit({g: v.astype(np.float32) for g, v in self.X.items()}, self.cc, self.reg,
                                         self.lat, self.lng, duel)
        np.testing.assert_allclose(a.mu, b.mu, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
