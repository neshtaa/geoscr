#!/usr/bin/env python3
"""
Tests for the GeoGuessr card ranking (engine/hints.py + data/model/clue_index.npz):
    python3 -m unittest test_clue_hints -v
"""
import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from engine.hints import (DEFAULT_RANK, INDEX_PATH, MIN_SUPPORT, ClueBase, ClueIndex, build_hints, clue_stem,
                          look_text, zoom_fov)

PARAMS = dict(DEFAULT_RANK, kw=0.0, region=0.0, knn=0.0, freq=1.0, rvote=0.0, k=2, bw=1.0, beta=0.1, beta_r=0.1)


def synthetic_index(path, params=PARAMS):
    """Country 'ZZ', two features, four labelled panoramas:
         p0 (0, 0) region ZZ-A: common, north    p1 (0, 1) region ZZ-A: common, north
         p2 (5, 5) region ZZ-B: common, south    p3 (9, 9) no features: common, rare"""
    cards = [{"id": "zz-common", "country": "ZZ", "title": "Common", "description": "On most roads.",
              "type": "pole", "category": "country", "pitch": 3.0, "zoom": 3.5, "seterra": []},
             {"id": "zz-north", "country": "ZZ", "title": "North", "description": "Northern hills.",
              "type": "nature", "category": "regional", "pitch": 0.0, "zoom": 1.5, "seterra": []},
             {"id": "zz-south", "country": "ZZ", "title": "South", "description": "Southern road lines.",
              "type": "road", "category": "regional", "pitch": -14.0, "zoom": 2.0, "seterra": []},
             {"id": "zz-rare", "country": "ZZ", "title": "Rare", "description": "Seldom seen.",
              "type": "misc", "category": "country", "seterra": []}]
    rows = [[0, 1], [0, 1], [0, 2], [0, 3]]
    ptr = np.cumsum([0] + [len(r) for r in rows])
    np.savez(path, meta=np.array(json.dumps({"params": params, "cards": cards})),
             feat_names=np.array(["road.a", "road.b"]), med=np.zeros(2), sc=np.ones(2), proj=np.eye(2),
             emb=np.array([[0, 0], [0, 1], [5, 5], [np.nan, np.nan]], np.float32),
             country=np.array(["ZZ"] * 4), region=np.array(["ZZ-A", "ZZ-A", "ZZ-B", ""]),
             pano_ids=np.array(["p0", "p1", "p2", "p3"]), card_ptr=ptr.astype(np.int32),
             card_idx=np.array([c for r in rows for c in r], np.int32))


class TestClueIndex(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.path = os.path.join(cls.tmp, "clue_index.npz")
        synthetic_index(cls.path)
        cls.kb = ClueBase(index_path=cls.path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def ranked(self, F=None, regions=None, params=None, exclude=None):
        idx = self.kb.index
        emb = idx.embed(F) if F else None
        return [c["id"] for _, c, _, _ in self.kb.rank_cards("ZZ", [], regions, emb, exclude, params)]

    def test_frequency(self):
        sc = self.kb.index.card_scores("ZZ")
        self.assertAlmostEqual(sc["freq"]["zz-common"], 1.0)
        self.assertAlmostEqual(sc["freq"]["zz-north"], 0.5)
        self.assertEqual(self.ranked()[:2], ["zz-common", "zz-north"])

    def test_knn_prefers_cards_of_similar_panoramas(self):
        p = dict(PARAMS, knn=1.0, freq=0.0)
        self.assertEqual(self.ranked({"road.a": 5.0, "road.b": 4.8}, params=p)[:2], ["zz-common", "zz-south"])
        self.assertEqual(self.ranked({"road.a": 0.0, "road.b": 0.4}, params=p)[:2], ["zz-common", "zz-north"])
        # no usable features -> the kNN vote falls back to the frequency
        self.assertEqual(self.ranked({"vehicle.x": 1.0}, params=p)[:2], ["zz-common", "zz-north"])

    def test_region_vote(self):
        p = dict(PARAMS, rvote=1.0, freq=0.0)
        self.assertEqual(self.ranked(regions={"ZZ-B": 1.0}, params=p)[:2], ["zz-common", "zz-south"])
        v = self.kb.index.card_scores("ZZ", region_probs={"ZZ-B": 0.5, "ZZ-X": 0.5}, params=p)["rvote"]
        self.assertGreater(v["zz-south"], 0.25 * 0.5)  # half of the mass sits in a region without labels
        self.assertLess(v["zz-south"], 1.0)

    def test_leave_one_out(self):
        sc = self.kb.index.card_scores("ZZ", exclude="p3")
        self.assertNotIn("zz-rare", sc["freq"])
        self.assertAlmostEqual(sc["freq"]["zz-north"], 2.0 / 3.0)
        self.assertEqual(self.kb.index.card_scores("XX"), {"freq": {}, "knn": {}, "rvote": {}})
        self.assertEqual(self.kb.index.support("ZZ", {"ZZ-A": 0.5}, exclude="p0"), (3, 0.5))

    def test_build_hints_cards_and_look(self):
        h = build_hints({"road.a": 0.0, "road.b": 0.2}, [("ZZ", 0.7)], 1, kb=self.kb, n_cards=2)
        cards = h["countries"][0]["geoguessr"]
        self.assertEqual([c["id"] for c in cards], ["zz-common", "zz-north"])
        self.assertIn("стовпи", cards[0]["look"])
        self.assertEqual(cards[0]["view"]["fov"], 20)
        # four labelled panoramas are too few to call a card frequent
        self.assertLess(4, MIN_SUPPORT)
        self.assertEqual(cards[0]["matched"], [])


class TestReasons(unittest.TestCase):
    """Country 'ZZ' with 12 labelled panoramas: 'zz-pole' on all, 'zz-hill' on the six near (0, 0),
    'zz-coast' on the six near (8, 8); 'zz-flag' is a catalogue card without placements."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.path = os.path.join(cls.tmp, "clue_index.npz")
        cards = [{"id": "zz-pole", "country": "ZZ", "title": "Pole", "description": "Wooden poles along the road.",
                  "type": "pole", "pitch": 2.0, "zoom": 2.0},
                 {"id": "zz-hill", "country": "ZZ", "title": "Hills", "description": "Rolling green hills inland.",
                  "type": "nature", "seterra": ["ISO-ZZ-A"]},
                 {"id": "zz-coast", "country": "ZZ", "title": "Coast", "description": "Sandy coast in the south.",
                  "type": "nature"}]
        rows = [[0, 1]] * 6 + [[0, 2]] * 6
        emb = [[0.0, 0.1 * i] for i in range(6)] + [[8.0, 8.0 + 0.1 * i] for i in range(6)]
        ptr = np.cumsum([0] + [len(r) for r in rows])
        params = dict(DEFAULT_RANK, kw=0.0, region=0.0, knn=0.5, freq=0.0, rvote=0.5, k=4, bw=1.0, beta=0.5,
                      beta_r=0.5)
        np.savez(cls.path, meta=np.array(json.dumps({"params": params, "cards": cards})),
                 feat_names=np.array(["road.a", "road.b"]), med=np.zeros(2), sc=np.ones(2), proj=np.eye(2),
                 emb=np.array(emb, np.float32), country=np.array(["ZZ"] * 12),
                 region=np.array(["ZZ-A"] * 6 + ["ZZ-B"] * 6), pano_ids=np.array(["p%d" % i for i in range(12)]),
                 card_ptr=ptr.astype(np.int32), card_idx=np.array([c for r in rows for c in r], np.int32))
        cls.kb = ClueBase(index_path=cls.path)
        cls.kb._merge("ZZ", "zz-flag", "Flag", "A red flag with poles", {})

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def card(self, cards, cid):
        return next(c for c in cards if c["id"] == cid)

    def test_similar_regional_frequent(self):
        F = {"road.a": 8.0, "road.b": 8.2}
        h = build_hints(F, [("ZZ", 0.9)], 1, kb=self.kb, n_cards=3)
        cards = h["countries"][0]["geoguessr"]
        self.assertEqual([c["id"] for c in cards[:2]], ["zz-pole", "zz-coast"])
        self.assertIn("Є на схожих панорамах", self.card(cards, "zz-coast")["matched"])
        self.assertIn("Часта картка країни", self.card(cards, "zz-pole")["matched"])
        regions = [{"code": "ZZ-A", "name": "A", "country": "ZZ", "probability": 1.0}]
        cards = build_hints({}, [("ZZ", 0.9)], 1, regions, kb=self.kb)["countries"][0]["geoguessr"]
        self.assertEqual([c["id"] for c in cards[:2]], ["zz-pole", "zz-hill"])
        self.assertIn("Часта в імовірному регіоні", self.card(cards, "zz-hill")["matched"])

    def test_zero_weight_matches_are_only_seen(self):
        F = {"structure.pole_n360": 6.0}  # poles observed; keywords carry no weight in these params
        cards = build_hints(F, [("ZZ", 0.9)], 1, kb=self.kb, n_cards=4)["countries"][0]["geoguessr"]
        flag = self.card(cards, "zz-flag")
        self.assertEqual(flag["matched"], [])
        self.assertIn("Стовпи ЛЕП уздовж дороги", flag["seen"])

    def test_country_without_labels_uses_keywords_and_regions(self):
        kb = ClueBase(index_path=self.path)
        ranked = [c["id"] for _, c, _, _ in kb.rank_cards("BA", [("red_soil", 1.0)])]
        default = [c["id"] for _, c, _, _ in kb.rank_cards("BA", [("red_soil", 1.0)], params=DEFAULT_RANK)]
        self.assertEqual(ranked, default)
        self.assertEqual(kb.index.support("BA"), (0, 0.0))
        regions = {"BR-RS": 1.0}
        top = kb.rank_cards("BR", [], regions)[0][1]
        self.assertIn("BR-RS", top["regions"])
        kb._merge("QQ", "qq-plain", "Plain", "Nothing special about this one.", {})
        kb._merge("QQ", "qq-soil", "Soil", "Deep red soil next to the road.", {})
        s, top, why, seen = kb.rank_cards("QQ", [("red_soil", 1.0)])[0]
        self.assertEqual((top["id"], why, seen), ("qq-soil", ["red_soil"], []))


class TestFallback(unittest.TestCase):
    def test_missing_index(self):
        kb = ClueBase(index_path=os.path.join(tempfile.gettempdir(), "no_such_clue_index.npz"))
        self.assertIsNone(kb.index)
        regions = [{"code": "BR-RS", "name": "Rio Grande do Sul", "country": "BR", "probability": 0.9}]
        h = build_hints({}, [("BR", 0.9)], 1, regions, kb=kb)
        cards = h["countries"][0]["geoguessr"]
        self.assertEqual(len(cards), 3)
        self.assertTrue(any("BR-RS" in c["regions"] for c in cards))
        # keyword ranking still works without the index
        F = {"road.right_hand_traffic": 0.05, "road.paved": 0.2, "landscape.soil_red": 0.3}
        h = build_hints(F, [("KE", 0.6), ("ZA", 0.2)], 2, kb=kb)
        self.assertTrue(h["countries"][0]["driving_side_consistent"])
        self.assertTrue(any(c["matched"] for c in h["countries"][0]["geoguessr"]))

    def test_corrupt_index(self):
        tmp = tempfile.mkdtemp()
        try:
            p = os.path.join(tmp, "clue_index.npz")
            with open(p, "wb") as fh:
                fh.write(b"not an npz")
            self.assertIsNone(ClueIndex.load(p))
            self.assertIsNone(ClueBase(index_path=p).index)
        finally:
            shutil.rmtree(tmp)

    def test_positional_signature(self):
        h = build_hints({}, [("JP", 0.5), ("KR", 0.3)], 2, [])
        self.assertEqual([c["country_code"] for c in h["countries"]], ["JP", "KR"])

    @unittest.skipUnless(os.path.exists(INDEX_PATH), "data/model/clue_index.npz missing (tools/build_clue_index.py)")
    def test_shipped_index(self):
        idx = ClueIndex.load()
        self.assertIsNotNone(idx)
        self.assertEqual(idx.emb.shape[1], idx.proj.shape[1])
        self.assertTrue(all(c.get("id") and c.get("country") for c in idx.cards))
        h = build_hints({}, [("BR", 0.9)], 1)
        self.assertEqual(len(h["countries"][0]["geoguessr"]), 3)


class TestCatalogue(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = ClueBase(index_path="")

    def test_stem(self):
        self.assertEqual(clue_stem("clue.br-ladder-poles-title"), "br-ladder-poles")
        self.assertEqual(clue_stem("clue.co-licenseplate-desc"), "co-licenseplate")

    def test_one_card_per_clue(self):
        norm = self.kb._norm
        for cc, cards in self.kb.gg.items():
            ids = [i for c in cards for i in [c["id"]] + c["aliases"]]
            self.assertEqual(len(ids), len(set(ids)), cc)
            texts = [(norm(c.get("title")), norm(c.get("description"))) for c in cards
                     if len(norm(c.get("description"))) >= 20]
            self.assertEqual(len(texts), len(set(texts)), cc)

    def test_duplicate_text_keeps_placement_key(self):
        kb = ClueBase(index_path="")
        kb._merge("QQ", "qq-master", "Shield", "A blue highway shield with a white number.", {})
        kb._merge("QQ", "qq-67-shield", "Shield", "A blue highway shield with a white number!",
                  {"gg_id": "QQ-67", "seterra": ["ISO-US-TX"]})
        self.assertEqual(len(kb.gg["QQ"]), 1)
        self.assertEqual(kb.gg["QQ"][0]["id"], "qq-67-shield")
        self.assertEqual(kb.card_id("QQ", "qq-master"), "qq-67-shield")
        self.assertEqual(kb.gg["QQ"][0]["regions"], ["US-TX"])

    def test_region_codes(self):
        rc = self.kb._region_codes
        self.assertEqual(rc("NO", ["AREA_NORWAY_VIKEN"]), ["NO-01", "NO-02", "NO-06"])
        self.assertEqual(rc("NO", ["AREA_NORWAY_TRONDELAG"]), ["NO-16", "NO-17"])
        for cc, i in (("IT", "AREA_ITALY"), ("AT", "AREA_AUSTRIA"), ("CL", "AREA_CHILE"), ("PL", "AREA_POLAND"),
                      ("VN", "AREA_VIETNAM"), ("US", "AREA_THEUNITEDSTATES"), ("NL", "AREA_THENETHERLANDS")):
            self.assertEqual(rc(cc, [i]), [], i)
        self.assertEqual(len(rc("ES", ["AREA_CASTILELEON"])), 9)
        self.assertEqual(rc("ES", ["AREA_VALENCIA"]), ["ES-A", "ES-CS", "ES-V"])
        self.assertEqual(rc("ZA", ["AREA_SAFRICA_NORTHERN", "AREA_SAFRICA_NORTHERNCAPE"]), ["ZA-LP", "ZA-NC"])
        self.assertEqual(rc("IN", ["AREA_INDIA_JAMMUKASHMIR"]), ["IN-JK"])
        self.assertEqual(rc("SE", ["AREA_OSTERGOTLANDSLAN", "AREA_GOTLANDSLAN"]), ["SE-E", "SE-I"])
        self.assertEqual(rc("ES", ["AREA_MADRID"]), ["ES-M"])
        self.assertEqual(rc("GH", ["ISO-GH-OT"]), ["GH-TV"])
        # the catalogue is built with the country names loaded: whole-country flag cards stay national
        flags = {cc: c["regions"] for cc, v in self.kb.gg.items() for c in v if c["id"] == cc.lower() + "-country-flag"}
        self.assertIn("CL", flags)
        self.assertEqual({cc: r for cc, r in flags.items() if r and cc != "GB"}, {})
        self.assertEqual(rc("KE", ["ISO-KE-15"]), ["KE-200"])
        self.assertEqual(rc("ID", ["ISO-ID-KU", "ISO-ID-JI"]), ["ID-KI", "ID-JI"])
        self.assertEqual(rc("LK", ["ISO-LK-3"]), ["LK-31", "LK-32", "LK-33"])
        self.assertEqual(rc("BR", ["AREA_BRAZIL_PARANA"]), ["BR-PR"])
        self.assertIn("GB-HLD", rc("GB", ["AREA_UK_SCOTLAND"]))

    def test_look_text(self):
        self.assertIn("вниз на дорогу", look_text("road", -14.0, 2.0))
        self.assertTrue(look_text("license-plates", -8.0, 3.5).startswith("Наблизьте камеру"))
        self.assertIsNone(look_text("misc", 0.0, 2.0))
        self.assertAlmostEqual(zoom_fov(1.0), 90.0)


if __name__ == "__main__":
    unittest.main()
