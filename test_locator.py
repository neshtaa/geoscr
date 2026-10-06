#!/usr/bin/env python3
"""
Test suite for GeoGuessr Locator and Plonk It Knowledge Base
"""

import unittest
import json
import os
from engine.rules_matcher import GeoKnowledgeBase
from engine.offline_engine import OfflineGeoLocator

class TestGeoLocator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = GeoKnowledgeBase()
        cls.locator = OfflineGeoLocator()

    def test_kb_loaded(self):
        self.assertGreater(len(self.kb.kb), 50, "Knowledge base should have at least 50 countries")

    def test_ukraine_query(self):
        ua = self.kb.get_country("UA")
        self.assertIsNotNone(ua, "Ukraine (UA) must be present in KB")
        self.assertEqual(ua["title"], "Ukraine")
        
        # Check that red car is mentioned in Ukraine tips
        all_text = " ".join([c["text"] for c in ua["all_clues"]]).lower()
        self.assertIn("red google car", all_text)

    def test_kenya_prediction(self):
        # Clues: snorkel, pickup truck
        res = self.locator.predict_from_features(driving_side="left", clues_list=["snorkel", "pickup truck"])
        top = res["top_prediction"]
        self.assertIsNotNone(top)
        self.assertEqual(top["country_code"], "KE")

    def test_ghana_tape_meta(self):
        res = self.locator.predict_from_features(clues_list=["black tape", "roof rack"])
        top = res["top_prediction"]
        self.assertIsNotNone(top)
        self.assertEqual(top["country_code"], "GH")

    def test_search_bollard(self):
        results = self.kb.search_clues("bollard", top_k=5)
        self.assertGreater(len(results), 0)

    def test_compare_countries(self):
        diffs = self.kb.get_differentiating_clues("UA", "RU")
        self.assertIsInstance(diffs, list)

    def test_real_image_brazil_soil(self):
        if os.path.exists("sample_brazil_soil.png"):
            res = self.locator.predict_image_heuristics("sample_brazil_soil.png")
            self.assertEqual(res["top_prediction"]["country_code"], "BR")

    def test_real_image_ukraine_car(self):
        if os.path.exists("sample_ukraine_red_car.png"):
            res = self.locator.predict_image_heuristics("sample_ukraine_red_car.png")
            self.assertEqual(res["top_prediction"]["country_code"], "UA")

    def test_postmatch_clues_loaded(self):
        self.assertGreater(len(self.locator.postmatch_kb.get("clues_by_country", {})), 30)
        clues = self.locator.postmatch_kb.get("all_clues") or self.locator.postmatch_kb.get("clues_by_id", {})
        self.assertGreater(len(clues), 200)

    def test_official_postmatch_clue_prediction(self):
        res = self.locator.predict_from_features(clues_list=["Orange Roof Tiles", "Portuguese"])
        self.assertEqual(res["top_prediction"]["country_code"], "BR")

        res_vn = self.locator.predict_from_features(clues_list=["Waystones with Coloured Tops", "Vietnamese"])
        self.assertEqual(res_vn["top_prediction"]["country_code"], "VN")

if __name__ == "__main__":
    unittest.main()
