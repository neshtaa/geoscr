"""
Plonk It Rule Matcher and Knowledge Base Query Engine
Allows filtering countries and querying detailed meta clues by visual observations.
"""

import json
import os
import re

class GeoKnowledgeBase:
    def __init__(self, kb_path="data/plonkit_kb.json", rules_path="data/country_rules.json", postmatch_kb_path=None):
        self.kb_path = kb_path
        self.rules_path = rules_path
        self.postmatch_kb_path = postmatch_kb_path
        if not self.postmatch_kb_path:
            if os.path.exists("data/geoguessr_master_clues.json"):
                self.postmatch_kb_path = "data/geoguessr_master_clues.json"
            else:
                self.postmatch_kb_path = "data/geoguessr_postmatch_clues.json"
        self.kb = {}
        self.rules = {}
        self.postmatch_kb = {}
        self.load()

    def load(self):
        if os.path.exists(self.kb_path):
            try:
                with open(self.kb_path, "r", encoding="utf-8") as f:
                    self.kb = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load {self.kb_path}: {e}")

        if os.path.exists(self.rules_path):
            try:
                with open(self.rules_path, "r", encoding="utf-8") as f:
                    self.rules = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load {self.rules_path}: {e}")

        if os.path.exists(self.postmatch_kb_path):
            try:
                with open(self.postmatch_kb_path, "r", encoding="utf-8") as f:
                    self.postmatch_kb = json.load(f)
            except Exception as e:
                pass

    def get_country(self, code_or_name):
        code_or_name_upper = code_or_name.upper().strip()
        data = None
        if code_or_name_upper in self.kb:
            data = dict(self.kb[code_or_name_upper])
        else:
            name_lower = code_or_name.lower().strip()
            for code, c_data in self.kb.items():
                if c_data["title"].lower() == name_lower or c_data["slug"].lower() == name_lower:
                    data = dict(c_data)
                    break
        
        if data:
            code = data.get("country_code", code_or_name_upper)
            if self.postmatch_kb and "clues_by_country" in self.postmatch_kb:
                data["official_geoguessr_clues"] = self.postmatch_kb["clues_by_country"].get(code, [])
            return data
        return None

    def search_clues(self, query, top_k=5):
        """Full-text search across all clues in both Plonk It and GeoGuessr match clues."""
        query_terms = [t.lower() for t in query.split() if len(t) > 2]
        if not query_terms:
            return []

        results = []
        for code, country in self.kb.items():
            for clue in country.get("all_clues", []):
                text_lower = clue["text"].lower()
                score = sum(1 for term in query_terms if term in text_lower)
                if score > 0:
                    results.append({
                        "source": "Plonk It Guide",
                        "country": country["title"],
                        "code": code,
                        "score": score,
                        "section": clue.get("section"),
                        "text": clue["text"],
                        "image_url": clue.get("image_url")
                    })

        # Also search official GeoGuessr clues (from master catalog and post-match)
        clues_dict = self.postmatch_kb.get("all_clues") or self.postmatch_kb.get("clues_by_id", {})
        for cid, clue in clues_dict.items():
            combined_text = f"{clue.get('title', '')} {clue.get('description', '')}".lower()
            score = sum(1.5 for term in query_terms if term in combined_text)
            if score > 0:
                cc = clue.get("country_code") or clue.get("countryCode", "GLOBAL")
                results.append({
                    "source": "GeoGuessr Official Clue Catalog",
                    "country": cc,
                    "code": cc,
                    "score": score,
                    "section": clue.get("category") or clue.get("type", "meta"),
                    "text": f"{clue.get('title')}: {clue.get('description')}",
                    "image_url": clue.get("image_url")
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def filter_candidates(self, driving_side=None, continent=None, features=None):
        """
        Filters candidates based on observed features:
        e.g. driving_side='left', continent='Asia', features=['snorkel', 'yellow plates']
        """
        candidates = []
        for code, country in self.kb.items():
            # Check continent
            if continent:
                country_cats = [c.lower() for c in country.get("continents", [])]
                if continent.lower() not in country_cats:
                    continue

            # Check driving side
            if driving_side:
                country_side = "left" if code in [c["code"] for c in self.rules.get("driving_side", {}).get("left", [])] else "right"
                if driving_side.lower() != country_side:
                    continue

            # Score matching features
            score = 1.0
            matched_clues = []
            if features:
                all_text = " ".join([c["text"] for c in country.get("all_clues", [])]).lower()
                for feat in features:
                    feat_lower = feat.lower()
                    if feat_lower in all_text:
                        score += 2.0
                        matched_clues.append(feat)

            candidates.append({
                "country": country["title"],
                "code": code,
                "continents": country.get("continents", []),
                "score": score,
                "matched_features": matched_clues
            })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates

    def get_differentiating_clues(self, country_code_1, country_code_2):
        """
        Returns key differences between two countries (e.g., Ukraine vs Russia,
        or Australia vs New Zealand).
        """
        c1 = self.get_country(country_code_1)
        c2 = self.get_country(country_code_2)
        if not c1 or not c2:
            return []

        differences = []
        # Check if country 1 has specific comparison tips against country 2
        c2_name = c2["title"].lower()
        for clue in c1.get("all_clues", []):
            if c2_name in clue["text"].lower() or f"not {c2_name}" in clue["text"].lower():
                differences.append({
                    "focus": c1["title"],
                    "tip": clue["text"]
                })
        return differences
