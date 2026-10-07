"""
Pure-math GeoGuessr locator: image -> country probabilities, score-optimal guess and hints.

Pipeline (no neural networks, no language models):
  1. the input (equirectangular panorama, in-game screenshots with their camera angles,
     or a single image) is put on a SphericalImage canvas;
  2. the five closed-form feature modules (engine/features) measure sun, road, landscape,
     camera vehicle and built structure;
  3. the calibrated Bayesian model (engine/model.py, trained by tools/train_model.py)
     turns the features into P(country | image) and a guess maximising the expected
     GeoGuessr score;
  4. engine/hints.py turns the measurements into observations and picks the matching
     clue cards from the GeoGuessr and Plonk It knowledge bases.
"""

import os
import time

import numpy as np
from PIL import Image

from . import features
from .hints import build_hints, clue_base
from .model import MODEL_DIR, GeoModel
from .panorama import SphericalImage


def load_spherical(image, heading=None, hfov=None, pitch=0.0):
    """Equirectangular (2:1) images become full panoramas, anything else a single view."""
    img = image if isinstance(image, Image.Image) else Image.open(image)
    w, h = img.size
    if abs(w / float(h) - 2.0) < 0.05:
        return SphericalImage.from_equirect(img, heading=heading)
    return SphericalImage.from_screenshot(img, yaw=heading or 0.0, pitch=pitch, hfov=hfov or 100.0,
                                          true_north=heading is not None)


class Locator:
    def __init__(self, model_dir=MODEL_DIR):
        if not os.path.exists(os.path.join(model_dir, "model.json")):
            raise FileNotFoundError("trained model not found in %s - run tools/train_model.py --save" % model_dir)
        self.model = GeoModel.load(model_dir)
        self.modules = {m: features.load(m) for m in self.model.groups}

    # ------------------------------------------------------------------ core
    def analyze(self, sph, top_k=5, n_hint_countries=3):
        t0 = time.time()
        X, evidence, flat = {}, {}, {}
        for name, mod in self.modules.items():
            try:
                r = mod.extract(sph)
                x = np.asarray(r["x"], np.float64)
                evidence[name] = r.get("evidence", {})
            except Exception as e:  # a failing module only removes its evidence
                x = np.full(len(mod.FEATURE_NAMES), np.nan)
                evidence[name] = {"error": repr(e)}
            X[name] = x[None, :]
            flat.update({"%s.%s" % (name, k): float(v) for k, v in zip(mod.FEATURE_NAMES, x)})
        t_feat = time.time() - t0
        m = self.model
        ev = m.evidence(X)
        lp = m.combine(ev)[0]
        post = np.exp(lp)
        order = np.argsort(-post)
        kb = clue_base()
        countries = [{"code": m.classes[i], "name": kb.country_name(m.classes[i]),
                      "probability": round(float(post[i]), 4)} for i in order[:top_k]]
        guess = m.locate(lp, ev["_d2"][0])
        # how much each evidence source moved the top country against the prior
        prior = m.prior()
        contrib = {}
        for c in order[:n_hint_countries]:
            row = {}
            for g in m.groups + ["knn"] + [k for k in ("glm", "sun") if k in ev]:
                w = m.weights.get(g, 0.0)
                e = ev[g][0]
                row[g] = round(float(w * (e[c] - np.dot(prior, e))), 2)
            contrib[m.classes[c]] = row
        hints = build_hints(flat, [(m.classes[i], post[i]) for i in order], n_hint_countries)
        return {
            "countries": countries,
            "guess": {"lat": round(guess["lat"], 5), "lng": round(guess["lng"], 5),
                      "expected_score": int(round(guess["expected_score"]))},
            "observations": hints["observations"],
            "hints": hints["countries"],
            "contributions": contrib,
            "measurements": evidence,
            "input": {"source": sph.source, "coverage": round(sph.coverage(), 3),
                      "true_heading_known": sph.heading is not None,
                      "car_axis_known": sph.car_heading is not None},
            "timing_ms": {"features": int(t_feat * 1000), "total": int((time.time() - t0) * 1000)},
        }

    # ------------------------------------------------------------ front-ends
    def analyze_image(self, image, heading=None, hfov=None, pitch=0.0):
        return self.analyze(load_spherical(image, heading, hfov, pitch))

    def analyze_views(self, views, width=2048):
        """views: [{image, yaw (true azimuth if true_north), pitch, hfov, mask?}, ...]"""
        return self.analyze(SphericalImage.from_views(views, width=width, heading=0.0))

    def analyze_pano(self, pano_id=None, lat=None, lng=None, radius=1000):
        """Official Street View panorama by id or nearest to lat/lng (for tests and calibration)."""
        from .streetview import download_panorama, get_metadata, search_pano
        meta = get_metadata(pano_id) if pano_id else search_pano(lat, lng, radius)
        if not meta:
            raise LookupError("no official Street View panorama found")
        sph = SphericalImage.from_equirect(download_panorama(meta), heading=meta.get("heading"))
        res = self.analyze(sph)
        res["panorama"] = {"pano_id": meta["pano_id"], "lat": meta["lat"], "lng": meta["lng"],
                           "country_code": meta.get("country_code"), "date": meta.get("date")}
        return res


def distance_report(res, lat, lng):
    """Distance / GeoGuessr points of a result against the true position."""
    from .geo import country_at, geoguessr_score, haversine_km
    d = float(haversine_km(lat, lng, res["guess"]["lat"], res["guess"]["lng"]))
    true_cc = country_at(lat, lng)
    codes = [c["code"] for c in res["countries"]]
    return {"distance_km": round(d, 1), "points": int(round(float(geoguessr_score(d)))), "true_country": true_cc,
            "rank_of_true_country": (codes.index(true_cc) + 1) if true_cc in codes else None}


_LOC = None


def get_locator():
    global _LOC
    if _LOC is None:
        _LOC = Locator()
    return _LOC


__all__ = ["Locator", "get_locator", "load_spherical", "distance_report"]
