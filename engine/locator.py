"""
Pure-math GeoGuessr locator: image -> country probabilities, score-optimal guess and hints.

Pipeline (no neural networks, no language models):
  1. the input (equirectangular panorama, in-game screenshots with their camera angles,
     or a single image) is put on a SphericalImage canvas;
  2. the five closed-form feature modules (engine/features) measure sun, road, landscape,
     camera vehicle and built structure;
  3. the calibrated Bayesian model (engine/model.py, trained by tools/train_model.py)
     turns the features into P(country | image) and a guess maximising the expected
     GeoGuessr score; with a map (id / slug / name, bounds, maxErrorDistance; missing fields
     from data/maps.json) the prior, the score scale and the bounds are the map's;
  3a. the calibrated parameters (evidence exponents, prior mixes, location / region kernels, region-model
     weights) are picked by input kind (input_kind): live captures (SphericalImage.source == "views") that
     cover at least VIEWS_MIN_COVERAGE of the sphere use the "views" set calibrated on live-grid renderings
     of the CALIB rounds, full panoramas and smaller captures the "pano" set (engine.model.GeoModel.for_input,
     engine.regions.RegionModel.params_for; "pano" when "views" is missing or stale);
  4. engine/hints.py turns the measurements into observations and picks the matching
     clue cards from the GeoGuessr and Plonk It knowledge bases;
  5. engine/clue_detect.py slides classical detectors of GeoGuessr's clue cards over the sphere:
     a calibrated country evidence source ("cards", only when >= 90% of the searched windows are
     visible), a soft update of the regions of the top countries (off while its calibrated
     exponent is 0) and, in the hints, the cards that lead / the directions worth pointing at
     (each off unless it beat the plain ranking on CALIB).
"""

import os
import sys
import time

import numpy as np
from PIL import Image

from . import features
from .clue_detect import clue_detectors
from .geo import bounds_dict, region_info, resolve_map
from .regions import load_region_model, location_weights, region_list, region_posterior
from .hints import build_hints, clue_base
from .model import MODEL_DIR, GeoModel
from .panorama import SphericalImage

# smallest sphere coverage that gets the "views" parameter set (the live grid covers ~0.95, a single
# 112.7 deg frame ~0.14; see input_kind)
VIEWS_MIN_COVERAGE = 0.5


def input_kind(sph):
    """Parameter set of an input: "views" for screenshot captures covering >= VIEWS_MIN_COVERAGE of the sphere
    (the set was calibrated on the live grid), "pano" for full panoramas and smaller captures."""
    return "views" if sph.source == "views" and sph.coverage() >= VIEWS_MIN_COVERAGE else "pano"


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
        self.cards = clue_detectors()
        if self.cards is None and self.model.weights.get("cards", 0.0) > 0:
            sys.stderr.write("model.json weights the card evidence but data/model/clue_detectors.npz is missing "
                             "- run tools/build_clue_detectors.py all; card evidence left out\n")

    # ------------------------------------------------------------------ core
    def analyze(self, sph, top_k=5, n_hint_countries=3, map_info=None, debug_dir=None):
        """map_info: None, an id / slug / name, or a dict with any of id, slug, name, bounds,
        maxErrorDistance.  debug_dir: also save the reconstructed sphere there (sphere.jpg)."""
        t0 = time.time()
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            Image.fromarray(sph.rgb).save(os.path.join(debug_dir, "sphere.jpg"), quality=90)
        mp = resolve_map(map_info)
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
        kind = input_kind(sph)
        m = self.model.for_input(kind)
        setup = m.map_setup(mp)
        where = {"score_scale_km": setup["scale_km"], "bounds": setup["bounds"]}
        t1 = time.time()
        det = cards_ll = None
        if self.cards is not None:
            try:
                det = self.cards.detect(sph)
                det["zt"] = self.cards.zt(det["z"])
                cards_ll = self.cards.country_evidence(det, m.classes)
                evidence["cards"] = {"coverage": round(det["coverage"], 3), "used": cards_ll is not None}
            except Exception as e:  # a failing detector bank only removes its evidence
                det, evidence["cards"] = None, {"error": repr(e)}
        t_cards = time.time() - t1
        ev = m.evidence(X, cards_ll=cards_ll)
        lp = m.combine(ev, setup["weights"], prior=setup["prior"])[0]
        post = np.exp(lp)
        order = np.argsort(-post)
        kb = clue_base()
        countries = [{"code": m.classes[i], "name": kb.country_name(m.classes[i]),
                      "probability": round(float(post[i]), 4)} for i in order[:top_k]]
        d2 = ev["_d2"][0]
        top = order[0]
        lp_top = np.full_like(lp, -1e9)
        lp_top[top] = 0.0
        rmodel = load_region_model()
        if rmodel is not None:
            # within-country region model (engine/regions.py): regions, and the guess placed on the
            # reference mass of the likely regions
            mix, per = region_posterior(X, post, classes=m.classes, by_country="both", bounds=setup["bounds"],
                                        params=rmodel.params_for(kind))
            regions = region_list(mix)
            guess = m.locate(lp, d2, w=location_weights(m, lp, d2, per, bounds=setup["bounds"]), **where)
            g_top = m.locate(lp_top, d2, w=location_weights(m, lp_top, d2, per, bounds=setup["bounds"]), **where)
        else:
            guess = m.locate(lp, d2, **where)
            # best point inside the most probable country (the global guess may hedge between countries)
            g_top = m.locate(lp_top, d2, **where)
            pr = m.region_posterior(lp, d2, bounds=setup["bounds"])
            if det is not None:  # regional cards found on the panorama sharpen the regions of the top countries
                pr = self._card_regions(pr, [m.classes[i] for i in order[:n_hint_countries]], det["zt"])
            regions = []
            for i in np.argsort(-pr)[:400]:
                if pr[i] <= 0:
                    break
                code, name, cc = region_info(int(i))
                regions.append({"code": code, "name": name, "country": cc, "probability": round(float(pr[i]), 5)})
        # how much each evidence source moved the top country against the prior
        prior = setup["prior"]
        contrib = {}
        for c in order[:n_hint_countries]:
            row = {}
            for g in m.groups + ["knn"] + [k for k in ("glm", "sun", "cards") if k in ev]:
                w = setup["weights"].get(g, 0.0)
                e = ev[g][0]
                row[g] = round(float(w * (e[c] - np.dot(prior, e))), 2)
            contrib[m.classes[c]] = row
        detected = None
        if det is not None:
            detected = {m.classes[i]: self.cards.detected_cards(m.classes[i], det) for i in order[:n_hint_countries]}
        hints = build_hints(flat, [(m.classes[i], post[i]) for i in order], n_hint_countries, regions,
                            detected=detected)
        return {
            "countries": countries,
            "regions": regions[:5],
            "top_country_point": {"lat": round(g_top["lat"], 5), "lng": round(g_top["lng"], 5),
                                  "expected_score_if_country_right": int(round(g_top["expected_score"]))},
            "guess": {"lat": round(guess["lat"], 5), "lng": round(guess["lng"], 5),
                      "expected_score": int(round(guess["expected_score"]))},
            "observations": hints["observations"],
            "hints": hints["countries"],
            "contributions": contrib,
            # the full country posterior and the prior it was computed with (engine/fusion.py fuses the
            # captures of one round by their evidence relative to that prior)
            "posterior": {m.classes[i]: float("%.4g" % post[i]) for i in range(len(post))},
            "prior": {m.classes[i]: float("%.4g" % prior[i]) for i in range(len(prior))},
            "map": None if mp is None else {
                "id": mp["id"], "slug": mp["slug"], "name": mp["name"], "bounds": bounds_dict(setup["bounds"]),
                "maxErrorDistance": mp["maxErrorDistance"], "score_scale_km": round(setup["scale_km"], 1),
                "world": mp["world"], "known": mp["known"], "prior": setup["kind"], "prior_counts": setup["counts"]},
            "measurements": evidence,
            "input": {"source": sph.source, "coverage": round(sph.coverage(), 3), "kind": kind, "params": m.param_set,
                      "region_params": kind if rmodel is not None and kind in rmodel.param_sets else "pano",
                      "true_heading_known": sph.heading is not None,
                      "car_axis_known": sph.car_heading is not None},
            "timing_ms": {"features": int(t_feat * 1000), "cards": int(t_cards * 1000),
                          "total": int((time.time() - t0) * 1000)},
        }

    def _card_regions(self, pr, countries, zt):
        """Region posterior (all countries) with the card update of engine.clue_detect applied inside
        each of the given countries (their total mass unchanged)."""
        nz = np.flatnonzero(pr > 0)
        by_cc = {}
        for i in nz:
            code, _, cc = region_info(int(i))
            by_cc.setdefault(cc, []).append((code, int(i)))
        pr = pr.copy()
        for cc in countries:
            items = by_cc.get(cc)
            if not items:
                continue
            tot = float(sum(pr[i] for _, i in items))
            up = self.cards.update_regions(cc, zt, {code: pr[i] / tot for code, i in items})
            for code, i in items:
                pr[i] = tot * up[code]
        return pr

    # ------------------------------------------------------------ front-ends
    def analyze_image(self, image, heading=None, hfov=None, pitch=0.0, map_info=None, debug_dir=None):
        return self.analyze(load_spherical(image, heading, hfov, pitch), map_info=map_info, debug_dir=debug_dir)

    def analyze_views(self, views, width=2048, map_info=None, debug_dir=None):
        """views: [{image, yaw (true azimuth if true_north), pitch, hfov, mask?}, ...]"""
        return self.analyze(SphericalImage.from_views(views, width=width, heading=0.0), map_info=map_info,
                            debug_dir=debug_dir)

    def analyze_pano(self, pano_id=None, lat=None, lng=None, radius=1000, map_info=None, debug_dir=None):
        """Official Street View panorama by id or nearest to lat/lng (for tests and calibration)."""
        from .streetview import download_panorama, get_metadata, search_pano
        meta = get_metadata(pano_id) if pano_id else search_pano(lat, lng, radius)
        if not meta:
            raise LookupError("no official Street View panorama found")
        sph = SphericalImage.from_equirect(download_panorama(meta), heading=meta.get("heading"))
        res = self.analyze(sph, map_info=map_info, debug_dir=debug_dir)
        res["panorama"] = {"pano_id": meta["pano_id"], "lat": meta["lat"], "lng": meta["lng"],
                           "country_code": meta.get("country_code"), "date": meta.get("date")}
        return res


def distance_report(res, lat, lng):
    """Distance / GeoGuessr points (with the result's map scale) of a result against the true position."""
    from .geo import country_at, geoguessr_points, haversine_km, region_at
    d = float(haversine_km(lat, lng, res["guess"]["lat"], res["guess"]["lng"]))
    true_cc = country_at(lat, lng)
    codes = [c["code"] for c in res["countries"]]
    reg = region_at(lat, lng)
    rcodes = [r["code"] for r in res.get("regions", [])]
    return {"true_region": reg[0] + " " + reg[1] if reg else None,
            "rank_of_true_region": (rcodes.index(reg[0]) + 1) if reg and reg[0] in rcodes else None,
            "distance_km": round(d, 1), "true_country": true_cc,
            "points": int(geoguessr_points(d, (res.get("map") or {}).get("maxErrorDistance"))),
            "rank_of_true_country": (codes.index(true_cc) + 1) if true_cc in codes else None}


_LOC = None


def get_locator():
    global _LOC
    if _LOC is None:
        _LOC = Locator()
    return _LOC


__all__ = ["Locator", "get_locator", "load_spherical", "distance_report"]
