#!/usr/bin/env python3
"""
Calibrate the sun-disk detector of engine/features/solar.py (logistic regression).

Every saturated-blob candidate of a panorama with known heading, latitude and capture
month is labelled "real sun" when its direction is within 3 deg of a solar position
that is possible at that latitude in that month (declination range of the month x all
hour angles).  A quadratic logistic model on the candidate statistics (halo profile,
size, compactness, sky context, camera shadow) is fitted with ridge-regularised IRLS
and written to data/model/sun_detector.json.

  python3 tools/calibrate_sun.py [--n 3000] [--workers 4]
"""
import argparse
import json
import math
import os
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from calibrate import DATASET, load_records  # noqa: E402

OUT = os.path.join(ROOT, "data", "model", "sun_detector.json")
FEATURES = ["core_Y", "core_sat", "h1_Y", "h2_Y", "far_Y", "iso", "h1_sky", "h2_sky", "far_sky", "h1_sat", "h2_sat",
            "far_sat", "req", "rmax", "compact", "area_share", "shadow", "sky_sat_frac", "sky_clear", "el",
            "h2_bstar", "far_bstar"]
LOG_KEYS = {"req": 0.3, "rmax": 0.3, "area_share": 1e-3}


def sun_error_deg(lat, month, az, el):
    """Smallest angle between the observed sun and any sun position possible at lat in that month."""
    days = np.arange((month - 1) * 30.4, month * 30.4, 2.0)
    dec = np.radians(23.44 * np.sin(2 * np.pi * (days + 10 - 81) / 365))
    d, h = np.meshgrid(dec, np.radians(np.arange(-180, 180, 0.5)))
    p = math.radians(lat)
    se = np.sin(p) * np.sin(d) + np.cos(p) * np.cos(d) * np.cos(h)
    e = np.arcsin(np.clip(se, -1, 1))
    A = np.arccos(np.clip((np.sin(d) - se * np.sin(p)) / np.maximum(np.cos(e) * np.cos(p), 1e-9), -1, 1))
    A = np.where(np.sin(h) > 0, 2 * np.pi - A, A)
    v = np.stack([np.sin(A) * np.cos(e), np.cos(A) * np.cos(e), np.sin(e)], -1)
    a, l = math.radians(az), math.radians(el)
    o = np.array([math.sin(a) * math.cos(l), math.cos(a) * math.cos(l), math.sin(l)])
    return float(np.degrees(np.arccos(np.clip((v @ o).max(), -1, 1))))


def _work(r):
    from engine.features import solar
    from engine.panorama import SphericalImage
    sph = SphericalImage.from_equirect(os.path.join(DATASET, "panos", r["pano_id"] + ".jpg"), heading=r["heading"])
    out = []
    for c in solar.extract(sph)["_cands"][:4]:
        row = {k: (float(c[k]) if c.get(k) is not None and np.isscalar(c[k]) else np.nan) for k in FEATURES}
        row["err"] = sun_error_deg(r["lat"], r["date"][1], (r["heading"] + c["phi"]) % 360.0, c["el"])
        row["pano"] = r["pano_id"]
        out.append(row)
    return out


def design(X, st=None):
    X = X.copy()
    for k, off in LOG_KEYS.items():
        X[:, FEATURES.index(k)] = np.log(X[:, FEATURES.index(k)] + off)
    if st is None:
        st = (np.nanmedian(X, 0), np.nanstd(X, 0) + 1e-6)
    Z = np.clip(np.nan_to_num((X - st[0]) / st[1]), -5, 5)
    return np.c_[np.ones(len(Z)), Z, Z ** 2], st


def fit_logistic(Z, y, lam=1.0):
    w = np.zeros(Z.shape[1])
    for _ in range(30):
        p = 1 / (1 + np.exp(-Z @ w))
        H = Z.T @ (Z * (p * (1 - p) + 1e-6)[:, None]) + lam * np.eye(len(w))
        w += np.linalg.solve(H, Z.T @ (y - p) - lam * w)
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    recs = [r for r in load_records() if r.get("heading") is not None and r.get("date")][: args.n]
    # candidates must be scored with the heuristic, not a previous fit: hide the old model
    if os.path.exists(OUT):
        os.rename(OUT, OUT + ".old")
    try:
        with Pool(args.workers) as pool:
            rows = [c for cs in pool.map(_work, recs, chunksize=8) for c in cs]
    finally:
        if os.path.exists(OUT + ".old"):
            os.rename(OUT + ".old", OUT)
    X = np.array([[c[k] for k in FEATURES] for c in rows], float)
    y = np.array([c["err"] < 3.0 for c in rows], float)
    fold = np.array([hash(c["pano"]) % 5 for c in rows])
    pred = np.zeros(len(y))
    for k in range(5):
        Z, st = design(X[fold != k])
        w = fit_logistic(Z, y[fold != k])
        pred[fold == k] = 1 / (1 + np.exp(-design(X[fold == k], st)[0] @ w))
    for t in (0.5, 0.7):
        s = pred > t
        print(f"P>{t}: {s.sum()} detections, precision {y[s].mean():.3f}, recall {y[s].sum() / y.sum():.3f}")
    Z, st = design(X)
    w = fit_logistic(Z, y)
    json.dump({"features": FEATURES, "median": [float(v) for v in st[0]], "std": [float(v) for v in st[1]],
               "w": [float(v) for v in w]}, open(OUT, "w"))
    print(f"{len(rows)} candidates from {len(recs)} panoramas -> {OUT}")


if __name__ == "__main__":
    main()
