#!/usr/bin/env python3
"""
Rebuild the sphere from a live capture and check the camera field of view.

  python3 tools/rebuild_views.py scratch/live/<game>_r<round>      -> sphere.jpg + FOV registration
  python3 tools/rebuild_views.py --probe a.jpg:yaw:pitch b.jpg:yaw:pitch   -> hfov of two frames (JSON)

A capture directory is written by play_live_visual.js: view_XX.jpg plus views.json
({"views": [{"file", "yaw", "pitch", "hfov", "vfov", "zoom"}, ...], ...}).

Registration: every view is back-projected on its own with hfov scaled by s and the mean
absolute grey difference over the overlaps of all view pairs is measured; s = 1.00 at the
minimum means the measured FOV is right.  The probe compares two frames taken from the same
point at known yaw/pitch and finds the hfov whose rotation homography maps one onto the other.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from engine.panorama import SphericalImage, _camera_basis, bilinear  # noqa: E402


def load_capture(path):
    with open(os.path.join(path, "views.json")) as f:
        meta = json.load(f)
    views = []
    for v in meta["views"]:
        img = Image.open(os.path.join(path, v["file"])).convert("RGB")
        views.append({"image": img, "yaw": float(v["yaw"]), "pitch": float(v["pitch"]), "hfov": float(v["hfov"])})
    return meta, views


def rebuild(views, width=2048):
    return SphericalImage.from_views(views, width=width)


def _grey(img):
    return np.asarray(img.convert("L"), np.float32)


def overlap_mad(views, scale, width=768):
    """Mean |grey difference| over the pairwise overlaps of individually projected views."""
    layers = []
    for v in views:
        sph = SphericalImage.from_views([dict(v, hfov=v["hfov"] * scale)], width=width)
        layers.append((_grey(Image.fromarray(sph.rgb)), sph.mask))
    total, count = 0.0, 0
    for i in range(len(layers)):
        for j in range(i + 1, len(layers)):
            m = layers[i][1] & layers[j][1]
            n = int(m.sum())
            if n < 50:
                continue
            total += float(np.abs(layers[i][0][m] - layers[j][0][m]).sum())
            count += n
    return total / count if count else float("nan"), count


def _refine(xs, ys):
    """Parabolic refinement of the minimum of a sampled curve."""
    k = int(np.nanargmin(ys))
    if 0 < k < len(xs) - 1:
        y0, y1, y2 = ys[k - 1], ys[k], ys[k + 1]
        den = y0 - 2 * y1 + y2
        if den > 0:
            return xs[k] + 0.5 * (y0 - y2) / den * (xs[k + 1] - xs[k])
    return xs[k]


def register_scale(views, lo=0.85, hi=1.15, step=0.01, width=768):
    scales = np.round(np.arange(lo, hi + step / 2, step), 4)
    rows = [(float(s),) + overlap_mad(views, float(s), width) for s in scales]
    best = _refine(np.array([r[0] for r in rows]), np.array([r[1] for r in rows]))
    return float(best), rows


def _pixel_rays(w, h, t):
    tv = t * h / w
    xs = ((np.arange(w) + 0.5) / w * 2 - 1) * t
    ys = (1 - (np.arange(h) + 0.5) / h * 2) * tv
    return np.meshgrid(xs, ys)


def pair_mad(grey_a, pose_a, grey_b, pose_b, hfov):
    """MAD between frame B and frame A warped into B by the pure rotation between them."""
    h, w = grey_b.shape
    t = math.tan(math.radians(hfov) / 2)
    tv = t * h / w
    xb, yb = _pixel_rays(w, h, t)
    fb, rb, ub = _camera_basis(*pose_b)
    fa, ra, ua = _camera_basis(*pose_a)
    d = fb[None, None, :] + xb[..., None] * rb[None, None, :] + yb[..., None] * ub[None, None, :]
    zc = d @ fa
    ok = zc > 1e-3
    zs = np.maximum(zc, 1e-3)
    px = ((d @ ra) / zs / t + 1) * w / 2 - 0.5
    py = (1 - (d @ ua) / zs / tv) * h / 2 - 0.5
    ok &= (px >= 0) & (px <= w - 1) & (py >= 0) & (py <= h - 1)
    if ok.mean() < 0.1:
        return float("nan"), float(ok.mean())
    warped = bilinear(grey_a[..., None], px[ok], py[ok])[..., 0]
    return float(np.abs(warped - grey_b[ok]).mean()), float(ok.mean())


def probe_fov(img_a, pose_a, img_b, pose_b, lo=10.0, hi=170.0, width=360):
    """hfov (deg) that best explains two frames taken at poses (yaw, pitch) from one point."""
    def prep(img):
        img = img if isinstance(img, Image.Image) else Image.open(img)
        h = max(8, int(round(img.size[1] * width / img.size[0])))
        return _grey(img.resize((width, h), Image.Resampling.BILINEAR))
    a, b = prep(img_a), prep(img_b)
    coarse = np.arange(lo, hi + 1e-6, 2.0)
    mads = np.array([pair_mad(a, pose_a, b, pose_b, f)[0] for f in coarse])
    k = int(np.nanargmin(mads))
    fine = np.arange(max(lo, coarse[k] - 2.5), min(hi, coarse[k] + 2.5) + 1e-6, 0.1)
    fm = np.array([pair_mad(a, pose_a, b, pose_b, f)[0] for f in fine])
    best = float(_refine(fine, fm))
    return {"hfov": round(best, 2), "mad": round(float(np.nanmin(fm)), 3),
            "mad_ratio": round(float(np.nanmin(fm) / np.nanmedian(mads)), 3)}


def _pose(spec):
    path, yaw, pitch = spec.rsplit(":", 2)
    return path, (float(yaw), float(pitch))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture", nargs="?", help="capture directory with views.json")
    ap.add_argument("--probe", nargs=2, metavar="IMG:YAW:PITCH", help="estimate hfov from two frames")
    ap.add_argument("--width", type=int, default=2048, help="sphere width")
    ap.add_argument("--no-register", action="store_true")
    ap.add_argument("--range", default="0.85,1.15,0.01", help="scale lo,hi,step")
    args = ap.parse_args()

    if args.probe:
        (pa, posea), (pb, poseb) = _pose(args.probe[0]), _pose(args.probe[1])
        print(json.dumps(probe_fov(pa, posea, pb, poseb)))
        return
    if not args.capture:
        ap.error("capture directory or --probe required")

    meta, views = load_capture(args.capture)
    sph = rebuild(views, args.width)
    out = np.where(sph.mask[..., None], sph.rgb, 0).astype(np.uint8)
    Image.fromarray(out).save(os.path.join(args.capture, "sphere.jpg"), quality=90)
    print("views %d, coverage %.1f%%, sphere.jpg %dx%d" % (len(views), 100 * sph.coverage(), sph.w, sph.h))
    for v, m in zip(views, meta["views"]):
        w, h = v["image"].size
        vf = 2 * math.degrees(math.atan(math.tan(math.radians(v["hfov"]) / 2) * h / w))
        print("  %-12s yaw %7.2f pitch %6.2f hfov %6.2f vfov %6.2f (aspect %6.2f) zoom %s  %s"
              % (m["file"], v["yaw"], v["pitch"], v["hfov"], m.get("vfov") or float("nan"), vf, m.get("zoom"),
                 m.get("fov_method", "")))
    if args.no_register or len(views) < 2:
        return
    lo, hi, step = (float(x) for x in args.range.split(","))
    best, rows = register_scale(views, lo, hi, step)
    print("registration (hfov x s): s  MAD  overlap px")
    for s, mad, n in rows:
        print("  %.3f  %7.3f  %d" % (s, mad, n))
    print("best s = %.3f" % best)
    with open(os.path.join(args.capture, "registration.json"), "w") as f:
        json.dump({"best_scale": best, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
