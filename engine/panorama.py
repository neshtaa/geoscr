"""
Spherical image representation shared by every feature extractor.

A ``SphericalImage`` is an equirectangular canvas (W = 2H) plus a validity mask
and the true azimuth of its centre column.  Full Street View panoramas fill the
whole sphere; screenshots taken in GeoGuessr are back-projected onto the same
canvas (pinhole camera model), so all features are computed in one angular
coordinate system no matter where the pixels came from.

Conventions
  column x  ->  relative longitude  phi = (x + 0.5) / W * 360 - 180   (deg)
  row y     ->  elevation           el  = 90 - (y + 0.5) / H * 180    (deg)
  true azimuth (clockwise from north) = heading + phi, when heading is known
  world frame for direction vectors: x = east, y = north, z = up
"""

import math

import numpy as np
from PIL import Image


def direction(az_deg, el_deg):
    """Unit direction vectors (..., 3) for azimuth/elevation arrays in degrees."""
    a, e = np.broadcast_arrays(np.radians(az_deg), np.radians(el_deg))
    ce = np.cos(e)
    return np.stack([np.sin(a) * ce, np.cos(a) * ce, np.sin(e)], axis=-1)


def bilinear(img, xs, ys, wrap_x=False):
    """Sample img (H, W, C) at float pixel coordinates (centre of pixel i is i)."""
    h, w = img.shape[:2]
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    fx = (xs - x0)[..., None]
    fy = (ys - y0)[..., None]
    if wrap_x:
        x0m, x1m = x0 % w, (x0 + 1) % w
    else:
        x0m, x1m = np.clip(x0, 0, w - 1), np.clip(x0 + 1, 0, w - 1)
    y0m, y1m = np.clip(y0, 0, h - 1), np.clip(y0 + 1, 0, h - 1)
    a = img[y0m, x0m].astype(np.float32)
    b = img[y0m, x1m].astype(np.float32)
    c = img[y1m, x0m].astype(np.float32)
    d = img[y1m, x1m].astype(np.float32)
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def _camera_basis(yaw_deg, pitch_deg):
    f = direction(np.array(yaw_deg), np.array(pitch_deg))
    a = math.radians(yaw_deg)
    r = np.array([math.cos(a), -math.sin(a), 0.0])
    u = np.cross(r, f)
    return f, r, u


class SphericalImage:
    def __init__(self, rgb, heading=None, mask=None, car_heading=None, source="pano"):
        self.rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        self.h, self.w = self.rgb.shape[:2]
        self.heading = heading            # true azimuth of the centre column (None = unknown)
        self.car_heading = car_heading    # relative longitude of the car's driving axis (0 for panos)
        self.mask = np.ones((self.h, self.w), bool) if mask is None else mask.astype(bool)
        self.source = source
        self._cache = {}

    # ------------------------------------------------------------------ ctors
    @classmethod
    def from_equirect(cls, image, heading=None, width=2048):
        """Full panorama whose centre column looks along the car (Street View layout)."""
        img = image if isinstance(image, Image.Image) else Image.open(image)
        img = img.convert("RGB")
        if img.size != (width, width // 2):
            img = img.resize((width, width // 2), Image.Resampling.BILINEAR)
        return cls(np.asarray(img), heading=heading, car_heading=0.0, source="pano")

    @classmethod
    def from_views(cls, views, width=2048, heading=0.0):
        """Back-project perspective screenshots onto the sphere.

        views: list of dicts {image, yaw (true azimuth deg, or relative if unknown),
               pitch (deg), hfov (deg)}.  ``heading`` is the true azimuth that the
               canvas centre should face (keep 0.0 so azimuth == longitude).
        """
        h = width // 2
        rgb = np.zeros((h, width, 3), np.float32)
        best = np.full((h, width), -2.0, np.float32)
        phi = (np.arange(width) + 0.5) / width * 360.0 - 180.0
        el = 90.0 - (np.arange(h) + 0.5) / h * 180.0
        az = (heading if heading is not None else 0.0) + phi
        d = direction(az[None, :], el[:, None])
        for v in views:
            img = v["image"]
            img = img if isinstance(img, Image.Image) else Image.open(img)
            arr = np.asarray(img.convert("RGB"))
            ih, iw = arr.shape[:2]
            f, r, u = _camera_basis(v["yaw"], v.get("pitch", 0.0))
            zc = d @ f
            ok = zc > 1e-3
            t = math.tan(math.radians(v["hfov"]) / 2)
            tv = t * ih / iw
            xc = np.where(ok, (d @ r) / np.maximum(zc, 1e-3), 9.0)
            yc = np.where(ok, (d @ u) / np.maximum(zc, 1e-3), 9.0)
            px = (xc / t + 1) * iw / 2 - 0.5
            py = (1 - yc / tv) * ih / 2 - 0.5
            inside = ok & (px >= 0) & (px <= iw - 1) & (py >= 0) & (py <= ih - 1)
            if "mask" in v and v["mask"] is not None:  # exclude UI overlays
                m = np.asarray(v["mask"], bool)
                inside &= m[np.clip(py.round().astype(int), 0, ih - 1), np.clip(px.round().astype(int), 0, iw - 1)]
            better = inside & (zc > best)  # prefer the view whose centre is closest
            if not better.any():
                continue
            ys, xs = np.nonzero(better)
            rgb[ys, xs] = bilinear(arr, px[ys, xs], py[ys, xs])
            best[ys, xs] = zc[ys, xs]
        mask = best > -2.0
        known = all(v.get("true_north", True) for v in views)
        return cls(np.clip(rgb, 0, 255).astype(np.uint8), heading=heading if known else None,
                   mask=mask, car_heading=None, source="views")

    @classmethod
    def from_screenshot(cls, image, yaw=0.0, pitch=0.0, hfov=100.0, true_north=False, width=1024):
        sph = cls.from_views([{"image": image, "yaw": yaw, "pitch": pitch, "hfov": hfov,
                               "true_north": true_north}], width=width)
        if not true_north:
            sph.heading = None
        return sph

    # -------------------------------------------------------------- geometry
    def longitudes(self):
        return (np.arange(self.w) + 0.5) / self.w * 360.0 - 180.0

    def elevations(self):
        return 90.0 - (np.arange(self.h) + 0.5) / self.h * 180.0

    def row_of(self, el_deg):
        return int(np.clip((90.0 - el_deg) / 180.0 * self.h, 0, self.h - 1))

    def col_of(self, phi_deg):
        return int(((phi_deg + 180.0) % 360.0) / 360.0 * self.w) % self.w

    def true_azimuth(self, phi_deg):
        if self.heading is None:
            return None
        return (self.heading + phi_deg) % 360.0

    def band(self, el_top, el_bottom):
        """Row slice covering elevations el_top (higher) .. el_bottom."""
        return slice(self.row_of(el_top), self.row_of(el_bottom) + 1)

    def coverage(self):
        return float(self.mask.mean())

    def resized(self, width):
        if width == self.w:
            return self
        key = ("resized", width)
        if key not in self._cache:
            img = Image.fromarray(self.rgb).resize((width, width // 2), Image.Resampling.BILINEAR)
            m = Image.fromarray(self.mask.astype(np.uint8) * 255).resize((width, width // 2), Image.Resampling.NEAREST)
            self._cache[key] = SphericalImage(np.asarray(img), self.heading, np.asarray(m) > 127,
                                              self.car_heading, self.source)
        return self._cache[key]

    def float_rgb(self):
        if "float" not in self._cache:
            self._cache["float"] = self.rgb.astype(np.float32)
        return self._cache["float"]

    # ------------------------------------------------------------- rendering
    def render_view(self, yaw_rel, pitch=0.0, hfov=100.0, size=(1280, 720)):
        """Perspective view looking at relative longitude yaw_rel (deg) - for tests/simulation."""
        iw, ih = size
        f, r, u = _camera_basis(yaw_rel, pitch)
        t = math.tan(math.radians(hfov) / 2)
        tv = t * ih / iw
        xs = ((np.arange(iw) + 0.5) / iw * 2 - 1) * t
        ys = (1 - (np.arange(ih) + 0.5) / ih * 2) * tv
        d = f[None, None, :] + xs[None, :, None] * r[None, None, :] + ys[:, None, None] * u[None, None, :]
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        az = np.degrees(np.arctan2(d[..., 0], d[..., 1]))
        el = np.degrees(np.arcsin(np.clip(d[..., 2], -1, 1)))
        px = (az + 180.0) / 360.0 * self.w - 0.5
        py = (90.0 - el) / 180.0 * self.h - 0.5
        out = bilinear(self.rgb, px, py, wrap_x=True)
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
