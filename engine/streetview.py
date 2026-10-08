"""
Minimal Google Street View client (no API key).

Used to build the calibration dataset and to analyse arbitrary panoramas by
pano id / coordinates. Only official Google coverage (image type 2) is
requested, which is the same imagery GeoGuessr serves on its World map.
"""

import http.client
import io
import json
import math
import socket
import threading

from PIL import Image

RPC_HOST = "maps.googleapis.com"
RPC_PATH = "/$rpc/google.internal.maps.mapsjs.v1.MapsJsInternalService/"
TILE_HOST = "streetviewpixels-pa.googleapis.com"
TILE_PATH = "/v1/tile?cb_client=apiv3&panoid={pano}&output=tile&x={x}&y={y}&zoom={z}&nbt=1&fover=2"
HEADERS = {
    "content-type": "application/json+protobuf",
    "x-user-agent": "grpc-web-javascript/0.1",
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
}


_LOCAL = threading.local()


def _connect(host, timeout):
    """IPv4 keep-alive connection (one per thread and host).  Reusing connections avoids a
    TCP+TLS handshake per tile; IPv4 first because some networks (WSL2) resolve IPv6
    addresses without having an IPv6 route."""
    addrs = sorted(socket.getaddrinfo(host, 443, 0, socket.SOCK_STREAM), key=lambda a: a[0] != socket.AF_INET)
    conn = http.client.HTTPSConnection(host, timeout=timeout)
    last = None
    for fam, typ, proto, _, sa in addrs:
        sock = socket.socket(fam, typ, proto)
        sock.settimeout(min(timeout, 8))
        try:
            sock.connect(sa)
        except OSError as e:
            sock.close()
            last = e
            continue
        sock.settimeout(timeout)
        conn.sock = conn._context.wrap_socket(sock, server_hostname=host)
        return conn
    raise last or OSError("cannot connect to " + host)


def _request(host, method, path, body=None, headers=None, timeout=20, tries=3):
    pool = getattr(_LOCAL, "conns", None)
    if pool is None:
        pool = _LOCAL.conns = {}
    err = None
    for _ in range(tries):
        conn = pool.get(host)
        try:
            if conn is None:
                conn = pool[host] = _connect(host, timeout)
            conn.request(method, path, body=body, headers=headers or {})
            resp = conn.getresponse()
            data = resp.read()
            if resp.status >= 400:
                raise OSError("HTTP %d for %s" % (resp.status, host))
            return data
        except (OSError, http.client.HTTPException) as e:
            err = e
            if conn is not None:
                conn.close()
            pool.pop(host, None)
    raise err


def _rpc(method, payload, timeout=20):
    data = _request(RPC_HOST, "POST", RPC_PATH + method, json.dumps(payload).encode(), HEADERS, timeout)
    return json.loads(data.decode("utf-8"))


def _parse_pano(p):
    """Parse the pano block shared by SingleImageSearch and GetMetadata (None if unavailable)."""
    try:
        return _parse_pano_block(p)
    except (IndexError, TypeError, KeyError):
        return None  # removed / blurred panoramas come back as an empty shell


def _parse_pano_block(p):
    info = {"pano_id": p[1][1]}
    tiles = p[2]
    info["size"] = tiles[2]  # [height, width] at max zoom
    info["levels"] = [lvl[0] for lvl in tiles[3][0]]  # [[h, w], ...] per zoom level
    info["tile_size"] = tiles[3][1]
    loc = p[5][0][1]
    info["lat"], info["lng"] = loc[0][2], loc[0][3]
    orient = (loc[2] if len(loc) > 2 and loc[2] else []) + [None, None, None]
    info["heading"], info["tilt"], info["roll"] = orient[:3]
    info["country_code"] = loc[4] if len(loc) > 4 and isinstance(loc[4], str) else None
    try:
        info["date"] = p[6][7]  # [year, month]
    except (IndexError, TypeError):
        info["date"] = None
    try:
        info["camera"] = p[6][5][2]
    except (IndexError, TypeError):
        info["camera"] = None
    return info


def search_pano(lat, lng, radius=1000):
    """Nearest official Street View panorama to (lat, lng) within radius metres, or None."""
    payload = [["apiv3", None, None, None, "US", None, None, None, None, None, [[0]]],
               [[None, None, lat, lng], radius],
               [None, ["en", "US"], None, None, None, None, None, None, [2], None, [[[2, True, 2]]]],
               [[1, 2, 3, 4, 8, 6]]]
    data = _rpc("SingleImageSearch", payload)
    if not data or data[0] != [0] or len(data) < 2 or not data[1]:
        return None
    return _parse_pano(data[1])


def get_metadata(pano_id):
    payload = [["apiv3", None, None, None, "US", None, None, None, None, None, [[0]]],
               ["en", "US"], [[[2, pano_id]]], [[1, 2, 3, 4, 8, 6]]]
    data = _rpc("GetMetadata", payload)
    if not data or data[0] != [0] or not data[1]:
        return None
    return _parse_pano(data[1][0])


def _fetch_tile(pano_id, x, y, z, timeout=20):
    data = _request(TILE_HOST, "GET", TILE_PATH.format(pano=pano_id, x=x, y=y, z=z),
                    headers={"User-Agent": HEADERS["User-Agent"]}, timeout=timeout)
    return Image.open(io.BytesIO(data)).convert("RGB")


def download_panorama(meta, min_width=1600, out_width=2048):
    """Stitch an equirectangular panorama (out_width x out_width/2) from tiles.

    Column 0.5*W is the car heading (meta["heading"], degrees clockwise from north).
    """
    levels = meta["levels"]
    zoom = len(levels) - 1
    for z, (h, w) in enumerate(levels):
        if w >= min_width:
            zoom = z
            break
    h, w = levels[zoom]
    ts = meta.get("tile_size") or [512, 512]
    nx, ny = math.ceil(w / ts[1]), math.ceil(h / ts[0])
    canvas = Image.new("RGB", (nx * ts[1], ny * ts[0]))
    for ty in range(ny):
        for tx in range(nx):
            canvas.paste(_fetch_tile(meta["pano_id"], tx, ty, zoom), (tx * ts[1], ty * ts[0]))
    canvas = canvas.crop((0, 0, w, h))
    return canvas.resize((out_width, out_width // 2), Image.Resampling.BICUBIC)
