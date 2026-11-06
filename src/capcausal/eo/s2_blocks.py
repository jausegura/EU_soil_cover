"""Sentinel-2 parcel statistics from block-wise reads on the Planetary Computer."""
from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rasterio
from rasterio.windows import Window

log = logging.getLogger(__name__)

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
BLOCK = 512
GRID_N = 5490
BANDS = ["B04", "B8A", "B11", "B12"]
VALID_SCL = (4, 5)
NDVI_THRESHOLDS = (0.20, 0.25, 0.30, 0.35)
NDTI_THRESHOLDS = (0.06, 0.08, 0.10, 0.12, 0.14)
GDAL_ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                GDAL_HTTP_MAX_RETRY="6", GDAL_HTTP_RETRY_DELAY="3", GDAL_HTTP_TIMEOUT="120",
                VSI_CACHE="FALSE", GDAL_CACHEMAX="64")
os.environ.update(GDAL_ENV)


@dataclass(frozen=True)
class TileGrid:
    tile: str
    epsg: int
    x0: float
    y0: float

    @property
    def transform(self):
        from affine import Affine
        return Affine(20.0, 0.0, self.x0, 0.0, -20.0, self.y0)


def stac_client():
    return pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)


def tile_grids(tiles: list[str], cache: Path) -> dict[str, TileGrid]:
    """Grid origin and CRS of each MGRS tile, read from one STAC item (cached as JSON)."""
    data = json.loads(cache.read_text()) if cache.exists() else {}
    missing = [t for t in tiles if t not in data]
    if missing:
        cat = stac_client()
        for t in missing:
            it = next(cat.search(collections=["sentinel-2-l2a"], query={"s2:mgrs_tile": {"eq": t}},
                                 datetime="2023-06-01/2023-09-30", max_items=1).items())
            tr = it.assets["B8A"].extra_fields.get("proj:transform") or it.properties["proj:transform"]
            epsg = it.properties.get("proj:epsg") or int(str(it.properties["proj:code"]).split(":")[1])
            data[t] = {"epsg": int(epsg), "x0": tr[2], "y0": tr[5]}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data, indent=1))
    return {t: TileGrid(t, **data[t]) for t in tiles}


def assign_home_tile(points_xy: np.ndarray, src_epsg: int, grids: dict[str, TileGrid],
                     margin_m: float = 200.0) -> np.ndarray:
    """For each point choose the tile in which it lies farthest from the tile edge."""
    from pyproj import Transformer
    best = np.full(len(points_xy), "", dtype=object)
    best_d = np.full(len(points_xy), -np.inf)
    size = GRID_N * 20.0
    for t, g in grids.items():
        tr = Transformer.from_crs(src_epsg, g.epsg, always_xy=True)
        x, y = tr.transform(points_xy[:, 0], points_xy[:, 1])
        d = np.minimum.reduce([x - g.x0, g.x0 + size - x, g.y0 - y, y - (g.y0 - size)])
        upd = (d > margin_m) & (d > best_d)
        best[upd] = t
        best_d[upd] = d[upd]
    return best


def rasterize_parcels(gdf, grid: TileGrid, uid_col: str = "uid", inner_buffer_m: float = 10.0) -> np.ndarray:
    """int32 raster of parcel uids (0 = background) on the tile 20 m grid."""
    from rasterio.features import rasterize
    g = gdf.to_crs(grid.epsg)
    geom = g.geometry.buffer(-inner_buffer_m)
    keep = ~geom.is_empty & geom.notna()
    shapes = ((geom_, int(u)) for geom_, u in zip(geom[keep], g.loc[keep, uid_col]))
    return rasterize(shapes, out_shape=(GRID_N, GRID_N), transform=grid.transform, fill=0,
                     dtype="int32", all_touched=False)


def block_windows(pidx: np.ndarray) -> dict[tuple[int, int], Window]:
    """Windows of 512x512 blocks that contain at least one parcel pixel."""
    out = {}
    nb = int(np.ceil(GRID_N / BLOCK))
    for bi in range(nb):
        for bj in range(nb):
            sl = pidx[bi * BLOCK:(bi + 1) * BLOCK, bj * BLOCK:(bj + 1) * BLOCK]
            if (sl > 0).any():
                out[(bi, bj)] = Window(bj * BLOCK, bi * BLOCK, sl.shape[1], sl.shape[0])
    return out


def _read_blocks(href: str, windows: dict, overview_level: int | None = None) -> dict:
    kw = {"overview_level": overview_level} if overview_level is not None else {}
    out = {}
    for attempt in range(4):
        try:
            with rasterio.open(href, **kw) as src:
                for k, w in windows.items():
                    if k not in out:
                        out[k] = src.read(1, window=w)
            return out
        except Exception as e:
            log.warning("read failed (%s), attempt %d: %s", href.split("?")[0][-60:], attempt + 1, e, exc_info=attempt == 0)
            time.sleep(5 * 2 ** attempt)
    raise RuntimeError(f"giving up on {href.split('?')[0]}")


def _offset(item) -> float:
    pb = str(item.properties.get("s2:processing_baseline", "00.00"))
    try:
        return 1000.0 if float(pb) >= 4.0 else 0.0
    except ValueError:
        return 0.0


def _accumulate(reads: list[dict], clear: dict, wins: dict, pidx: np.ndarray, n_uid: int, off: float) -> np.ndarray:
    """Per-parcel sums for a set of blocks; rows: n, sum NDVI, sum NDTI, sum B11, threshold counts."""
    nt = len(NDVI_THRESHOLDS) + len(NDTI_THRESHOLDS)
    acc = np.zeros((4 + nt, n_uid + 1), dtype=np.float64)
    for k in reads[0]:
        m, w = clear[k], wins[k]
        p = pidx[w.row_off:w.row_off + w.height, w.col_off:w.col_off + w.width]
        raw = [r[k] for r in reads]
        ok = m & (raw[0] > 0) & (raw[1] > 0) & (raw[2] > 0) & (raw[3] > 0)
        if not ok.any():
            continue
        red, nir, s1, s2 = (np.clip(r[ok].astype("float32") - off, 1, None) for r in raw)
        u = p[ok]
        ndvi = (nir - red) / (nir + red)
        ndti = (s1 - s2) / (s1 + s2)
        acc[0] += np.bincount(u, minlength=n_uid + 1)
        acc[1] += np.bincount(u, weights=ndvi, minlength=n_uid + 1)
        acc[2] += np.bincount(u, weights=ndti, minlength=n_uid + 1)
        acc[3] += np.bincount(u, weights=s1 / 10000.0, minlength=n_uid + 1)
        for i, t in enumerate(NDVI_THRESHOLDS):
            acc[4 + i] += np.bincount(u, weights=(ndvi < t).astype("float64"), minlength=n_uid + 1)
        for j, t in enumerate(NDTI_THRESHOLDS):
            acc[4 + len(NDVI_THRESHOLDS) + j] += np.bincount(
                u, weights=((ndvi < 0.30) & (ndti < t)).astype("float64"), minlength=n_uid + 1)
    return acc


def process_item(item, pidx: np.ndarray, windows: dict, n_uid: int, min_clear_px: int = 200, chunk_blocks: int = 4,
                 pool: ThreadPoolExecutor | None = None, uids: np.ndarray | None = None) -> tuple[pd.DataFrame, dict]:
    """Per-parcel statistics for one scene."""
    item = planetary_computer.sign(item)
    scl = _read_blocks(item.assets["SCL"].href, windows)
    clear = {}
    for k, a in scl.items():
        p = pidx[windows[k].row_off:windows[k].row_off + a.shape[0], windows[k].col_off:windows[k].col_off + a.shape[1]]
        m = np.isin(a, VALID_SCL) & (p > 0)
        if m.sum() >= min_clear_px:
            clear[k] = m
    del scl
    meta = {"item": item.id, "date": item.datetime.date().isoformat(), "cloud": item.properties.get("eo:cloud_cover"),
            "baseline": item.properties.get("s2:processing_baseline"), "n_blocks": len(windows),
            "n_clear_blocks": len(clear)}
    if not clear:
        return pd.DataFrame(), meta
    wins = {k: windows[k] for k in clear}
    keys = list(wins)
    chunks = [keys[i:i + chunk_blocks] for i in range(0, len(keys), chunk_blocks)]
    off = _offset(item)
    hrefs = [(item.assets[b].href, 0 if b == "B04" else None) for b in BANDS]

    def one_chunk(ch):
        w = {k: wins[k] for k in ch}
        reads = [_read_blocks(h, w, ov) for h, ov in hrefs]
        return _accumulate(reads, clear, wins, pidx, n_uid, off)

    runner = pool.map if pool is not None else map
    acc = None
    for part in runner(one_chunk, chunks):
        acc = part if acc is None else acc + part
    n = acc[0]
    sel = np.nonzero(n > 0)[0]
    sel = sel[sel > 0]
    out_uid = sel if uids is None else uids[sel - 1]
    df = pd.DataFrame({"uid": out_uid.astype("int32"), "n_valid": n[sel].astype("int32"),
                       "ndvi": (acc[1, sel] / n[sel]).astype("float32"),
                       "ndti": (acc[2, sel] / n[sel]).astype("float32"),
                       "b11": (acc[3, sel] / n[sel]).astype("float32")})
    for i, t in enumerate(NDVI_THRESHOLDS):
        df[f"f_ndvi_lt{int(t * 100):02d}"] = (acc[4 + i, sel] / n[sel]).astype("float32")
    for j, t in enumerate(NDTI_THRESHOLDS):
        df[f"f_bare_ndti_lt{int(t * 100):02d}"] = (acc[4 + len(NDVI_THRESHOLDS) + j, sel] / n[sel]).astype("float32")
    df["date"] = pd.Timestamp(meta["date"])
    return df, meta


def season_items(tile: str, start: str, end: str, max_cloud: float = 80.0) -> list:
    cat = stac_client()
    items = list(cat.search(collections=["sentinel-2-l2a"], datetime=f"{start}/{end}",
                            query={"s2:mgrs_tile": {"eq": tile}, "eo:cloud_cover": {"lt": max_cloud}}).items())
    best = {}
    for it in items:
        key = (it.datetime.date(), it.properties.get("sat:relative_orbit"), it.properties.get("platform"))
        pb = str(it.properties.get("s2:processing_baseline", "0"))
        if key not in best or pb > str(best[key].properties.get("s2:processing_baseline", "0")):
            best[key] = it
    return sorted(best.values(), key=lambda i: i.datetime)
