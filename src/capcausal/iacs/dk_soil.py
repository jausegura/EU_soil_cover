"""JB soil classes and GAEC 6 soil zones."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyogrio
import rasterio
from affine import Affine
from rasterio.features import rasterize

from capcausal.utils.paths import DATA_DIR

SHP = DATA_DIR / "raw" / "dk_lbst" / "Jordbundskort_2019" / "Jordbundskort_2019.shp"
RASTER = DATA_DIR / "interim" / "dk" / "jb_2019_25m.tif"
RES = 25.0
BOUNDS = (440000.0, 6040000.0, 900000.0, 6410000.0)


def jb_raster(path: Path = RASTER) -> Path:
    if path.exists():
        return path
    gdf = pyogrio.read_dataframe(SHP, columns=["JB_nr"])
    gdf = gdf[gdf.geometry.notna()]
    x0, y0, x1, y1 = BOUNDS
    w, h = int((x1 - x0) / RES), int((y1 - y0) / RES)
    tr = Affine(RES, 0, x0, 0, -RES, y1)
    arr = rasterize(((g, int(v)) for g, v in zip(gdf.geometry, gdf.JB_nr) if v is not None),
                    out_shape=(h, w), transform=tr, fill=0, dtype="uint8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", width=w, height=h, count=1, dtype="uint8", crs="EPSG:25832",
                       transform=tr, compress="deflate", tiled=True) as dst:
        dst.write(arr, 1)
    return path


def jb_at(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """JB class at coordinates (EPSG:25832); 0 = outside the map."""
    with rasterio.open(jb_raster()) as src:
        arr = src.read(1)
        r, c = rasterio.transform.rowcol(src.transform, x, y)
    r, c = np.asarray(r), np.asarray(c)
    ok = (r >= 0) & (r < arr.shape[0]) & (c >= 0) & (c < arr.shape[1])
    out = np.zeros(len(r), dtype="uint8")
    out[ok] = arr[r[ok], c[ok]]
    return out


def glm6_zone(jb: np.ndarray) -> np.ndarray:
    """'sand_1feb' (JB 1-4), 'loam_20oct' (JB 5, 6, 10, 11), 'other_1oct' (rest), 'unknown' (0)."""
    jb = np.asarray(jb)
    z = np.full(jb.shape, "other_1oct", dtype=object)
    z[np.isin(jb, [1, 2, 3, 4])] = "sand_1feb"
    z[np.isin(jb, [5, 6, 10, 11])] = "loam_20oct"
    z[jb == 0] = "unknown"
    return z
