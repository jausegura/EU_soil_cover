"""Sentinel-2 parcel statistics from openEO on the Copernicus Data Space."""
from __future__ import annotations

import os

import openeo

NDTI_EDGES = (0.06, 0.08, 0.10, 0.12, 0.14)
NDVI_LOW = 0.30


def connect() -> openeo.Connection:
    from dotenv import load_dotenv

    from capcausal.utils.paths import ROOT
    load_dotenv(ROOT / ".env")
    con = openeo.connect("openeo.dataspace.copernicus.eu")
    con.authenticate_oidc_resource_owner_password_credentials(
        username=os.environ["CDSE_USERNAME"], password=os.environ["CDSE_PASSWORD"], client_id="cdse-public")
    return con


def class_cube(con, epsg: int, bbox: tuple[float, float, float, float], start: str, end: str,
               max_cloud: float = 80.0):
    """bbox = (xmin, ymin, xmax, ymax) in the tile CRS, aligned to the 20 m grid."""
    xmin, ymin, xmax, ymax = bbox
    cube = con.load_collection(
        "SENTINEL2_L2A",
        spatial_extent={"west": xmin, "south": ymin, "east": xmax, "north": ymax, "crs": epsg},
        temporal_extent=[start, end], bands=["B04", "B8A", "B11", "B12", "SCL"], max_cloud_cover=max_cloud)
    cube = cube.resample_spatial(resolution=20, projection=epsg, method="near")
    red, nir, s1, s2, scl = (cube.band(b) for b in ("B04", "B8A", "B11", "B12", "SCL"))
    ndvi = (nir - red) / (nir + red)
    ndti = (s1 - s2) / (s1 + s2)
    valid = ((scl == 4) | (scl == 5)) & (red > 0) & (nir > 0) & (s1 > 0) & (s2 > 0)
    low = ndvi < NDVI_LOW
    nbin = None
    for e in NDTI_EDGES:
        term = (ndti >= e) * 1
        nbin = term if nbin is None else nbin + term
    code = valid * (1 + low * (1 + nbin))
    return code.linear_scale_range(0, 255, 0, 255)


def parcel_stats(code, pidx, windows: dict, uids, min_clear_px: int = 200):
    """Per-parcel statistics from one class raster, with the block rule of s2_blocks.process_item."""
    import numpy as np
    import pandas as pd
    n_uid = len(uids)
    acc = np.zeros((2 + len(NDTI_EDGES), n_uid + 1))
    n_clear = 0
    for w in windows.values():
        c = code[w.row_off:w.row_off + w.height, w.col_off:w.col_off + w.width]
        p = pidx[w.row_off:w.row_off + w.height, w.col_off:w.col_off + w.width]
        ok = (c >= 1) & (c <= 7) & (p > 0)
        if ok.sum() < min_clear_px:
            continue
        n_clear += 1
        u, cc = p[ok], c[ok]
        acc[0] += np.bincount(u, minlength=n_uid + 1)
        acc[1] += np.bincount(u, weights=(cc >= 2).astype(float), minlength=n_uid + 1)
        for j in range(len(NDTI_EDGES)):
            acc[2 + j] += np.bincount(u, weights=((cc >= 2) & (cc <= 2 + j)).astype(float), minlength=n_uid + 1)
    n = acc[0]
    sel = np.nonzero(n > 0)[0]
    sel = sel[sel > 0]
    df = pd.DataFrame({"uid": np.asarray(uids)[sel - 1].astype("int32"), "n_valid": n[sel].astype("int32"),
                       "f_ndvi_lt30": (acc[1, sel] / n[sel]).astype("float32")})
    for j, e in enumerate(NDTI_EDGES):
        df[f"f_bare_ndti_lt{int(round(e * 100)):02d}"] = (acc[2 + j, sel] / n[sel]).astype("float32")
    return df, n_clear
