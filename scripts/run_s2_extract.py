#!/usr/bin/env python
"""Parcel statistics from the Planetary Computer."""
from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from capcausal.eo import s2_blocks as sb
from capcausal.utils.paths import DATA_DIR

S2_DIR = DATA_DIR / "interim" / "s2"
DK_DIR = DATA_DIR / "interim" / "dk"
DK_TILES = ["32UMF", "32UMG", "32UNF", "32UNG", "32UPF", "32UPG", "32VMH", "32VMJ", "32VMK", "32VNH",
            "32VNJ", "32VNK", "32VPH", "32VPJ", "32VPK", "33UUA", "33UUB", "33VUC", "33VUD", "33VUE"]
log = logging.getLogger("s2")


def home_tiles(season: int, grids: dict) -> pd.Series:
    """Home MGRS tile per parcel uid (cached per season)."""
    out = S2_DIR / "home" / f"home_{season}.parquet"
    if out.exists():
        return pd.read_parquet(out).set_index("uid")["tile"]
    g = gpd.read_parquet(DK_DIR / f"parcels_{season}.parquet", columns=["uid", "geometry"])
    pts = g.geometry.representative_point()
    xy = np.c_[pts.x.values, pts.y.values]
    tile = sb.assign_home_tile(xy, 25832, grids)
    s = pd.Series(tile, index=g.uid.values, name="tile")
    out.parent.mkdir(parents=True, exist_ok=True)
    s.rename_axis("uid").reset_index().to_parquet(out)
    return s


def parcel_raster(tile: str, season: int, grid, home: pd.Series, groups=None, root=None) -> np.ndarray:
    root = root or S2_DIR
    f = root / tile / f"pidx_{season}.npz"
    if f.exists():
        return np.load(f)["pidx"]
    uids = home.index[home.values == tile]
    g = gpd.read_parquet(DK_DIR / f"parcels_{season}.parquet", columns=["uid", "crop_group", "geometry"])
    g = g[g.uid.isin(uids)]
    if groups:
        g = g[g.crop_group.isin(groups)]
    pidx = sb.rasterize_parcels(g, grid)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, pidx=pidx)
    npx = np.bincount(pidx.ravel(), minlength=int(g.uid.max()) + 1 if len(g) else 1)
    sel = np.nonzero(npx)[0]
    sel = sel[sel > 0]
    pd.DataFrame({"uid": sel.astype("int32"), "n_pix": npx[sel].astype("int32")}).to_parquet(
        root / tile / f"npix_{season}.parquet")
    return pidx


def run(tiles, seasons, start, end, max_cloud, scene_workers, band_workers, groups=None, root=None):
    root = root or S2_DIR
    grids = sb.tile_grids(DK_TILES, S2_DIR / "tile_grids.json")
    band_pool = ThreadPoolExecutor(band_workers)
    for season in seasons:
        home = home_tiles(season, grids)
        for tile in tiles:
            t0 = time.time()
            pidx = parcel_raster(tile, season, grids[tile], home, groups, root)
            uids, inv = np.unique(pidx, return_inverse=True)
            if uids[0] == 0:
                pidx = inv.reshape(pidx.shape).astype("int32")
                uids = uids[1:]
            else:
                pidx = (inv.reshape(pidx.shape) + 1).astype("int32")
            wins = sb.block_windows(pidx)
            n_uid = len(uids)
            if not wins:
                log.info("tile %s season %d: no parcels, skipped", tile, season)
                continue
            ys = season + (1 if end < start else 0)
            items = sb.season_items(tile, f"{season}-{start}", f"{ys}-{end}", max_cloud)
            odir = root / tile / str(season)
            odir.mkdir(parents=True, exist_ok=True)
            todo = [it for it in items if not (odir / f"{it.id}.json").exists()]
            log.info("tile %s season %d: %d blocks, %d scenes (%d to do)", tile, season, len(wins), len(items), len(todo))

            def one(it):
                df, meta = sb.process_item(it, pidx, wins, n_uid, pool=band_pool, uids=uids)
                if len(df):
                    df.to_parquet(odir / f"{it.id}.parquet")
                (odir / f"{it.id}.json").write_text(json.dumps(meta))
                return meta

            with ThreadPoolExecutor(scene_workers) as ex:
                futs = [ex.submit(one, it) for it in todo]
                for k, fu in enumerate(as_completed(futs), 1):
                    try:
                        m = fu.result()
                        log.info("  [%d/%d] %s clear blocks %d/%d", k, len(todo), m["date"], m["n_clear_blocks"], m["n_blocks"])
                    except Exception as e:
                        log.error("  scene failed: %s", e)
            log.info("tile %s season %d done in %.1f min", tile, season, (time.time() - t0) / 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", nargs="+", default=DK_TILES)
    ap.add_argument("--seasons", nargs="+", type=int, required=True)
    ap.add_argument("--start", default="08-01")
    ap.add_argument("--end", default="10-31")
    ap.add_argument("--max-cloud", type=float, default=80.0)
    ap.add_argument("--scene-workers", type=int, default=3)
    ap.add_argument("--band-workers", type=int, default=24)
    ap.add_argument("--groups", nargs="*", help="restrict to these crop groups (e.g. maize_grain)")
    ap.add_argument("--outdir", help="output root (default data/interim/s2)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(a.tiles, a.seasons, a.start, a.end, a.max_cloud, a.scene_workers, a.band_workers, a.groups,
        Path(a.outdir) if a.outdir else None)
