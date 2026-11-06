#!/usr/bin/env python
"""Parcel statistics from openEO batch jobs on the Copernicus Data Space."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from capcausal.eo import cdse_openeo as co
from capcausal.eo import s2_blocks as sb
from capcausal.utils.paths import DATA_DIR

PC = DATA_DIR / "interim" / "s2"
OUT = DATA_DIR / "interim" / "s2_cdse"
RAW = DATA_DIR / "interim" / "cdse_raw"
log = logging.getLogger("cdse")


def tile_inputs(tile: str, season: int):
    pidx = np.load(PC / tile / f"pidx_{season}.npz")["pidx"]
    uids, inv = np.unique(pidx, return_inverse=True)
    loc = inv.reshape(pidx.shape).astype("int32")
    uids = uids[1:]
    wins = sb.block_windows(loc)
    return loc, uids, wins


def bbox_of(wins: dict, grid: dict) -> tuple:
    r0 = min(w.row_off for w in wins.values())
    r1 = max(w.row_off + w.height for w in wins.values())
    c0 = min(w.col_off for w in wins.values())
    c1 = max(w.col_off + w.width for w in wins.values())
    x0, y0 = grid["x0"], grid["y0"]
    return (x0 + c0 * 20, y0 - r1 * 20, x0 + c1 * 20, y0 - r0 * 20)


def run_one(con, tile: str, season: int, grid: dict, start: str, end: str) -> dict:
    odir = OUT / tile / str(season)
    done = odir / "_DONE.json"
    if done.exists():
        return json.loads(done.read_text())
    loc, uids, wins = tile_inputs(tile, season)
    if not wins:
        odir.mkdir(parents=True, exist_ok=True)
        info = {"tile": tile, "season": season, "skipped": "no parcels"}
        done.write_text(json.dumps(info))
        return info
    bbox = bbox_of(wins, grid)
    rdir = RAW / tile / str(season)
    if not (rdir / "_job.json").exists():
        cube = co.class_cube(con, grid["epsg"], bbox, f"{season}-{start}", f"{season}-{end}")
        for attempt in range(30):
            try:
                job = cube.create_job(title=f"capcausal {tile} {season}", out_format="GTiff")
                job.start_and_wait(print=lambda *a, **k: None, max_poll_interval=60)
                break
            except Exception as e:
                if "429" not in str(e) or attempt == 29:
                    raise
                log.info("%s %s: 429, retry in %d s", tile, season, 60 * (1 + attempt % 5))
                time.sleep(60 * (1 + attempt % 5))
        if job.status() != "finished":
            raise RuntimeError(f"job {job.job_id} status {job.status()}")
        md = job.describe()
        rdir.mkdir(parents=True, exist_ok=True)
        job.get_results().download_files(rdir)
        (rdir / "_job.json").write_text(json.dumps({"id": job.job_id, "costs": md.get("costs"),
                                                    "usage": md.get("usage"), "status": md.get("status")}))
    jm = json.loads((rdir / "_job.json").read_text())
    odir.mkdir(parents=True, exist_ok=True)
    shutil.copy(PC / tile / f"npix_{season}.parquet", odir.parent / f"npix_{season}.parquet")
    n_dates = 0
    for f in sorted(rdir.glob("*.tif")):
        date = f.stem.split("_")[-1][:10]
        with rasterio.open(f) as src:
            code = np.zeros(loc.shape, dtype="uint8")
            c0 = int(round((src.transform.c - grid["x0"]) / 20))
            r0 = int(round((grid["y0"] - src.transform.f) / 20))
            a = src.read(1)
            code[r0:r0 + a.shape[0], c0:c0 + a.shape[1]] = a
        df, n_clear = co.parcel_stats(code, loc, wins, uids)
        meta = {"item": f"cdse_{tile}_{date}", "date": date, "n_blocks": len(wins), "n_clear_blocks": n_clear,
                "baseline": "cdse_collection1", "job": jm["id"]}
        if len(df):
            df["date"] = pd.Timestamp(date)
            df.to_parquet(odir / f"{date}.parquet")
        (odir / f"{date}.json").write_text(json.dumps(meta))
        n_dates += 1
    info = {"tile": tile, "season": season, "job": jm["id"], "costs": jm.get("costs"), "n_dates": n_dates,
            "finished": datetime.now(timezone.utc).isoformat()}
    done.write_text(json.dumps(info))
    jobs = OUT / "jobs.csv"
    new = not jobs.exists()
    with jobs.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(info))
        if new:
            w.writeheader()
        w.writerow(info)
    return info


def main(seasons, tiles, start, end, parallel):
    grids = json.loads((PC / "tile_grids.json").read_text())
    con = co.connect()
    todo = [(t, s) for s in seasons for t in tiles if (PC / t / f"pidx_{s}.npz").exists()]
    total = 0.0
    with ThreadPoolExecutor(parallel) as ex:
        futs = {}
        for t, s in todo:
            futs[ex.submit(run_one, con, t, s, grids[t], start, end)] = (t, s)
            time.sleep(2)
        for fu in as_completed(futs):
            t, s = futs[fu]
            try:
                info = fu.result()
                total += float(info.get("costs") or 0)
                log.info("%s %s done: %s dates, cost %s (cumulative %.0f)", t, s, info.get("n_dates"), info.get("costs"), total)
            except Exception as e:
                log.error("%s %s failed: %s", t, s, str(e)[:300])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", type=int, required=True)
    ap.add_argument("--tiles", nargs="*")
    ap.add_argument("--start", default="08-01")
    ap.add_argument("--end", default="10-31")
    ap.add_argument("--parallel", type=int, default=3)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    tiles = a.tiles or sorted(p.name for p in PC.iterdir() if p.is_dir() and len(p.name) == 5)
    main(a.seasons, tiles, a.start, a.end, a.parallel)
