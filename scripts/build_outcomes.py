#!/usr/bin/env python
"""Parcel-season outcome tables."""
from __future__ import annotations

import argparse
import logging

import pandas as pd
import yaml

from capcausal.analysis import outcomes as oc
from capcausal.analysis.panel import season_table
from capcausal.utils.paths import PROCESSED_DIR, ROOT

WINDOWS = {"augsep": ("08-01", "09-30"), "oct": ("10-01", "10-31"), "oct2to20": ("10-02", "10-20"),
           "augoct": ("08-01", "10-31")}
log = logging.getLogger("outcomes")


def build(season: int, ndti: float, maj: float, min_pix: int, min_valid_frac: float) -> pd.DataFrame:
    obs = oc.load_obs(season, min_pix=min_pix, min_valid_frac=min_valid_frac)
    if obs.empty:
        log.warning("no observations for %d", season)
        return obs
    obs = oc.classify_obs(obs, ndti_thr=ndti, maj=maj)
    m = oc.window_metrics(obs, season, WINDOWS)
    sp = oc.bare_spell(obs[obs.bare | obs.green], season)
    cloud = obs.groupby("uid").agg(tile=("tile", "first"), n_obs_total=("date", "size"))
    tab = season_table(season)
    out = tab.merge(m, on="uid", how="inner").merge(sp, on="uid", how="left").merge(
        cloud.reset_index(), on="uid", how="left")
    out["bare_spell_days"] = out.bare_spell_days.fillna(0)
    return out


if __name__ == "__main__":
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())["outcomes"]["bare_soil"]
    ap = argparse.ArgumentParser()
    ap.add_argument("seasons", nargs="+", type=int)
    ap.add_argument("--ndti", type=float, default=cfg.get("ndti_max", 0.10))
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out_dir = PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in a.seasons:
        df = build(s, a.ndti, cfg.get("majority", 0.5), cfg.get("min_pix", 8), cfg.get("min_valid_frac", 0.5))
        if len(df):
            df.to_parquet(out_dir / f"parcel_season_{s}.parquet")
            log.info("season %d: %d parcel-seasons", s, len(df))
