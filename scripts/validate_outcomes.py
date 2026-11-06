#!/usr/bin/env python
"""Calibration of the NDTI threshold against legally implied reference states."""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

from capcausal.analysis import outcomes as oc
from capcausal.analysis.panel import harvested_arable, season_table
from capcausal.utils.paths import RESULTS_DIR

THRESHOLDS = (0.06, 0.08, 0.10, 0.12, 0.14)
log = logging.getLogger("validate")


def labelled(season: int, min_obs: int) -> pd.DataFrame:
    obs = oc.load_obs(season)
    if obs.empty:
        return obs
    tab = season_table(season)
    lab = pd.Series(pd.NA, index=tab.index, dtype="object")
    lab[harvested_arable(tab) & tab.autumn_sown_next.fillna(False).astype(bool) &
        ~tab.crop_group.isin(["maize_silage", "maize_grain", "beet"])] = "P1_winter_next"
    lab[tab.crop_group == "grass_permanent"] = "N1_perm_grass"
    lab[(tab.crop_group == "grass_rotation") & (tab.crop_group_next == "grass_rotation")] = "N2_grass_kept"
    org = tab["organic"].fillna(0) if "organic" in tab else 0
    lab[tab.crop_group.isin(["cereal_winter", "cereal_spring"]) & (tab.glm6_zone == "sand_1feb") & (org == 0) &
        tab.crop_group_next.isin(["cereal_spring", "legume", "maize_silage", "rapeseed_spring"])] = "N3_stubble_ban"
    tab["label"] = lab
    tab = tab[tab.label.notna()][["uid", "label", "glm6_zone", "area_ha"]]
    w = obs[(obs.date >= f"{season}-08-01") & (obs.date <= f"{season}-10-31")].merge(tab, on="uid")
    rows = []
    for thr in THRESHOLDS:
        c = f"f_bare_ndti_lt{int(round(thr * 100)):02d}"
        w["_b"] = w[c] >= 0.5
        g = w.groupby(["uid", "label", "tile", "glm6_zone"]).agg(n=("_b", "size"), any_bare=("_b", "max"),
                                                                share_bare=("_b", "mean"),
                                                                n_bare=("_b", "sum")).reset_index()
        g["thr"] = thr
        rows.append(g[g.n >= min_obs])
    out = pd.concat(rows, ignore_index=True)
    out["season"] = season
    return out


def metrics(d: pd.DataFrame, by: list[str], neg: tuple[str, ...] = ("N3",)) -> pd.DataFrame:
    """Sensitivity on P1 and specificity on the negative sets whose label starts with `neg`."""
    pos = d[d.label.str.startswith("P")].groupby(by + ["thr"]).any_bare.agg(["mean", "size"]).rename(
        columns={"mean": "sensitivity", "size": "n_pos"})
    neg = d[d.label.str.startswith(neg)].groupby(by + ["thr"]).any_bare.agg(["mean", "size"]).rename(
        columns={"mean": "fpr", "size": "n_neg"})
    m = pos.join(neg, how="outer")
    m["specificity"] = 1 - m.fpr
    m["youden"] = m.sensitivity + m.specificity - 1
    return m.drop(columns="fpr").reset_index()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", type=int, required=True)
    ap.add_argument("--calib", nargs="+", type=int, required=True)
    ap.add_argument("--min-obs", type=int, default=6)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = RESULTS_DIR / "validation"
    out.mkdir(parents=True, exist_ok=True)
    d = pd.concat([labelled(s, a.min_obs) for s in a.seasons], ignore_index=True)
    d["obs_tercile"] = d.groupby(["season", "thr"]).n.transform(lambda x: pd.qcut(x, 3, labels=["low", "mid", "high"],
                                                                                    duplicates="drop"))
    d["calib"] = d.season.isin(a.calib)
    cal = metrics(d[d.calib], [])
    best = float(cal.sort_values("youden", ascending=False).thr.iloc[0])
    by_season = metrics(d, ["season"])
    by_tile = metrics(d[d.thr == best], ["tile"])
    by_density = metrics(d[d.thr == best], ["obs_tercile"])
    by_zone = metrics(d[d.thr == best], ["glm6_zone"])
    hold = metrics(d[~d.calib & (d.thr == best)], [])
    for name, t in {"calibration": cal, "by_season": by_season, "by_tile": by_tile, "by_density": by_density,
                    "by_zone": by_zone, "holdout": hold}.items():
        t.to_csv(out / f"{name}.csv", index=False)
    hb = hold.iloc[0] if len(hold) else cal[cal.thr == best].iloc[0]
    ob = d[d.thr == best].groupby("label")[["n_bare", "n"]].sum()
    obs_rate = (ob.n_bare / ob.n).to_dict()
    summary = {
        "chosen_ndti_threshold": best, "calibration_seasons": a.calib, "seasons": a.seasons,
        "holdout_sensitivity": float(hb.sensitivity), "holdout_specificity": float(hb.specificity),
        "n_pos_holdout": int(hb.n_pos), "n_neg_holdout": int(hb.n_neg),
        "obs_bare_rate_by_label": obs_rate,
        "paper_numbers": {
            "ValNdtiThr": f"{best:.2f}",
            "ValSens": f"{float(hb.sensitivity):.2f}", "ValSpec": f"{float(hb.specificity):.2f}",
            "ValYouden": f"{float(hb.youden):.2f}",
            "ValNpos": f"{int(hb.n_pos):,}", "ValNneg": f"{int(hb.n_neg):,}",
            "ValSensMin": f"{by_season[by_season.thr == best].sensitivity.min():.2f}",
            "ValSensMax": f"{by_season[by_season.thr == best].sensitivity.max():.2f}",
            "ValSpecMin": f"{by_season[by_season.thr == best].specificity.min():.2f}",
            "ValSpecMax": f"{by_season[by_season.thr == best].specificity.max():.2f}",
            "ValObsBareP": f"{obs_rate.get('P1_winter_next', float('nan')):.3f}",
            "ValObsBareStubble": f"{obs_rate.get('N3_stubble_ban', float('nan')):.3f}",
            "ValObsBareGrass": f"{obs_rate.get('N1_perm_grass', float('nan')):.4f}",
            "ValSpecGrass": f"{float(metrics(d[~d.calib & (d.thr == best)], [], neg=('N1', 'N2')).specificity.iloc[0]):.3f}",
            "ValSensLowDens": f"{float(by_density.set_index('obs_tercile').sensitivity.get('low', float('nan'))):.2f}",
            "ValSensHighDens": f"{float(by_density.set_index('obs_tercile').sensitivity.get('high', float('nan'))):.2f}",
            "ValSpecLowDens": f"{float(by_density.set_index('obs_tercile').specificity.get('low', float('nan'))):.2f}",
            "ValSpecHighDens": f"{float(by_density.set_index('obs_tercile').specificity.get('high', float('nan'))):.2f}",
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("best threshold %.2f; holdout sens %.3f spec %.3f", best, hb.sensitivity, hb.specificity)
    print(cal.to_string())
