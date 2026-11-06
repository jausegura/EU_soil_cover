#!/usr/bin/env python
"""Uniform thinning of densely observed parcel-seasons."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import yaml

from capcausal.analysis import outcomes as oc
from capcausal.analysis.panel import harvested_arable, season_table
from capcausal.utils.paths import RESULTS_DIR, ROOT

SEED = 42
KS = (2, 3, 4, 5, 6, 8, 10, 12)
MIN_FULL = 14
N_DRAWS = 20
MAX_PARCELS = 40000


def parcel_series(season: int, ndti: float) -> pd.DataFrame:
    obs = oc.classify_obs(oc.load_obs(season), ndti_thr=ndti)
    obs = obs[(obs.date >= f"{season}-08-01") & (obs.date <= f"{season}-09-30")]
    tab = season_table(season)
    keep = tab.loc[harvested_arable(tab), "uid"]
    obs = obs[obs.uid.isin(keep)]
    n = obs.groupby("uid").size()
    dense = n[n >= MIN_FULL].index
    obs = obs[obs.uid.isin(dense)]
    return obs[["uid", "date", "bare"]]


def simulate(obs: pd.DataFrame, rng) -> pd.DataFrame:
    uids = obs.uid.unique()
    if len(uids) > MAX_PARCELS:
        uids = rng.choice(uids, MAX_PARCELS, replace=False)
    g = obs[obs.uid.isin(uids)].groupby("uid").bare.apply(np.asarray)
    full_share = g.apply(np.mean)
    full_any = g.apply(np.max)
    rows, draws = [], []
    for k in KS:
        shares, anys = [], []
        for arr in g.values:
            idx = np.array([rng.choice(len(arr), k, replace=False) for _ in range(N_DRAWS)])
            sub = arr[idx]
            shares.append(sub.mean(axis=1))
            anys.append(sub.max(axis=1))
        shares, anys = np.array(shares), np.array(anys)
        err = shares - full_share.values[:, None]
        pos = full_any.values.astype(bool)
        rows.append({"k": k, "n_parcels": len(g), "bias_share": float(err.mean()), "rmse_share": float(np.sqrt((err ** 2).mean())),
                     "detect_any": float(anys[pos].mean()), "mean_full_share": float(full_share.mean()),
                     "share_any_full": float(pos.mean())})
        for j in range(N_DRAWS):
            draws.append({"k": k, "draw": j, "n_parcels": len(g), "bias_share": float(err[:, j].mean()),
                          "detect_any": float(anys[pos, j].mean())})
    return pd.DataFrame(rows), pd.DataFrame(draws)


def eligible(season: int) -> dict:
    """Characteristics of the parcel-seasons eligible for thinning (>= MIN_FULL valid Aug-Sep observations) against all harvested arable parcel-seasons with >= 3 observations."""
    from capcausal.utils.paths import PROCESSED_DIR
    p = pd.read_parquet(PROCESSED_DIR / f"parcel_season_{season}.parquet",
                        columns=["uid", "crop_group", "glm6_zone", "area_ha", "organic", "n_augsep", "share_bare_augsep", "tile"])
    p = p[harvested_arable(p) & (p.n_augsep >= 3)]
    e = p[p.n_augsep >= MIN_FULL]
    f = lambda x: {"n": len(x), "sand": (x.glm6_zone == "sand_1feb").mean(), "area": x.area_ha.mean(),
                   "organic": x.organic.fillna(0).mean(), "share_bare": x.share_bare_augsep.mean(), "n_obs": x.n_augsep.mean(),
                   "tiles": x.tile.nunique()}
    return {"season": season, **{f"elig_{k}": v for k, v in f(e).items()}, **{f"all_{k}": v for k, v in f(p).items()}}


def main():
    ndti = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())["outcomes"]["bare_soil"]["ndti_max"]
    rng = np.random.default_rng(SEED)
    out = RESULTS_DIR / "revisit"
    out.mkdir(parents=True, exist_ok=True)
    curves, draws, elig = [], [], []
    for s in range(2019, 2026):
        c, dr = simulate(parcel_series(s, ndti), rng)
        c["season"] = s
        dr["season"] = s
        curves.append(c)
        draws.append(dr)
        elig.append(eligible(s))
        print(s, c[["k", "n_parcels", "rmse_share", "detect_any"]].round(3).to_dict("records")[:2])
    cv = pd.concat(curves, ignore_index=True)
    cv.to_csv(out / "curves.csv", index=False)
    pooled = cv.groupby("k").apply(lambda x: pd.Series({
        "rmse_share": np.average(x.rmse_share, weights=x.n_parcels),
        "bias_share": np.average(x.bias_share, weights=x.n_parcels),
        "detect_any": np.average(x.detect_any, weights=x.n_parcels)})).reset_index()
    dr = pd.concat(draws, ignore_index=True)
    dr.to_csv(out / "draws.csv", index=False)
    pdw = dr.groupby(["k", "draw"]).apply(lambda x: pd.Series({
        "bias_share": np.average(x.bias_share, weights=x.n_parcels),
        "detect_any": np.average(x.detect_any, weights=x.n_parcels)})).reset_index()
    q = pdw.groupby("k").agg(bias_lo=("bias_share", lambda v: v.quantile(0.025)), bias_hi=("bias_share", lambda v: v.quantile(0.975)),
                             det_lo=("detect_any", lambda v: v.quantile(0.025)), det_hi=("detect_any", lambda v: v.quantile(0.975))).reset_index()
    pooled = pooled.merge(q, on="k")
    pooled.to_csv(out / "pooled.csv", index=False)
    el = pd.DataFrame(elig)
    el.to_csv(out / "eligible.csv", index=False)
    pk = pooled.set_index("k")
    pn = {"RvMinFull": str(MIN_FULL), "RvNparcels": f"{int(cv.groupby('season').n_parcels.first().sum()):,}",
          "RvDetectThree": f"{pk.loc[3, 'detect_any']:.2f}", "RvDetectSix": f"{pk.loc[6, 'detect_any']:.2f}",
          "RvDetectTwelve": f"{pk.loc[12, 'detect_any']:.2f}",
          "RvRmseThree": f"{pk.loc[3, 'rmse_share']:.3f}", "RvRmseSix": f"{pk.loc[6, 'rmse_share']:.3f}",
          "RvRmseTwelve": f"{pk.loc[12, 'rmse_share']:.3f}",
          "RvBiasMaxAbs": f"{pooled.bias_share.abs().max():.4f}",
          "RvNeligible": f"{int(el.elig_n.sum()):,}", "RvEligShare": f"{el.elig_n.sum() / el.all_n.sum() * 100:.1f}",
          "RvEligSand": f"{np.average(el.elig_sand, weights=el.elig_n):.2f}", "RvAllSand": f"{np.average(el.all_sand, weights=el.all_n):.2f}",
          "RvEligBare": f"{np.average(el.elig_share_bare, weights=el.elig_n):.3f}", "RvAllBare": f"{np.average(el.all_share_bare, weights=el.all_n):.3f}",
          "RvNdraws": str(N_DRAWS), "RvDetThreeLo": f"{pk.loc[3, 'det_lo']:.2f}", "RvDetThreeHi": f"{pk.loc[3, 'det_hi']:.2f}"}
    (out / "summary.json").write_text(json.dumps({"paper_numbers": pn}, indent=2))
    print(pooled.round(4))


if __name__ == "__main__":
    main()
