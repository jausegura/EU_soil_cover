#!/usr/bin/env python
"""Thinning with contiguous gaps, real date patterns and wet-date removal."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import yaml

from capcausal.analysis import outcomes as oc
from capcausal.analysis.panel import harvested_arable, season_table
from capcausal.utils.paths import DATA_DIR, RESULTS_DIR, ROOT

SEED, KS, MIN_FULL, N_DRAWS, MAX_PARCELS, MAX_PATTERNS = 42, (3, 6, 9), 14, 10, 4000, 300


def load(season, ndti):
    obs = oc.classify_obs(oc.load_obs(season), ndti_thr=ndti)
    obs = obs[(obs.date >= f"{season}-08-01") & (obs.date <= f"{season}-09-30")]
    tab = season_table(season)
    keep = tab.loc[harvested_arable(tab), ["uid", "x", "y"]]
    obs = obs.merge(keep, on="uid")
    return obs[["uid", "tile", "date", "bare", "x", "y"]]


def precip_lookup(season):
    f = DATA_DIR / "interim" / "weather" / "era5_dk_daily.parquet"
    w = pd.read_parquet(f)
    w = w[w.date.dt.year == season]
    return w


def main():
    ndti = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())["outcomes"]["bare_soil"]["ndti_max"]
    rng = np.random.default_rng(SEED)
    out = RESULTS_DIR / "revisit_realistic"
    out.mkdir(parents=True, exist_ok=True)
    from pyproj import Transformer
    tr = Transformer.from_crs(25832, 4326, always_xy=True)
    rows, drows = [], []
    for season in range(2019, 2026):
        obs = load(season, ndti)
        n = obs.groupby("uid").size()
        dense_ids = n[n >= MIN_FULL].index
        sparse = obs[obs.uid.isin(n[(n >= 3) & (n < MIN_FULL)].index)]
        if len(dense_ids) < 50:
            continue
        if len(dense_ids) > MAX_PARCELS:
            dense_ids = rng.choice(dense_ids, MAX_PARCELS, replace=False)
        dense = obs[obs.uid.isin(dense_ids)].sort_values(["uid", "date"])
        w = precip_lookup(season)
        wkey = w.set_index(["lat", "lon", "date"]).precip_mm
        sparse_by_tile = {}
        for t, g in sparse.groupby("tile"):
            pats = g.groupby("uid").date.apply(lambda s: frozenset(s)).tolist()
            if len(pats) > MAX_PATTERNS:
                pats = [pats[i] for i in rng.choice(len(pats), MAX_PATTERNS, replace=False)]
            sparse_by_tile[t] = pats
        res = {(sch, k): [] for sch in ("contiguous", "real_mask", "wet_first") for k in KS}
        for uid, g in dense.groupby("uid", sort=False):
            dates, bare = g.date.values, g.bare.values.astype(float)
            full_share, full_any = bare.mean(), bare.max()
            L = len(dates)
            lon, lat = tr.transform(g.x.iloc[0], g.y.iloc[0])
            la, lo = round(lat * 2) / 2, round(lon * 2) / 2
            pr = np.array([wkey.get((la, lo, pd.Timestamp(d)), 0.0) for d in dates])
            tile = g.tile.iloc[0]
            for k in KS:
                for j in range(N_DRAWS):
                    gap = L - k
                    start = rng.integers(0, L - gap + 1)
                    keep = np.r_[np.arange(0, start), np.arange(start + gap, L)]
                    res[("contiguous", k)].append((bare[keep].mean() - full_share, bare[keep].max(), full_any, j))
                    dset = set(dates)
                    pats = [p for p in sparse_by_tile.get(tile, []) if len(p & dset) >= k]
                    if pats:
                        p = pats[rng.integers(len(pats))]
                        idx = np.array([i for i, d in enumerate(dates) if d in p])
                        idx = rng.choice(idx, k, replace=False) if len(idx) > k else idx
                    else:
                        idx = rng.choice(L, k, replace=False)
                    res[("real_mask", k)].append((bare[idx].mean() - full_share, bare[idx].max(), full_any, j))
                    order = np.lexsort((rng.random(L), -pr))
                    keep = np.sort(order[gap:])
                    res[("wet_first", k)].append((bare[keep].mean() - full_share, bare[keep].max(), full_any, j))
        for (sch, k), v in res.items():
            a = np.array(v)
            pos = a[:, 2] > 0
            rows.append({"season": season, "scheme": sch, "k": k, "n_parcels": len(dense_ids), "bias_share": a[:, 0].mean(),
                         "rmse_share": np.sqrt((a[:, 0] ** 2).mean()), "detect_any": a[pos, 1].mean() if pos.any() else np.nan})
            for j in range(N_DRAWS):
                m = a[:, 3] == j
                drows.append({"season": season, "scheme": sch, "k": k, "draw": j, "n_parcels": len(dense_ids),
                              "bias_share": a[m, 0].mean(), "detect_any": a[m & pos, 1].mean()})
        print(season, len(dense_ids), flush=True)
    cv = pd.DataFrame(rows)
    cv.to_csv(out / "curves.csv", index=False)
    pooled = cv.groupby(["scheme", "k"]).apply(lambda x: pd.Series({
        "bias_share": np.average(x.bias_share, weights=x.n_parcels), "rmse_share": np.average(x.rmse_share, weights=x.n_parcels),
        "detect_any": np.average(x.detect_any, weights=x.n_parcels)})).reset_index()
    dr = pd.DataFrame(drows)
    dr.to_csv(out / "draws.csv", index=False)
    pdw = dr.groupby(["scheme", "k", "draw"]).apply(lambda x: pd.Series({
        "bias_share": np.average(x.bias_share, weights=x.n_parcels),
        "detect_any": np.average(x.detect_any, weights=x.n_parcels)})).reset_index()
    q = pdw.groupby(["scheme", "k"]).agg(bias_lo=("bias_share", lambda v: v.quantile(0.025)), bias_hi=("bias_share", lambda v: v.quantile(0.975)),
                                         det_lo=("detect_any", lambda v: v.quantile(0.025)), det_hi=("detect_any", lambda v: v.quantile(0.975))).reset_index()
    pooled = pooled.merge(q, on=["scheme", "k"])
    pooled.to_csv(out / "pooled.csv", index=False)
    pk = pooled.set_index(["scheme", "k"])
    pn = {}
    for sch, tag in (("contiguous", "Cont"), ("real_mask", "Mask"), ("wet_first", "Wet")):
        for k, kt in ((3, "Three"), (6, "Six"), (9, "Nine")):
            pn[f"Rr{tag}Bias{kt}"] = f"{pk.loc[(sch, k), 'bias_share']:+.3f}"
            pn[f"Rr{tag}Det{kt}"] = f"{pk.loc[(sch, k), 'detect_any']:.2f}"
    pn["RrBiasMaxAbs"] = f"{pooled.bias_share.abs().max():.3f}"
    pn["RrNparcels"] = f"{int(cv.groupby('season').n_parcels.first().sum()):,}"
    (out / "summary.json").write_text(json.dumps({"paper_numbers": pn}, indent=2))
    print(pooled.round(4).to_string())


if __name__ == "__main__":
    main()
