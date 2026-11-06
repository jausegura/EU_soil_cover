#!/usr/bin/env python
"""Validation on the window and population of the policy tests, on two scales."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import stats

from capcausal.analysis.panel import harvested_arable, season_table
from capcausal.utils.paths import PROCESSED_DIR, RESULTS_DIR, ROOT

MIN_OBS = 3
SPRING_NEXT = {"cereal_spring", "maize_silage", "maize_grain", "legume", "beet", "potatoes", "rapeseed_spring"}
RULES = {"base": "NDVI$<$0.30 and NDTI$<$0.10 (main)", "tau06": "NDTI boundary 0.06", "tau08": "NDTI boundary 0.08",
         "tau12": "NDTI boundary 0.12", "tau14": "NDTI boundary 0.14", "ndvi": "NDVI$<$0.30 only",
         "maj30": "30\\% of pixels bare", "maj70": "70\\% of pixels bare", "vf30": "$\\ge$30\\% clear pixels",
         "vf70": "$\\ge$70\\% clear pixels", "vf90": "$\\ge$90\\% clear pixels"}


def cluster_ci(y: np.ndarray, cl: np.ndarray):
    """Mean with a CR1 cluster-robust 95% CI using t(G-1) critical values."""
    y = np.asarray(y, float)
    m = y.mean()
    s = pd.Series(y - m).groupby(np.asarray(cl)).sum()
    G, n = len(s), len(y)
    se = np.sqrt(G / (G - 1) * (s ** 2).sum()) / n
    q = stats.t.ppf(0.975, G - 1)
    return m, m - q * se, m + q * se, G


def labelled(season: int) -> pd.DataFrame:
    tab = season_table(season)
    tab = tab.copy()
    aut = tab.autumn_sown_next.fillna(False).astype(bool)
    tab["succ"] = np.where(aut, "autumn", np.where(tab.crop_group_next.isin(SPRING_NEXT), "spring", "other"))
    tab["system"] = np.where(tab["organic"].fillna(0) == 1, "organic", "conventional")
    pop = harvested_arable(tab) & (tab.overlap_next >= 0.5) & (tab.glm6_zone != "unknown")
    lab = pd.Series(pd.NA, index=tab.index, dtype="object")
    lab[pop & (tab.succ == "autumn") & ~tab.crop_group.isin(["maize_silage", "maize_grain", "beet"])] = "P"
    lab[pop & tab.crop_group.isin(["cereal_winter", "cereal_spring"]) & (tab.glm6_zone == "sand_1feb") & (tab.succ == "spring")
        & (tab.system == "conventional")] = "N3"
    lab[(tab.crop_group == "grass_permanent") & (tab.glm6_zone != "unknown")] = "N1"
    tab["label"] = lab
    tab = tab[tab.label.notna()][["uid", "label", "succ", "system", "glm6_zone"]]
    alt = pd.read_parquet(PROCESSED_DIR / f"alt_augsep_{season}.parquet")
    tiles = pd.read_parquet(PROCESSED_DIR / f"parcel_season_{season}.parquet", columns=["uid", "tile"])
    g = tab.merge(alt, on="uid").merge(tiles, on="uid")
    g["season"] = season
    g["cluster"] = g.tile + "_" + str(season)
    return g


def rates(sub: pd.DataFrame, rule: str = "base") -> dict:
    n, a, sh = f"n_{rule}", f"any_{rule}", f"share_{rule}"
    sub = sub[sub[n] >= MIN_OBS]
    rec = {}
    for lab, key, flip in (("P", "sens", False), ("N3", "spec", True), ("N1", "spec_grass", True)):
        x = sub[sub.label == lab]
        if len(x) < 30:
            continue
        v = x[a].values.astype(float)
        m, lo, hi, k = cluster_ci(1 - v if flip else v, x.cluster.values)
        rec.update({key: m, f"{key}_lo": lo, f"{key}_hi": hi, f"n_{lab}": len(x), f"k_{lab}": k,
                    f"obs_bare_{lab}": float((x[sh] * x[n]).sum() / x[n].sum()), f"nobs_{lab}": int(x[n].sum())})
    return rec


def logistic_baseline(d: pd.DataFrame) -> dict:
    """Multivariate baseline: logistic regression on parcel-season spectral summaries, trained on 2019-2021 and tested on 2022-2025 (sandy zone, P vs N3)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    feats = [c for c in d.columns if c.startswith(("mean_", "max_"))] + ["n_base"]
    x = d[(d.glm6_zone == "sand_1feb") & d.label.isin(["P", "N3"]) & (d.n_base >= MIN_OBS)].dropna(subset=feats)
    y = (x.label == "P").astype(int)
    tr, te = x.season <= 2021, x.season >= 2022
    sc = StandardScaler().fit(x.loc[tr, feats])
    m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(x.loc[tr, feats]), y[tr])
    p_tr = m.predict_proba(sc.transform(x.loc[tr, feats]))[:, 1]
    thr = max(np.unique(np.round(p_tr, 3)), key=lambda t: (p_tr[y[tr] == 1] >= t).mean() - (p_tr[y[tr] == 0] >= t).mean())
    p = m.predict_proba(sc.transform(x.loc[te, feats]))[:, 1] >= thr
    yt = y[te].values
    xs = x[te]
    se_ = cluster_ci(p[yt == 1].astype(float), xs.cluster.values[yt == 1])
    sp_ = cluster_ci(1 - p[yt == 0].astype(float), xs.cluster.values[yt == 0])
    return {"rule": "logistic", "sens": se_[0], "sens_lo": se_[1], "sens_hi": se_[2], "spec": sp_[0], "spec_lo": sp_[1],
            "spec_hi": sp_[2], "n_P": int((yt == 1).sum()), "n_N3": int((yt == 0).sum())}


def main():
    out = RESULTS_DIR / "validation_aligned"
    out.mkdir(parents=True, exist_ok=True)
    d = pd.concat([labelled(s) for s in range(2019, 2026)], ignore_index=True)
    rows = []
    subsets = [("all", d), ("heldout_2022_2025", d[d.season >= 2022]), ("pre_2019_2021", d[d.season <= 2021]),
               *[(f"season_{s}", d[d.season == s]) for s in range(2019, 2026)],
               *[(f"zone_{z}", d[d.glm6_zone == z]) for z in ("sand_1feb", "loam_20oct", "other_1oct")],
               *[(f"system_{x}", d[d.system == x]) for x in ("conventional", "organic")]]
    for name, sub in subsets:
        rows.append({"subset": name, **rates(sub)})
    r = pd.DataFrame(rows)
    r.to_csv(out / "rates.csv", index=False)
    p = d[(d.label == "P") & (d.n_base >= MIN_OBS)].copy()
    p["any"] = p.any_base.astype(float)
    p["ts"] = p.cluster
    fz = pf.feols("any ~ C(glm6_zone, Treatment('sand_1feb')) | ts", data=p, vcov={"CRV1": "ts"})
    zt = fz.tidy().reset_index()
    zt.to_csv(out / "zone_within_tile.csv", index=False)
    sand = d[(d.glm6_zone == "sand_1feb") & (d.season >= 2022)]
    rr = []
    for rule in RULES:
        x = rates(sand, rule)
        rr.append({"rule": rule, **{k: x.get(k) for k in ("sens", "sens_lo", "sens_hi", "spec", "spec_lo", "spec_hi", "spec_grass", "n_P", "n_N3")}})
    rr.append(logistic_baseline(d))
    rr = pd.DataFrame(rr)
    rr["youden"] = rr.sens + rr.spec - 1
    rr.to_csv(out / "rules.csv", index=False)
    print(r.round(3).to_string()); print(zt.round(4).to_string()); print(rr.round(3).to_string())
    h = r.set_index("subset").loc["heldout_2022_2025"]
    seas = r[r.subset.str.startswith("season_")]
    sysr = r.set_index("subset")
    zw = zt.set_index("Coefficient")
    zl = [c for c in zw.index if "loam" in c][0]
    rb = rr.set_index("rule")
    pn = {"VaSens": f"{h.sens:.2f}", "VaSensLo": f"{h.sens_lo:.2f}", "VaSensHi": f"{h.sens_hi:.2f}",
          "VaSpec": f"{h.spec:.2f}", "VaSpecLo": f"{h.spec_lo:.2f}", "VaSpecHi": f"{h.spec_hi:.2f}",
          "VaSpecGrass": f"{h.spec_grass:.3f}",
          "VaNpos": f"{int(h.n_P):,}", "VaNneg": f"{int(h.n_N3):,}", "VaKclusters": f"{int(h.k_P)}",
          "VaSensMin": f"{seas.sens.min():.2f}", "VaSensMax": f"{seas.sens.max():.2f}",
          "VaSpecMin": f"{seas.spec.min():.2f}", "VaSpecMax": f"{seas.spec.max():.2f}",
          "VaObsBarePos": f"{h.obs_bare_P:.3f}", "VaObsBareNeg": f"{h.obs_bare_N3:.3f}", "VaObsBareGrass": f"{h.obs_bare_N1:.4f}",
          "VaSensOrg": f"{sysr.loc['system_organic', 'sens']:.2f}", "VaSensOrgLo": f"{sysr.loc['system_organic', 'sens_lo']:.2f}",
          "VaSensOrgHi": f"{sysr.loc['system_organic', 'sens_hi']:.2f}",
          "VaSensConv": f"{sysr.loc['system_conventional', 'sens']:.2f}",
          "VaObsPosOrg": f"{sysr.loc['system_organic', 'obs_bare_P']:.3f}", "VaObsPosConv": f"{sysr.loc['system_conventional', 'obs_bare_P']:.3f}",
          "VaSensSand": f"{sysr.loc['zone_sand_1feb', 'sens']:.2f}", "VaSensLoam": f"{sysr.loc['zone_loam_20oct', 'sens']:.2f}",
          "VaZoneLoamWithin": f"{zw.loc[zl, 'Estimate']:.3f}", "VaZoneLoamWithinLo": f"{zw.loc[zl, '2.5%']:.3f}",
          "VaZoneLoamWithinHi": f"{zw.loc[zl, '97.5%']:.3f}",
          "RuMainJ": f"{rb.loc['base', 'youden']:.2f}", "RuNdviJ": f"{rb.loc['ndvi', 'youden']:.2f}",
          "RuNdviSens": f"{rb.loc['ndvi', 'sens']:.2f}", "RuNdviSpec": f"{rb.loc['ndvi', 'spec']:.2f}",
          "RuLogitJ": f"{rb.loc['logistic', 'youden']:.2f}", "RuLogitSens": f"{rb.loc['logistic', 'sens']:.2f}",
          "RuLogitSpec": f"{rb.loc['logistic', 'spec']:.2f}",
          "RuMaxJ": f"{rr.youden.max():.2f}", "RuMaxRule": rr.loc[rr.youden.idxmax(), "rule"],
          "RuMainSens": f"{rb.loc['base', 'sens']:.2f}", "RuMainSpec": f"{rb.loc['base', 'spec']:.2f}"}
    (out / "summary.json").write_text(json.dumps({"paper_numbers": pn}, indent=2))
    AG = RESULTS_DIR / "tables"

    def ci(x, k):
        return "--" if pd.isna(x.get(k)) else f"{x[k]:.2f} ({x[k + '_lo']:.2f}--{x[k + '_hi']:.2f})"
    lab = {"all": "All seasons", "heldout_2022_2025": "Held out 2022--2025", "pre_2019_2021": "Calibration 2019--2021",
           **{f"season_{s}": str(s) for s in range(2019, 2026)}, "zone_sand_1feb": "Sandy (JB 1--4)",
           "zone_loam_20oct": "Loamy (JB 5, 6, 10, 11)", "zone_other_1oct": "Other soils",
           "system_conventional": "Conventional", "system_organic": "Organic"}
    L = [r"\begin{tabular}{lrcrcrr}", r"\toprule",
         r"Subset & $n$ P & Detection rate P & $n$ N3 & Specificity N3 & Obs.\ bare P & Obs.\ bare N3 \\", r"\midrule"]
    for x in r.to_dict("records"):
        nP = f"{int(x['n_P']):,}".replace(",", "{,}") if pd.notna(x.get("n_P")) else "--"
        nN = f"{int(x['n_N3']):,}".replace(",", "{,}") if pd.notna(x.get("n_N3")) else "--"
        oN = f"{x['obs_bare_N3']:.3f}" if pd.notna(x.get("obs_bare_N3")) else "--"
        L.append(f"{lab[x['subset']]} & {nP} & {ci(x, 'sens')} & {nN} & {ci(x, 'spec')} & {x['obs_bare_P']:.3f} & {oN} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (AG / "supp_val_aligned.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    L = [r"\begin{tabular}{lccc}", r"\toprule", r"Rule & Detection rate P & Specificity N3 & Youden's $J$ \\", r"\midrule"]
    for x in rr.to_dict("records"):
        name = RULES.get(x["rule"], "Logistic model on parcel spectral summaries")
        L.append(f"{name} & {ci(x, 'sens')} & {ci(x, 'spec')} & {x['youden']:.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (AG / "supp_rules.tex").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
