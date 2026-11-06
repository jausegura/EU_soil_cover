#!/usr/bin/env python
"""Regression-discontinuity diagnostics."""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pyfixest as pf

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import analyze_glm6 as A

from capcausal.analysis import estimators as est
from capcausal.utils.paths import DATA_DIR, RESULTS_DIR, ROOT

OUT = RESULTS_DIR / "rd_diagnostics"


def coverage_covariates(seasons):
    """Satellite coverage of each holding's harvested parcels: mean number of valid Aug-Sep observations and share of parcels with at least three (all declared harvested parcels in the denominator)."""
    rows = []
    for s in seasons:
        t = pd.read_parquet(DATA_DIR / "interim" / "dk" / f"season_{s}.parquet", columns=["uid", "journalnr", "crop_group"])
        t = t[t.crop_group.isin(A.HARVESTED | {"maize_silage", "maize_grain"})]
        o = pd.read_parquet(DATA_DIR / "processed_cdse" / f"parcel_season_{s}.parquet", columns=["uid", "n_augsep"])
        t = t.merge(o, on="uid", how="left")
        t["n_augsep"] = t.n_augsep.fillna(0)
        g = t.groupby("journalnr").agg(mean_nobs=("n_augsep", "mean"), share_ge3obs=("n_augsep", lambda v: (v >= 3).mean())).reset_index()
        g["season"] = s
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def holding_covariates(seasons):
    rows = []
    for s in seasons:
        p = pd.read_parquet(DATA_DIR / "processed_cdse" / f"parcel_season_{s}.parquet",
                            columns=["journalnr", "glm6_zone", "autumn_sown_next", "area_ha", "organic", "agri_area_ha"])
        g = p.groupby("journalnr").agg(sand_share=("glm6_zone", lambda v: (v == "sand_1feb").mean()),
                                        autumn_share=("autumn_sown_next", lambda v: v.fillna(False).astype(float).mean()),
                                        n_parcels_cov=("area_ha", "size"), mean_parcel_ha=("area_ha", "mean"),
                                        organic_share=("organic", lambda v: v.fillna(0).mean()),
                                        agri_area_ha=("agri_area_ha", "first")).reset_index()
        g["season"] = s
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def binned(df, x, y, cutoff, h, width=0.5):
    d = df[(df[x] - cutoff).abs() <= h].copy()
    d["bin"] = np.floor((d[x] - cutoff) / width) * width + width / 2
    return d.groupby("bin").agg(y=(y, "mean"), n=(y, "size")).reset_index()


def diff_in_disc_poly(df, y, x, cutoff, h, post, cluster, order, fe="season"):
    d = df.copy()
    d["_r"] = d[x] - cutoff
    d = d[d._r.abs() <= h].copy()
    d["_D"] = (d[x] <= cutoff).astype(float)
    d["_P"] = d[post].astype(float)
    d["_w"] = (1 - d._r.abs() / h).clip(lower=0)
    d = d[d._w > 0]
    terms = ["_D"] + [f"_r{k}" for k in range(1, order + 1)] + [f"_D:_r{k}" for k in range(1, order + 1)]
    for k in range(1, order + 1):
        d[f"_r{k}"] = d._r ** k
    rhs = " + ".join(terms) + " + " + " + ".join(f"_P:{t}" for t in terms) + f" | {fe}"
    f = pf.feols(f"{y} ~ {rhs}", data=d, vcov={"CRV1": cluster}, weights="_w")
    t = f.tidy().loc["_P:_D"]
    return t["Estimate"], t["Std. Error"], t["2.5%"], t["97.5%"], int(f._N)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d_all = A.load(list(range(2019, 2026)))
    fC = A.farm_outcomes(d_all, 3)
    fC = fC[(fC.agri_area_ha > 0) & (fC.organic < 0.5)].copy()
    fC["post"] = (fC.season >= 2024).astype(int)
    fC_main = fC[fC.season >= 2023]
    cov = holding_covariates(range(2019, 2026))
    fC_main = fC_main.merge(cov.drop(columns="agri_area_ha"), on=["journalnr", "season"], how="left")
    fD = pd.read_parquet(DATA_DIR / "processed" / "farm_rotation.parquet")
    sys.path.insert(0, str(ROOT / "scripts"))
    import analyze_rotation as R
    sD = R.sample(fD)
    pn, tabs = {}, []
    for name, df, x, y, post_var, pre_mask in (("C", fC_main, "agri_area_ha", "share_bare_augsep", "post", fC_main.season == 2023),
                                                ("D", sD, "arable_area_ha", "share_change", "post", sD.year <= 2022)):
        w = df[(df[x] - 10).abs() <= 5]
        cnt = w.groupby([w[post_var], w[x] <= 10]).size()
        pn[f"Rd{name}NbelowPre"], pn[f"Rd{name}NabovePre"] = f"{cnt.get((0, True), 0):,}", f"{cnt.get((0, False), 0):,}"
        pn[f"Rd{name}NbelowPost"], pn[f"Rd{name}NabovePost"] = f"{cnt.get((1, True), 0):,}", f"{cnt.get((1, False), 0):,}"
        for per, m in (("pre", df[post_var] == 0), ("post", df[post_var] == 1)):
            binned(df[m], x, y, 10, 8).assign(design=name, period=per).to_csv(OUT / f"bins_{name}_{per}.csv", index=False)
        from rdrobust import rdrobust
        g = df[df[post_var] == 1]
        r = rdrobust(y=g[y].values, x=(10 - g[x].values), c=0.0)
        h_opt = float(r.bws.iloc[0, 0])
        pn[f"Rd{name}Hopt"] = f"{h_opt:.2f}"
        rows = []
        for order in (1, 2):
            for hf, hl in ((0.5, "half"), (1.0, "opt"), (1.5, "onehalf"), (None, "main5")):
                h = 5.0 if hf is None else h_opt * hf
                e, se, lo, hi, n = diff_in_disc_poly(df, y, x, 10.0, h, post_var, "cluster", order, fe="season" if name == "C" else "year")
                rows.append({"design": name, "order": order, "bandwidth": hl, "h": h, "est": e, "se": se, "lo": lo, "hi": hi, "n": n})
        t = pd.DataFrame(rows)
        tabs.append(t)
        pn[f"Rd{name}PolyMin"], pn[f"Rd{name}PolyMax"] = f"{t.est.min():.3f}", f"{t.est.max():.3f}"
        pn[f"Rd{name}Quad"] = f"{t[(t.order == 2) & (t.bandwidth == 'main5')].est.iloc[0]:.3f}"
        pn[f"Rd{name}QuadLo"] = f"{t[(t.order == 2) & (t.bandwidth == 'main5')].lo.iloc[0]:.3f}"
        pn[f"Rd{name}QuadHi"] = f"{t[(t.order == 2) & (t.bandwidth == 'main5')].hi.iloc[0]:.3f}"
    covs = ("sand_share", "autumn_share", "n_parcels_cov", "mean_parcel_ha", "organic_share", "arable_area_ha",
            "mean_nobs", "share_ge3obs")
    fAll = fC.merge(cov.drop(columns="agri_area_ha"), on=["journalnr", "season"], how="left").merge(
        coverage_covariates(range(2019, 2026)), on=["journalnr", "season"], how="left")
    cc = []
    for c in covs:
        r = est.rd_by_year(fAll, c, "agri_area_ha", 10.0, "season", cluster="cluster")
        cc.append(r.assign(covariate=c))
    cc = pd.concat(cc, ignore_index=True)
    cc.to_csv(OUT / "covariate_by_year_C.csv", index=False)
    cc.loc[cc.year == 2023].to_csv(OUT / "covariate_continuity_C.csv", index=False)
    fm = fAll[fAll.season >= 2023]
    cd = []
    for c in covs:
        e, se, lo, hi, n = diff_in_disc_poly(fm.dropna(subset=[c]), c, "agri_area_ha", 10.0, 5.0, "post", "cluster", 1)
        z = e / se
        from scipy.stats import norm
        cd.append({"covariate": c, "est": e, "se": se, "lo": lo, "hi": hi, "p": 2 * norm.sf(abs(z)), "n": n})
    inc = []
    for s_ in range(2019, 2026):
        fa = pd.read_parquet(DATA_DIR / "interim" / "dk" / f"farms_{s_}.parquet", columns=["journalnr", "agri_area_ha", "cvr"])
        fa = fa[fa.agri_area_ha > 0].copy()
        fa["season"] = s_
        fa["included"] = fa.journalnr.isin(set(fC.loc[fC.season == s_, "journalnr"])).astype(float)
        fa["cluster"] = fa.cvr.fillna(fa.journalnr).astype(str)
        inc.append(fa)
    inc = pd.concat(inc, ignore_index=True)
    inc["post"] = (inc.season >= 2024).astype(int)
    e, se, lo, hi, n = diff_in_disc_poly(inc[inc.season >= 2023], "included", "agri_area_ha", 10.0, 5.0, "post", "cluster", 1)
    from scipy.stats import norm
    cd.append({"covariate": "included", "est": e, "se": se, "lo": lo, "hi": hi, "p": 2 * norm.sf(abs(e / se)), "n": n})
    ry = est.rd_by_year(inc, "included", "agri_area_ha", 10.0, "season", cluster="cluster").assign(covariate="included")
    cc = pd.concat([cc, ry], ignore_index=True)
    cc.to_csv(OUT / "covariate_by_year_C.csv", index=False)
    cd = pd.DataFrame(cd)
    cd.to_csv(OUT / "covariate_did_C.csv", index=False)
    c23 = cc[(cc.year == 2023) & (cc.covariate.isin(covs[:6]))]
    pn["RdCCovMinP"] = f"{c23.p.min():.3f}"
    pn["RdCCovNsig"] = str(int((c23.p < 0.05).sum()))
    pn["RdCCovN"] = str(len(c23))
    pn["RdCCovDidNsig"] = str(int((cd.p < 0.05).sum()))
    pn["RdCCovDidN"] = str(len(cd))
    words = ["none", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    nsig = int((cd.p < 0.05).sum())
    pn["RdCCovDidNsigWord"] = words[nsig] if nsig < len(words) else str(nsig)
    pn["RdCCovDidMinP"] = f"{cd.p.min():.3f}"
    pn["RdCCovDidMinPVar"] = cd.loc[cd.p.idxmin(), "covariate"]
    pn["RdCInclDid"] = f"{cd.set_index('covariate').loc['included', 'est']:.3f}"
    pn["RdCInclDidP"] = f"{cd.set_index('covariate').loc['included', 'p']:.3f}"
    dens = pd.read_csv(RESULTS_DIR / "design_c" / "density_tests.csv")
    dens = dens[dens.year.between(2019, 2025)]
    pn["RdCDensNsig"] = str(int((dens.p < 0.05).sum()))
    pn["RdCDensSigYears"] = ", ".join(str(int(y)) for y in dens.loc[dens.p < 0.05, "year"]) or "none"
    desc = pd.read_csv(RESULTS_DIR / "design_a" / "descriptives.csv")
    pre_conv = desc[(desc.conventional == 1) & (desc.season <= 2022)]
    gap = (np.average(pre_conv[pre_conv.succ == "autumn"].share_bare, weights=pre_conv[pre_conv.succ == "autumn"].n)
           - np.average(pre_conv[pre_conv.succ == "spring"].share_bare, weights=pre_conv[pre_conv.succ == "spring"].n))
    pn["RdCPrespecEffect"] = f"{gap:.3f}"
    pn["RdCPrespecHalf"] = f"{gap / 2:.3f}"
    pd.concat(tabs).to_csv(OUT / "poly_bandwidth.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps({"paper_numbers": pn}, indent=2))
    t = pd.concat(tabs)
    lines = [r"\begin{tabular}{llcc}", r"\toprule", r"Design & Polynomial, bandwidth & Estimate (SE) & $n$ \\", r"\midrule"]
    lab = {"half": "0.5 $\\times$ CCT", "opt": "CCT-optimal", "onehalf": "1.5 $\\times$ CCT", "main5": "5 ha (main)"}
    for r in t.itertuples():
        lines.append(f"{r.design} & order {r.order}, {lab[r.bandwidth]} ({r.h:.1f} ha) & {r.est:.3f} ({r.se:.3f}) & {r.n:,} \\\\".replace(",", "{,}", 0))
    lines += [r"\bottomrule", r"\end{tabular}"]
    (RESULTS_DIR / "tables" / "table_rd_diag.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(t.round(3).to_string()); print(cc.round(3).to_string()); print(pn)


if __name__ == "__main__":
    main()
