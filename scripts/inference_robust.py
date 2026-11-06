#!/usr/bin/env python
"""Tile-season and season-level inference for the GAEC 6 contrasts."""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pyfixest as pf

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import analyze_glm6 as A

from capcausal.utils.paths import RESULTS_DIR, ROOT

OUT = RESULTS_DIR / "inference_robust"
Y = "share_bare_augsep"
FE = "tile^season + crop_group^treat + glm6_zone^treat"
CONTRASTS = {"A0": ("conventional == 1", "aut"), "A1": ("succ == 'spring'", "conventional"), "A2": ("succ == 'autumn'", "conventional")}


def prep(s, label):
    q, tr = CONTRASTS[label]
    d = s.query(q).copy()
    if tr == "aut":
        d["treat"] = (d.succ == "autumn").astype(int)
    else:
        d["treat"] = d[tr].astype(int)
    d["_tp"] = d.treat * d.post
    d["ts"] = d.tile.astype(str) + "_" + d.season.astype(str)
    return d


def fit(d, vcov):
    return pf.feols(f"{Y} ~ _tp | {FE}", data=d, vcov=vcov)


def tidy(f, name):
    t = f.tidy().loc["_tp"]
    return {"spec": name, "est": t["Estimate"], "se": t["Std. Error"], "p": t["Pr(>|t|)"], "lo": t["2.5%"], "hi": t["97.5%"]}


def wcr_season(f, d):
    """Restricted wild cluster bootstrap (WCR-t) with season clusters and Webb weights, enumerating all 6^G weight patterns."""
    import itertools
    x = np.asarray(f._X).ravel()
    y = np.asarray(f._Y).ravel()
    if len(x) != len(d):
        return {"spec": "wild_season_webb", "note": "length mismatch"}
    codes, _ = pd.factorize(d["season"])
    G = codes.max() + 1
    sxy = np.bincount(codes, weights=x * y, minlength=G)
    sxx_g = np.bincount(codes, weights=x * x, minlength=G)
    sxx = sxx_g.sum()

    def tstat(w):
        b = (w * sxy).sum(1) / sxx
        score = w * sxy - b[:, None] * sxx_g
        se = np.sqrt(G / (G - 1) * (score ** 2).sum(1)) / sxx
        return b / se
    b_hat = sxy.sum() / sxx
    score_hat = sxy - b_hat * sxx_g
    t_hat = b_hat / (np.sqrt(G / (G - 1) * (score_hat ** 2).sum()) / sxx)
    webb = np.array([-np.sqrt(1.5), -1, -np.sqrt(0.5), np.sqrt(0.5), 1, np.sqrt(1.5)])
    W = np.array(list(itertools.product(range(6), repeat=G)), dtype=np.int8)
    tb = np.concatenate([tstat(webb[W[i:i + 50000]]) for i in range(0, len(W), 50000)])
    return {"spec": "wild_season_webb", "est": float(b_hat), "p": float((np.abs(tb) >= abs(t_hat) - 1e-12).mean())}


def season_permutation(beta: pd.Series):
    """Randomization inference over seasons: the event-study coefficients (base season = 0) are the season-level treated-minus-comparison contrasts net of fixed effects; the statistic is the mean over the three 'post' seasons minus the mean over the other four, for all C(7,3) = 35 assignments."""
    import itertools

    from scipy import stats
    seasons = list(beta.index)
    obs_post = [s for s in seasons if s >= 2023]
    stat = lambda post: beta[post].mean() - beta[[s for s in seasons if s not in post]].mean()
    s_obs = stat(obs_post)
    perm = np.array([stat(list(c)) for c in itertools.combinations(seasons, 3)])
    tt = stats.ttest_ind(beta[obs_post], beta[[s for s in seasons if s not in obs_post]], equal_var=True)
    return {"spec": "season_permutation", "est": float(s_obs), "p": float((np.abs(perm) >= abs(s_obs) - 1e-12).mean()),
            "p_ttest": float(tt.pvalue), "rank": int((np.abs(perm) >= abs(s_obs) - 1e-12).sum())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d_all = A.load(list(range(2019, 2026)))
    s = A.sample_a(d_all, 3)
    rows, pn, es_rows = [], {}, []
    for label in CONTRASTS:
        d = prep(s, label)
        r = []
        r.append(tidy(fit(d, {"CRV1": "cluster"}), "crv_company"))
        r.append(tidy(fit(d, {"CRV1": "ts"}), "crv_tile_season"))
        try:
            r.append(tidy(fit(d, {"CRV1": "cluster + ts"}), "crv_twoway"))
        except Exception as e:
            r.append({"spec": "crv_twoway", "note": str(e)[:80]})
        g = d.groupby(["ts", "tile", "season", "treat", "crop_group", "glm6_zone"], observed=True).agg(
            y=(Y, "mean"), n=("uid", "size"), post=("post", "first")).reset_index()
        g["_tp"] = g.treat * g.post
        g[Y] = g.y
        fa = pf.feols(f"{Y} ~ _tp | {FE}", data=g, vcov={"CRV1": "ts"}, weights="n")
        r.append(tidy(fa, "agg_tile_season"))
        try:
            fb = fit(d, {"CRV1": "ts"})
            Xd = np.asarray(fb._X).ravel()
            Yd = np.asarray(fb._Y).ravel()
            b_hat = float(fb.coef().iloc[0])
            resid = Yd - b_hat * Xd
            cl_codes, cl_ids = pd.factorize(d.loc[fb._data.index if hasattr(fb, "_data") else d.index, "ts"] if False else d["ts"])
            if len(cl_codes) != len(Xd):
                raise RuntimeError("cluster vector length mismatch")
            rng = np.random.default_rng(42)
            sxx = float((Xd * Xd).sum())
            G = len(cl_ids)
            bs = np.empty(999)
            for r_ in range(999):
                w = rng.choice([-1.0, 1.0], size=G)[cl_codes]
                bs[r_] = float((Xd * (resid * w)).sum()) / sxx
            p_wild = float((np.abs(bs) >= abs(b_hat)).mean())
            ci_lo, ci_hi = b_hat + np.quantile(bs, 0.025), b_hat + np.quantile(bs, 0.975)
            r.append({"spec": "wild_cluster_boot", "est": b_hat, "p": p_wild, "lo": ci_lo, "hi": ci_hi})
        except Exception as e:
            r.append({"spec": "wild_cluster_boot", "note": str(e)[:120]})
        los = []
        for yr in sorted(d.season.unique()):
            sub = d[d.season != yr]
            if sub.post.nunique() == 2:
                los.append({"drop": int(yr), "est": fit(sub, {"CRV1": "ts"}).tidy().loc["_tp", "Estimate"]})
        lot = []
        for t in sorted(d.tile.unique()):
            sub = d[d.tile != t]
            lot.append({"drop": t, "est": fit(sub, {"CRV1": "ts"}).tidy().loc["_tp", "Estimate"]})
        los, lot = pd.DataFrame(los), pd.DataFrame(lot)
        r.append({"spec": "leave_one_season", "lo": los.est.min(), "hi": los.est.max(), "est": los.est.median()})
        r.append({"spec": "leave_one_tile", "lo": lot.est.min(), "hi": lot.est.max(), "est": lot.est.median()})
        fe_ = pf.feols(f"{Y} ~ i(season, treat, ref=2022) | {FE}", data=d, vcov={"CRV1": "ts"})
        pre = [c for c in fe_._coefnames if any(f"::{y}" in c for y in (2019, 2020, 2021))]
        try:
            w = fe_.wald_test(R=np.array([[1.0 if c == pc else 0.0 for c in fe_._coefnames] for pc in pre]))
            pw = float(w["pvalue"]) if isinstance(w, dict) else float(getattr(w, "pvalue", np.nan))
        except Exception:
            b = fe_.coef()[pre].values
            V = fe_._vcov[[fe_._coefnames.index(c) for c in pre]][:, [fe_._coefnames.index(c) for c in pre]]
            from scipy.stats import chi2
            stat = float(b @ np.linalg.solve(V, b))
            pw = float(chi2.sf(stat, len(pre)))
        r.append({"spec": "joint_pretrend_wald", "p": pw})
        es = fe_.tidy().reset_index()
        es["season"] = es.Coefficient.str.extract(r"::(\d+)").astype(int)
        es_rows.append(es.assign(contrast=label))
        fsn = fit(d, {"CRV1": "season"})
        r.append(tidy(fsn, "crv_season"))
        r.append(wcr_season(fsn, d))
        beta = es.set_index("season").Estimate.reindex(range(2019, 2026)).fillna(0.0)
        r.append(season_permutation(beta))
        r.append({"spec": "counts", "n": int(len(d)), "companies": int(d.cluster.nunique()), "tiles": int(d.tile.nunique()),
                  "tile_seasons": int(d.ts.nunique()), "seasons": int(d.season.nunique())})
        t = pd.DataFrame(r).assign(contrast=label)
        rows.append(t)
        ti = t.set_index("spec")
        tag = {"A0": "Zero", "A1": "One", "A2": "Two"}[label]
        pn[f"Ir{tag}Est"] = f"{ti.loc['crv_company', 'est']:.3f}"
        pn[f"Ir{tag}SeCompany"] = f"{ti.loc['crv_company', 'se']:.4f}"
        pn[f"Ir{tag}SeTs"] = f"{ti.loc['crv_tile_season', 'se']:.4f}"
        pn[f"Ir{tag}LoTs"] = f"{ti.loc['crv_tile_season', 'lo']:.3f}"
        pn[f"Ir{tag}HiTs"] = f"{ti.loc['crv_tile_season', 'hi']:.3f}"
        pn[f"Ir{tag}PTs"] = ("<0.001" if ti.loc['crv_tile_season', 'p'] < 0.001 else f"{ti.loc['crv_tile_season', 'p']:.3f}")
        if "p" in ti.columns and pd.notna(ti.loc["wild_cluster_boot", "p"]):
            pn[f"Ir{tag}PWild"] = f"{ti.loc['wild_cluster_boot', 'p']:.3f}"
        pn[f"Ir{tag}LosLo"], pn[f"Ir{tag}LosHi"] = f"{ti.loc['leave_one_season', 'lo']:.3f}", f"{ti.loc['leave_one_season', 'hi']:.3f}"
        pn[f"Ir{tag}LotLo"], pn[f"Ir{tag}LotHi"] = f"{ti.loc['leave_one_tile', 'lo']:.3f}", f"{ti.loc['leave_one_tile', 'hi']:.3f}"
        pn[f"Ir{tag}PreWaldP"] = ("<0.001" if ti.loc['joint_pretrend_wald', 'p'] < 0.001 else f"{ti.loc['joint_pretrend_wald', 'p']:.3f}")
        fp3 = lambda v: "<0.001" if v < 0.001 else f"{v:.3f}"
        pn[f"Ir{tag}SeSeason"] = f"{ti.loc['crv_season', 'se']:.4f}"
        pn[f"Ir{tag}PSeason"] = fp3(ti.loc["crv_season", "p"])
        pn[f"Ir{tag}PWebb"] = fp3(ti.loc["wild_season_webb", "p"])
        pn[f"Ir{tag}PPerm"] = f"{ti.loc['season_permutation', 'p']:.3f}"
        pn[f"Ir{tag}PermRank"] = str(int(ti.loc["season_permutation", "rank"]))
        pn[f"Ir{tag}PTtest"] = fp3(ti.loc["season_permutation", "p_ttest"])
        pn[f"Ir{tag}SeasonStat"] = f"{ti.loc['season_permutation', 'est']:.3f}"
        for k in ("companies", "tiles", "tile_seasons"):
            pn[f"Ir{tag}N{k.title().replace('_', '')}"] = f"{int(ti.loc['counts', k]):,}"
    pd.concat(es_rows, ignore_index=True).to_csv(OUT / "event_study_ts.csv", index=False)
    tab = pd.concat(rows, ignore_index=True)
    tab.to_csv(OUT / "table.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps({"paper_numbers": pn}, indent=2))
    labels = {"crv_company": "Clustered by company", "crv_tile_season": "Clustered by tile $\\times$ season",
              "crv_twoway": "Two-way (company, tile $\\times$ season)", "agg_tile_season": "Tile $\\times$ season cell means",
              "wild_cluster_boot": "Wild cluster bootstrap (tile $\\times$ season), $p$",
              "leave_one_season": "Leave one season out (range)", "leave_one_tile": "Leave one tile out (range)",
              "joint_pretrend_wald": "Joint pre-trend Wald test, $p$",
              "crv_season": "Clustered by season (7 clusters)",
              "wild_season_webb": "Wild cluster bootstrap (season, Webb weights), $p$",
              "season_permutation": "Permutation of post seasons (35 assignments), $p$"}
    fmt_n = lambda v: f"{int(v):,}".replace(",", "{,}")
    lines = [r"\begin{tabular}{lccc}", r"\toprule", r"Specification & A0 & A1 & A2 \\", r"\midrule"]
    for k, lab in (("n", "Parcel-seasons"), ("companies", "Companies"), ("tiles", "Tiles"), ("tile_seasons", "Tile-seasons")):
        lines.append(f"{lab} & " + " & ".join(fmt_n(tab[(tab.contrast == c) & (tab.spec == 'counts')][k].iloc[0]) for c in CONTRASTS) + r" \\")
    lines.append(r"\midrule")
    for spec, lab in labels.items():
        if spec == "crv_season":
            lines.append(r"\multicolumn{4}{l}{\emph{National season shocks}} \\")
        cells = []
        for c in CONTRASTS:
            x = tab[(tab.contrast == c) & (tab.spec == spec)]
            if x.empty or ("note" in x.columns and pd.notna(x.note.iloc[0]) if "note" in x.columns else False):
                cells.append("--"); continue
            x = x.iloc[0]
            if spec in ("wild_cluster_boot", "joint_pretrend_wald", "wild_season_webb", "season_permutation"):
                cells.append(f"{x.p:.3f}" if x.p >= 0.001 else "$<$0.001")
            elif spec.startswith("leave"):
                cells.append(f"{x.lo:.3f} to {x.hi:.3f}")
            else:
                cells.append(f"{x.est:.3f} ({x.se:.4f})")
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (RESULTS_DIR / "tables" / "table_inference.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(tab.round(4).to_string())


if __name__ == "__main__":
    main()
