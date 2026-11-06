"""Event-study, difference-in-differences and difference-in-discontinuities estimators."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyfixest as pf


def _tidy(fit, keep: str | None = None) -> pd.DataFrame:
    t = fit.tidy().reset_index().rename(columns={"Coefficient": "term", "Estimate": "est", "Std. Error": "se",
                                                 "Pr(>|t|)": "p", "2.5%": "lo", "97.5%": "hi"})
    if keep:
        t = t[t.term.str.contains(keep, regex=False)]
    return t[["term", "est", "se", "p", "lo", "hi"]].reset_index(drop=True)


def event_study(df: pd.DataFrame, y: str, treat: str, time: str, base: int, fe: str,
                cluster: str, weights: str | None = None) -> pd.DataFrame:
    """y ~ sum_k 1[time=k] x treat (k != base) | fe; returns one row per period, including the base."""
    fe_cols = [c.strip() for c in fe.replace("^", "+").split("+")]
    cols = list(dict.fromkeys([y, treat, time, cluster] + fe_cols + ([weights] if weights else [])))
    d = df[cols].dropna()
    fit = pf.feols(f"{y} ~ i({time}, {treat}, ref={base}) | {fe}", data=d, vcov={"CRV1": cluster},
                   weights=weights)
    t = _tidy(fit)
    t["period"] = t.term.str.extract(r"::(-?\d+)").astype(float).astype("Int64")
    t = pd.concat([t, pd.DataFrame({"term": ["base"], "est": [0.0], "se": [0.0], "p": [np.nan], "lo": [0.0],
                                    "hi": [0.0], "period": [base]})], ignore_index=True)
    t["n_obs"] = fit._N
    return t.sort_values("period").reset_index(drop=True)


def did(df: pd.DataFrame, y: str, treat: str, post: str, fe: str, cluster: str,
        weights: str | None = None, extra: str = "") -> tuple[pd.DataFrame, object]:
    d = df.copy()
    d["_tp"] = d[treat].astype(float) * d[post].astype(float)
    rhs = "_tp" + (f" + {extra}" if extra else "")
    fit = pf.feols(f"{y} ~ {rhs} | {fe}", data=d, vcov={"CRV1": cluster}, weights=weights)
    t = _tidy(fit, keep="_tp")
    t["n_obs"] = fit._N
    return t, fit


def cs_did(df: pd.DataFrame, y: str, group: str, time: str, unit: str | None = None,
           control_group: str = "nevertreated") -> pd.DataFrame:
    """Callaway & Sant'Anna ATT(g,t) aggregated to an event study (csdid package)."""
    from csdid.att_gt import ATTgt
    d = df[[c for c in [y, group, time, unit] if c]].dropna().copy()
    if unit is None:
        d["_id"] = np.arange(len(d))
        unit = "_id"
        panel = False
    else:
        panel = True
    att = ATTgt(yname=y, tname=time, idname=unit, gname=group, data=d, control_group=control_group,
                panel=panel).fit(est_method="dr")
    att.aggte(typec="dynamic")
    r = att.atte
    return pd.DataFrame({"event_time": r["egt"], "est": r["att_egt"], "se": np.asarray(r["se_egt"]).ravel(),
                         "overall_att": float(np.ravel(r["overall_att"])[0]),
                         "overall_se": float(np.ravel(r["overall_se"])[0])})


def diff_in_disc(df: pd.DataFrame, y: str, x: str, cutoff: float, h: float, post: str, cluster: str,
                 fe: str | None = None, weights: str | None = None) -> pd.DataFrame:
    """Local-linear difference-in-discontinuities with a triangular kernel of bandwidth h."""
    d = df.copy()
    d["_r"] = d[x] - cutoff
    d = d[d._r.abs() <= h].copy()
    d["_D"] = (d[x] <= cutoff).astype(float)
    d["_P"] = d[post].astype(float)
    d["_w"] = (1 - d._r.abs() / h) * (d[weights] if weights else 1.0)
    d = d[d._w > 0]
    rhs = "_D + _r + _D:_r + _P:_D + _P:_r + _P:_D:_r"
    rhs += " | " + fe if fe else " + _P"
    fit = pf.feols(f"{y} ~ {rhs}", data=d, vcov={"CRV1": cluster}, weights="_w")
    t = _tidy(fit)
    t = t[t.term.isin(["_D", "_P:_D"])].copy()
    t["term"] = t.term.map({"_D": "disc_pre", "_P:_D": "diff_in_disc"})
    t["n_obs"], t["h"] = fit._N, h
    return t.reset_index(drop=True)


def rd_by_year(df: pd.DataFrame, y: str, x: str, cutoff: float, year: str, cluster: str | None = None,
               h: float | None = None) -> pd.DataFrame:
    """Sharp RD estimate of the jump at the cutoff (treated side x <= cutoff) for each year."""
    from rdrobust import rdrobust
    rows = []
    for yr, g in df.groupby(year):
        g = g[[y, x] + ([cluster] if cluster else [])].dropna()
        kw = dict(y=g[y].values, x=(cutoff - g[x].values), c=0.0)
        if cluster:
            kw["cluster"] = pd.factorize(g[cluster])[0]
        if h:
            kw["h"] = h
        try:
            r = rdrobust(**kw)
            rows.append({"year": yr, "est": float(r.coef.iloc[0, 0]), "se": float(r.se.iloc[2, 0]),
                         "ci_lo": float(r.ci.iloc[2, 0]), "ci_hi": float(r.ci.iloc[2, 1]),
                         "p": float(r.pv.iloc[2, 0]), "h": float(r.bws.iloc[0, 0]),
                         "n_eff": int(sum(r.N_h))})
        except Exception as e:
            rows.append({"year": yr, "error": str(e)[:200]})
    return pd.DataFrame(rows)


def misclass_correct(share_obs, se: float, sp: float):
    """Rogan–Gladen correction p = (p_obs + sp - 1) / (se + sp - 1)."""
    return (np.asarray(share_obs) + sp - 1.0) / (se + sp - 1.0)
