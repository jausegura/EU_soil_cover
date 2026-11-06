#!/usr/bin/env python
"""Implied effects under assumed differential misclassification."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from capcausal.utils.paths import RESULTS_DIR, ROOT

OUT = RESULTS_DIR / "measurement_error_diff"


def invert(pstar, se, sp):
    return (pstar - (1 - sp)) / (se + sp - 1)


def cells_a(desc, succ, treat_sys):
    """Cell means for a conventional (1) vs organic (0) contrast for one successor."""
    d = desc[desc.succ == succ]
    pre = d[d.season <= 2022].groupby("conventional").apply(lambda x: np.average(x.share_bare, weights=x.n))
    post = d[d.season >= 2023].groupby("conventional").apply(lambda x: np.average(x.share_bare, weights=x.n))
    return {"t_pre": pre[1], "t_post": post[1], "c_pre": pre[0], "c_post": post[0]}


def cells_a0(desc):
    pre = desc[(desc.conventional == 1) & (desc.season <= 2022)].groupby("succ").apply(lambda x: np.average(x.share_bare, weights=x.n))
    post = desc[(desc.conventional == 1) & (desc.season >= 2023)].groupby("succ").apply(lambda x: np.average(x.share_bare, weights=x.n))
    return {"t_pre": pre["autumn"], "t_post": post["autumn"], "c_pre": pre["spring"], "c_post": post["spring"]}


def did(c, se_t, se_c, sp_pre, sp_post):
    tt = invert(c["t_post"], se_t, sp_post) - invert(c["t_pre"], se_t, sp_pre)
    cc = invert(c["c_post"], se_c, sp_post) - invert(c["c_pre"], se_c, sp_pre)
    return tt - cc


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    va = pd.read_csv(RESULTS_DIR / "validation_aligned" / "rates.csv").set_index("subset")
    sp_lo, sp_hi = 1 - va.loc["all", "obs_bare_N3"], 1 - va.loc["all", "obs_bare_N1"]
    seas = va[va.index.str.startswith("season_")].copy()
    seas["season"] = seas.index.str[7:].astype(int)
    fp = lambda x: (x.obs_bare_N3 * x.nobs_N3).sum() / x.nobs_N3.sum()
    fp_all = fp(seas)
    fp_pre_f, fp_post_f = fp(seas[seas.season <= 2022]) / fp_all, fp(seas[seas.season >= 2023]) / fp_all
    se_ratio = va.loc["system_organic", "obs_bare_P"] / va.loc["system_conventional", "obs_bare_P"]
    desc = pd.read_csv(RESULTS_DIR / "design_a" / "descriptives.csv")
    contrasts = {"A0": cells_a0(desc), "A1": cells_a(desc, "spring", 1), "A2": cells_a(desc, "autumn", 1)}
    rows = []
    for name, c in contrasts.items():
        obs = did(c, 1, 1, 1, 1)
        for se in (0.5, 0.7, 0.9):
            for sp_name, sp in (("lower", sp_lo), ("upper", sp_hi)):
                for scen in ("nondiff", "sp_by_period", "se_by_group", "both"):
                    sp_pre = 1 - (1 - sp) * fp_pre_f if scen in ("sp_by_period", "both") else sp
                    sp_post = 1 - (1 - sp) * fp_post_f if scen in ("sp_by_period", "both") else sp
                    sp_pre, sp_post = min(sp_pre, 0.9999), min(sp_post, 0.9999)
                    se_c = se * se_ratio if (scen in ("se_by_group", "both") and name != "A0") else se
                    rows.append({"contrast": name, "Se": se, "Sp_bound": sp_name, "scenario": scen, "observed_did": obs,
                                 "implied_did": did(c, se, se_c, sp_pre, sp_post)})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "scenarios.csv", index=False)
    pn = {"MdFpPreFactor": f"{fp_pre_f:.2f}", "MdFpPostFactor": f"{fp_post_f:.2f}", "MdSeRatioOrg": f"{se_ratio:.2f}"}
    for name in contrasts:
        g = t[t.contrast == name]
        tag = {"A0": "Zero", "A1": "One", "A2": "Two"}[name]
        pn[f"Md{tag}Obs"] = f"{g.observed_did.iloc[0]:+.3f}"
        pn[f"Md{tag}NondiffMin"] = f"{g[g.scenario == 'nondiff'].implied_did.min():+.3f}"
        pn[f"Md{tag}NondiffMax"] = f"{g[g.scenario == 'nondiff'].implied_did.max():+.3f}"
        pn[f"Md{tag}DiffMin"] = f"{g[g.scenario != 'nondiff'].implied_did.min():+.3f}"
        pn[f"Md{tag}DiffMax"] = f"{g[g.scenario != 'nondiff'].implied_did.max():+.3f}"
        sign_flip = (np.sign(g.implied_did) != np.sign(g.observed_did.iloc[0])).any()
        pn[f"Md{tag}SignFlip"] = "yes" if sign_flip else "no"
    (OUT / "summary.json").write_text(json.dumps({"paper_numbers": pn}, indent=2))
    lab = {"nondiff": "common error", "sp_by_period": "Sp differs pre/post", "se_by_group": "Se differs by system", "both": "both differ"}
    lines = [r"\begin{tabular}{llccc}", r"\toprule",
             r"Contrast & Error scenario & Se 0.5 & Se 0.7 & Se 0.9 \\", r"\midrule"]
    for c in contrasts:
        g = t[t.contrast == c]
        lines.append(f"{c} & observed (not rescaled) & \\multicolumn{{3}}{{c}}{{{g.observed_did.iloc[0]:+.3f}}} \\\\")
        for sc in ("nondiff", "sp_by_period", "se_by_group", "both"):
            if c == "A0" and sc == "se_by_group":
                continue
            cells = []
            for se in (0.5, 0.7, 0.9):
                x = g[(g.scenario == sc) & (g.Se == se)].set_index("Sp_bound").implied_did
                cells.append(f"{x['lower']:+.3f} / {x['upper']:+.3f}")
            lines.append(f" & {lab[sc]} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule" if c != list(contrasts)[-1] else r"\bottomrule")
    lines += [r"\end{tabular}"]
    (RESULTS_DIR / "tables" / "table_me_diff.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(t.round(3).to_string())
    print(pn)


if __name__ == "__main__":
    main()
