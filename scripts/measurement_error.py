#!/usr/bin/env python
"""Rescaling under assumed common misclassification."""
import json

import pandas as pd

from capcausal.utils.paths import RESULTS_DIR, ROOT

OUT = RESULTS_DIR / "measurement_error"
SE_GRID = (0.5, 0.6, 0.7, 0.8, 0.9)


def obs_bounds():
    r = pd.read_csv(RESULTS_DIR / "validation_aligned" / "rates.csv").set_index("subset").loc["all"]
    return 1 - r.obs_bare_N3, 1 - r.obs_bare_N1


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sp_lo, sp_hi = obs_bounds()
    effects = {}
    inf = pd.read_csv(RESULTS_DIR / "inference_robust" / "table.csv")
    for lab, key in (("A1", "A"), ("A2", "Atwo")):
        r = inf[(inf.contrast == lab) & (inf.spec == "crv_tile_season")].iloc[0]
        effects[key] = (r.est, r.lo, r.hi)
    c = pd.read_csv(RESULTS_DIR / "design_c" / "diff_in_disc.csv")
    r = c[(c.spec == "pre2023_h5") & (c.term == "diff_in_disc") & (c.outcome == "share_bare_augsep")].iloc[0]
    effects["C"] = (r.est, r.lo, r.hi)
    rows = []
    for se in SE_GRID:
        for sp_name, sp in (("lower", sp_lo), ("upper", sp_hi)):
            k = se + sp - 1
            for dname, (e, lo, hi) in effects.items():
                rows.append({"design": dname, "Se": se, "Sp_bound": sp_name, "Sp": sp, "multiplier": 1 / k,
                             "est_corr": e / k, "lo_corr": lo / k, "hi_corr": hi / k})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "table.csv", index=False)
    desc = pd.read_csv(RESULTS_DIR / "design_a" / "descriptives.csv")
    pre = desc[(desc.conventional == 1) & (desc.season <= 2022)]
    wavg = lambda x: (x.share_bare * x.n).sum() / x.n.sum()
    gap = wavg(pre[pre.succ == "autumn"]) - wavg(pre[pre.succ == "spring"])
    hiC = effects["C"][2]
    be = {name: hiC / E + 1 - sp_hi for name, E in (("full", gap), ("half", gap / 2))}
    pn = {"MeSpLo": f"{sp_lo:.3f}", "MeSpHi": f"{sp_hi:.3f}",
          "MeMultMin": f"{t.multiplier.min():.2f}", "MeMultMax": f"{t.multiplier.max():.2f}",
          "MeFactorMin": f"{t.multiplier.min():.2f}", "MeFactorMax": f"{t.multiplier.max():.2f}",
          "MePrespec": f"{gap:.3f}", "MePrespecHalf": f"{gap / 2:.3f}", "MeCHiObs": f"{hiC:.3f}",
          "MeCBreakEvenFull": f"{be['full']:.2f}", "MeCBreakEvenHalf": f"{be['half']:.2f}",
          "MeCHiCorrMin": f"{t[t.design == 'C'].hi_corr.min():.3f}", "MeCHiCorrMax": f"{t[t.design == 'C'].hi_corr.max():.3f}"}
    for dname in effects:
        g = t[t.design == dname]
        pn[f"Me{dname}CorrMin"] = f"{g.est_corr.abs().min() * (1 if g.est_corr.mean() > 0 else -1):.3f}"
        pn[f"Me{dname}CorrMax"] = f"{g.est_corr.abs().max() * (1 if g.est_corr.mean() > 0 else -1):.3f}"
    (OUT / "summary.json").write_text(json.dumps({"paper_numbers": pn}, indent=2))
    lines = [r"\begin{tabular}{ccccc}", r"\toprule",
             "Se (assumed) & Sp & " + " & ".join({"A": "A1 (spring successor)", "Atwo": "A2 (autumn successor)", "C": "C (2024 exemption)"}[d] for d in effects) + r" \\", r"\midrule"]
    lines.append("\\multicolumn{2}{l}{Not rescaled} & " + " & ".join(f"{e:.3f} [{lo:.3f}, {hi:.3f}]" for e, lo, hi in effects.values()) + r" \\")
    lines.append(r"\midrule")
    for (se, spn), g in t.groupby(["Se", "Sp_bound"], sort=False):
        cells = [f"{g[g.design == d].est_corr.iloc[0]:.3f} [{g[g.design == d].lo_corr.iloc[0]:.3f}, "
                 f"{g[g.design == d].hi_corr.iloc[0]:.3f}]" for d in effects]
        lines.append(f"{se:.1f} & {g.Sp.iloc[0]:.3f} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (RESULTS_DIR / "tables" / "table_me.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(t.round(3).to_string()); print(pn)


if __name__ == "__main__":
    main()
