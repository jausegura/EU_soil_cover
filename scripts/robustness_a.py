#!/usr/bin/env python
"""Sample-composition and indicator-definition robustness of the GAEC 6 contrasts."""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pyfixest as pf

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import analyze_glm6 as A

from capcausal.utils.paths import DATA_DIR, PROCESSED_DIR, RESULTS_DIR, ROOT

OUT = RESULTS_DIR / "robustness_a"
DK = DATA_DIR / "interim" / "dk"
SEASONS = list(range(2019, 2026))
FE = "tile^season + crop_group^treat + glm6_zone^treat"


def matching_stats():
    rows, links = [], {}
    for s in range(2019, 2025):
        a = pd.read_parquet(DK / f"parcels_{s}.parquet", columns=["uid", "area_ha", "agricultural"])
        b = pd.read_parquet(DK / f"parcels_{s + 1}.parquet", columns=["uid", "area_ha"]).rename(columns={"uid": "uid_b", "area_ha": "area_b"})
        m = pd.read_parquet(DK / f"match_{s}_{s + 1}.parquet").rename(columns={"uid_a": "uid"})
        m = m.sort_values("overlap", ascending=False).drop_duplicates("uid")
        x = a[a.agricultural].merge(m, on="uid", how="left").merge(b, on="uid_b", how="left")
        ratio = x.area_b / x.area_ha
        n_in = x.groupby("uid_b").uid.transform("size")
        stable = (x.overlap >= 0.9) & ratio.between(0.9, 1.1) & (n_in == 1)
        merged = x.uid_b.notna() & (n_in > 1)
        partial = x.uid_b.notna() & ~stable & ~merged
        rows.append({"pair": f"{s}-{s + 1}", "n": len(x), "one_to_one": stable.mean(), "merged": merged.mean(),
                     "boundary_change": partial.mean(), "unmatched": x.uid_b.isna().mean(),
                     "area_one_to_one": x.area_ha[stable].sum() / x.area_ha.sum()})
        links[s] = x.loc[stable, ["uid", "uid_b"]]
    return pd.DataFrame(rows), links


def stable_fields(links) -> pd.DataFrame:
    """(season, uid, field) for parcels on chains linked one-to-one in every year 2019-2025."""
    chain = links[2019].rename(columns={"uid": "u2019", "uid_b": "u2020"})
    for s in range(2020, 2025):
        chain = chain.merge(links[s].rename(columns={"uid": f"u{s}", "uid_b": f"u{s + 1}"}), on=f"u{s}")
    out = [pd.DataFrame({"season": s, "uid": chain[f"u{s}"].values, "field": chain.u2019.values}) for s in SEASONS]
    return pd.concat(out, ignore_index=True)


def load_alt():
    return pd.concat([pd.read_parquet(PROCESSED_DIR / f"alt_augsep_{s}.parquet") for s in SEASONS], ignore_index=True)


def prep(s, label):
    if label == "A0":
        d = s[s.conventional == 1].copy()
        d["treat"] = (d.succ == "autumn").astype(int)
    else:
        d = s[s.succ == ("spring" if label == "A1" else "autumn")].copy()
        d["treat"] = d.conventional.astype(int)
    d["ts"] = d.tile.astype(str) + "_" + d.season.astype(str)
    d["_tp"] = d.treat * d.post
    d["cg_treat"] = d.crop_group.astype(str) + "_" + d.treat.astype(str)
    return d


def did(d, y="share_bare_augsep", fe=FE, weights=None, extra=""):
    d = d.dropna(subset=[y])
    try:
        f = pf.feols(f"{y} ~ _tp{extra} | {fe}", data=d, vcov={"CRV1": "ts"}, weights=weights, fixef_maxiter=20000, fixef_tol=1e-6)
    except Exception as e:
        print("failed:", y, fe, str(e)[:100], flush=True)
        return {"est": np.nan, "se": np.nan, "lo": np.nan, "hi": np.nan, "p": np.nan, "n": len(d)}
    t = f.tidy().loc["_tp"]
    return {"est": t["Estimate"], "se": t["Std. Error"], "lo": t["2.5%"], "hi": t["97.5%"], "p": t["Pr(>|t|)"], "n": int(f._N)}


def reweight(d):
    """Weights that give every season the pre-period (2019-2022) distribution of treat x crop group x soil zone x tile cells."""
    cell = ["treat", "crop_group", "glm6_zone", "tile"]
    pre = d[d.season <= 2022].groupby(cell).size() / (d.season <= 2022).sum()
    cur = d.groupby(["season"] + cell).size() / d.groupby("season").size()
    w = (pre.rename("p_pre").reset_index().merge(cur.rename("p_s").reset_index(), on=cell))
    w["rw"] = w.p_pre / w.p_s
    return d.merge(w[["season"] + cell + ["rw"]], on=["season"] + cell, how="inner")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mt, links = matching_stats()
    mt.to_csv(OUT / "matching.csv", index=False)
    sf = stable_fields(links)
    d_all = A.load(SEASONS)
    s = A.sample_a(d_all, 3)
    s = s.merge(sf, on=["season", "uid"], how="left")
    s["stable"] = s.field.notna()
    alt = load_alt()
    s = s.merge(alt, on=["season", "uid"], how="left")
    comp = []
    for per, g in s.groupby(s.season >= 2023):
        comp.append({"period": "2023-2025" if per else "2019-2022", "n_per_season": len(g) / g.season.nunique(),
                     "sand": (g.glm6_zone == "sand_1feb").mean(), "loam": (g.glm6_zone == "loam_20oct").mean(),
                     "organic": 1 - g.conventional.mean(), "autumn_succ": (g.succ == "autumn").mean(),
                     "winter_cereal": (g.crop_group == "cereal_winter").mean(), "spring_cereal": (g.crop_group == "cereal_spring").mean(),
                     "rapeseed": (g.crop_group == "rapeseed_winter").mean(), "mean_area": g.area_ha.mean(),
                     "mean_obs": g.n_augsep.mean(), "stable_field": g.stable.mean()})
    comp = pd.DataFrame(comp)
    comp.to_csv(OUT / "composition.csv", index=False)
    specs = {
        "main": dict(),
        "area_weighted": dict(weights="area_ha"),
        "stable_fields": dict(sub=lambda d: d[d.stable]),
        "stable_fields_field_fe": dict(sub=lambda d: d[d.stable], fe="field", extra=" + C(ts) + C(cg_treat)"),
        "aug_and_sep_obs": dict(sub=lambda d: d[(d.n_aug >= 1) & (d.n_sep >= 1)]),
        "successor_overlap_90": dict(sub=lambda d: d[d.overlap_next >= 0.9]),
        "reweighted_pre_composition": dict(sub=reweight, weights="rw"),
        "ndti_0.06": dict(y="share_tau06"), "ndti_0.14": dict(y="share_tau14"),
        "ndvi_only": dict(y="share_ndvi"), "pixel_majority_30": dict(y="share_maj30"), "pixel_majority_70": dict(y="share_maj70"),
        "valid_pixels_30": dict(y="share_vf30"), "valid_pixels_70": dict(y="share_vf70"), "valid_pixels_90": dict(y="share_vf90"),
    }
    rows = []
    for label in ("A0", "A1", "A2"):
        base = prep(s, label)
        for name, sp in specs.items():
            d = sp.get("sub", lambda x: x)(base)
            r = did(d, y=sp.get("y", "share_bare_augsep"), fe=sp.get("fe", FE), weights=sp.get("weights"), extra=sp.get("extra", ""))
            rows.append({"contrast": label, "spec": name, **r})
            print(label, name, round(r["est"], 4), round(r["se"], 4), r["n"], flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "did.csv", index=False)
    print(mt.round(3).to_string()); print(comp.round(3).to_string()); print(t.round(4).to_string())
    pn = {"MtOneToOneMin": f"{mt.one_to_one.min() * 100:.1f}", "MtOneToOneMax": f"{mt.one_to_one.max() * 100:.1f}",
          "MtMergedMax": f"{mt.merged.max() * 100:.1f}", "MtBoundaryMax": f"{mt.boundary_change.max() * 100:.1f}",
          "MtUnmatchedMax": f"{mt.unmatched.max() * 100:.1f}",
          "StableShare": f"{s.stable.mean() * 100:.1f}", "StableFields": f"{int(sf.field.nunique()):,}",
          "CompSandPre": f"{comp.sand.iloc[0]:.3f}", "CompSandPost": f"{comp.sand.iloc[1]:.3f}",
          "CompAutPre": f"{comp.autumn_succ.iloc[0]:.3f}", "CompAutPost": f"{comp.autumn_succ.iloc[1]:.3f}",
          "CompOrgPre": f"{comp.organic.iloc[0] * 100:.1f}", "CompOrgPost": f"{comp.organic.iloc[1] * 100:.1f}"}
    for label, tag in (("A0", "Zero"), ("A1", "One"), ("A2", "Two")):
        g = t[t.contrast == label].set_index("spec")
        samp = g.loc[["area_weighted", "stable_fields", "stable_fields_field_fe", "aug_and_sep_obs", "successor_overlap_90", "reweighted_pre_composition"]]
        ind = g.loc[[k for k in specs if k.startswith(("ndti", "pixel", "valid"))]]
        pn[f"Rb{tag}SampMin"], pn[f"Rb{tag}SampMax"] = f"{samp.est.min():.3f}", f"{samp.est.max():.3f}"
        pn[f"Rb{tag}IndMin"], pn[f"Rb{tag}IndMax"] = f"{ind.est.min():.3f}", f"{ind.est.max():.3f}"
        pn[f"Rb{tag}Stable"] = f"{g.loc['stable_fields', 'est']:.3f}"
        pn[f"Rb{tag}StableFe"] = f"{g.loc['stable_fields_field_fe', 'est']:.3f}"
        pn[f"Rb{tag}Ndvi"] = f"{g.loc['ndvi_only', 'est']:.3f}"
    (OUT / "summary.json").write_text(json.dumps({"paper_numbers": pn}, indent=2))
    AG = RESULTS_DIR / "tables"
    L = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r"Years & Parcels & One-to-one & Boundary change & Merged & Unmatched & Area one-to-one \\", r"\midrule"]
    for r in mt.itertuples():
        L.append(f"{r.pair} & {r.n:,} & {r.one_to_one:.3f} & {r.boundary_change:.3f} & {r.merged:.3f} & {r.unmatched:.3f} & {r.area_one_to_one:.3f} \\\\".replace(",", "{,}", 1))
    L += [r"\bottomrule", r"\end{tabular}"]
    (AG / "supp_matching.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    lab = {"n_per_season": "Parcels per season", "sand": "Sandy soil (JB 1--4)", "loam": "Loamy soil (JB 5, 6, 10, 11)",
           "organic": "Organic", "autumn_succ": "Autumn-sown successor", "winter_cereal": "Winter cereal harvested",
           "spring_cereal": "Spring cereal harvested", "rapeseed": "Winter rapeseed harvested", "mean_area": "Mean parcel area (ha)",
           "mean_obs": "Valid observations Aug--Sep", "stable_field": "On a stable field 2019--2025"}
    L = [r"\begin{tabular}{lrr}", r"\toprule", r" & 2019--2022 & 2023--2025 \\", r"\midrule"]
    for k, v in lab.items():
        a, b = comp[k].iloc[0], comp[k].iloc[1]
        fmt = (lambda x: f"{x:,.0f}".replace(",", "{,}")) if k == "n_per_season" else ((lambda x: f"{x:.2f}") if k in ("mean_area", "mean_obs") else (lambda x: f"{x:.3f}"))
        L.append(f"{v} & {fmt(a)} & {fmt(b)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (AG / "supp_composition.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    slab = {"main": "Main specification", "area_weighted": "Weighted by parcel area", "stable_fields": "Stable fields 2019--2025",
            "stable_fields_field_fe": "Stable fields, field fixed effects", "aug_and_sep_obs": "Observed in August and in September",
            "successor_overlap_90": "Successor link $\\ge$90\\% overlap", "reweighted_pre_composition": "Reweighted to 2019--2022 composition",
            "ndti_0.06": "NDTI boundary 0.06", "ndti_0.14": "NDTI boundary 0.14", "ndvi_only": "NDVI only (no NDTI)",
            "pixel_majority_30": "30\\% of pixels bare", "pixel_majority_70": "70\\% of pixels bare",
            "valid_pixels_30": "Observation valid with $\\ge$30\\% clear pixels", "valid_pixels_70": "Observation valid with $\\ge$70\\% clear pixels",
            "valid_pixels_90": "Observation valid with $\\ge$90\\% clear pixels"}
    L = [r"\begin{tabular}{llll}", r"\toprule", r"Specification & A0 & A1 & A2 \\", r"\midrule"]
    for k, v in slab.items():
        cells = []
        for c in ("A0", "A1", "A2"):
            r = t[(t.contrast == c) & (t.spec == k)].iloc[0]
            cells.append(f"{r.est:.3f} ({r.se:.3f})")
        L.append(f"{v} & " + " & ".join(cells) + r" \\")
        if k in ("reweighted_pre_composition",):
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}"]
    (AG / "supp_robust_a.tex").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
