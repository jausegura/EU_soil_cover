#!/usr/bin/env python
"""GAEC 7 difference-in-discontinuities at 10 ha of arable land."""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from capcausal.analysis import estimators as est
from capcausal.analysis.rotation import farm_rotation
from capcausal.utils.paths import DATA_DIR, RESULTS_DIR

YEARS = list(range(2017, 2026))
OUT = RESULTS_DIR / "design_d"
log = logging.getLogger("design_d")


def farm_panel() -> pd.DataFrame:
    f = DATA_DIR / "processed" / "farm_rotation.parquet"
    if f.exists():
        return pd.read_parquet(f)
    d = pd.concat([farm_rotation(t) for t in YEARS], ignore_index=True)
    org = []
    for t in YEARS:
        p = DATA_DIR / "interim" / "dk" / f"organic_{t}.parquet"
        if p.exists():
            o = pd.read_parquet(p).merge(pd.read_parquet(DATA_DIR / "interim" / "dk" / f"parcels_{t}.parquet",
                                                        columns=["uid", "journalnr", "area_ha"]), on="uid")
            o["org_ha"] = o.organic * o.area_ha
            g = o.groupby("journalnr")[["org_ha", "area_ha"]].sum()
            org.append(pd.DataFrame({"journalnr": g.index, "year": t, "organic_share": (g.org_ha / g.area_ha).values}))
    if org:
        d = d.merge(pd.concat(org), on=["journalnr", "year"], how="left")
    else:
        d["organic_share"] = np.nan
    f.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(f)
    return d


def sample(d: pd.DataFrame) -> pd.DataFrame:
    s = d[(d.arable_area_ha > 0) & (d.arable_area_ha <= 40) & d.share_change.notna()].copy()
    s["grassfallow_share"] = (s.grass_rot + s.fallow) / s.arable_area_ha
    s = s[(s.grassfallow_share <= 0.75) & (s.match_rate >= 0.8)]
    s = s[(s.organic_share.fillna(0) < 0.5)]
    s["post"] = (s.year >= 2023).astype(int)
    s["cluster"] = s.cvr.fillna(s.journalnr).astype(str)
    return s


def fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def write_table(dd: pd.DataFrame) -> None:
    from capcausal.utils.paths import ROOT
    labels = {"h3": "Bandwidth 3 ha", "h4": "Bandwidth 4 ha", "h5": "Main (bandwidth 5 ha)", "h6": "Bandwidth 6 ha",
              "h8": "Bandwidth 8 ha", "h5_donut0.5": "Donut ($|r|>0.5$ ha)", "h5_incl2026": "Adding claim year 2026", "placebo_year2020": "Placebo year 2020 (2017--2022)",
              "placebo_cut7": "Placebo cutoff 7 ha", "placebo_cut8": "Placebo cutoff 8 ha", "placebo_cut12": "Placebo cutoff 12 ha",
              "placebo_cut13": "Placebo cutoff 13 ha", "placebo_cut15": "Placebo cutoff 15 ha"}
    t = dd[dd.term == "diff_in_disc"]
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"Specification & Share changed & Meets 35\% rule & Same category 4 years \\", r"\midrule"]
    for spec, lab in labels.items():
        cells = []
        for y in ("share_change", "meets35", "share_4yr"):
            r = t[(t.spec == spec) & (t.outcome == y)]
            cells.append("--" if r.empty else f"{r.est.iloc[0]:.3f} ({r.se.iloc[0]:.3f})")
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (RESULTS_DIR / "tables" / "table_d.tex").write_text("\n".join(lines) + "\n")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)
    d = farm_panel()
    s = sample(d)
    s.groupby("year").agg(n=("journalnr", "size"), share_change=("share_change", "mean"),
                          meets35=("meets35", "mean")).to_csv(OUT / "descriptives_by_year.csv")
    ys = ["share_change", "meets35", "share_4yr"]
    rd = pd.concat([est.rd_by_year(s, y, "arable_area_ha", 10.0, "year", cluster="cluster").assign(outcome=y)
                    for y in ys], ignore_index=True)
    rd.to_csv(OUT / "rd_by_year.csv", index=False)

    f26 = DATA_DIR / "processed" / "farm_rotation_2026.parquet"
    if not f26.exists():
        r26 = farm_rotation(2026)
        r26["organic_share"] = np.nan
        r26.to_parquet(f26)
    s26 = sample(pd.read_parquet(f26))
    rows = []
    for y in ys:
        base = s if y != "share_4yr" else s[s.year >= 2019]
        for h in (3.0, 4.0, 5.0, 6.0, 8.0):
            r = est.diff_in_disc(base, y, "arable_area_ha", 10.0, h, "post", "cluster", fe="year")
            rows.append(r.assign(outcome=y, spec=f"h{h:g}"))
        donut = base[(base.arable_area_ha - 10).abs() > 0.5]
        rows.append(est.diff_in_disc(donut, y, "arable_area_ha", 10.0, 5.0, "post", "cluster", fe="year")
                    .assign(outcome=y, spec="h5_donut0.5"))
        for c in (7.0, 8.0, 12.0, 13.0, 15.0):
            side = base[(base.arable_area_ha <= 10) if c < 10 else (base.arable_area_ha > 10)]
            rows.append(est.diff_in_disc(side, y, "arable_area_ha", c, 2.5, "post", "cluster", fe="year")
                        .assign(outcome=y, spec=f"placebo_cut{c:g}"))
        if y != "share_4yr":
            ext = pd.concat([base, s26]) if len(s26) else base
            rows.append(est.diff_in_disc(ext, y, "arable_area_ha", 10.0, 5.0, "post", "cluster", fe="year")
                        .assign(outcome=y, spec="h5_incl2026"))
        pre = base[base.year <= 2022].copy()
        pre["fake_post"] = (pre.year >= 2020).astype(int)
        rows.append(est.diff_in_disc(pre, y, "arable_area_ha", 10.0, 5.0, "fake_post", "cluster", fe="year")
                    .assign(outcome=y, spec="placebo_year2020"))
    dd = pd.concat(rows, ignore_index=True)
    dd.to_csv(OUT / "diff_in_disc.csv", index=False)

    from rddensity import rddensity
    dens = []
    for t, g in s.groupby("year"):
        x = g.arable_area_ha.values - 10.0
        x = x[np.abs(x) <= 8]
        try:
            r = rddensity(X=-x, c=0.0)
            dens.append({"year": t, "T": float(r.hat["diff"] / r.sd_jk["diff"]) if hasattr(r, "sd_jk") else np.nan,
                         "p": float(r.test["p_jk"]) if hasattr(r, "test") else np.nan, "n": len(x)})
        except Exception as e:
            dens.append({"year": t, "error": str(e)[:200], "n": len(x)})
    pd.DataFrame(dens).to_csv(OUT / "density_tests.csv", index=False)

    main_r = dd[(dd.spec == "h5") & (dd.term == "diff_in_disc")].set_index("outcome")
    pn = {}
    for y, tag in (("share_change", "Chg"), ("meets35", "Meets"), ("share_4yr", "Fouryr")):
        r = main_r.loc[y]
        pn[f"DdDisc{tag}"] = f"{r.est:.3f}"
        pn[f"DdDisc{tag}Lo"] = f"{r.lo:.3f}"
        pn[f"DdDisc{tag}Hi"] = f"{r.hi:.3f}"
        pn[f"DdDisc{tag}P"] = fmt_p(r.p)
    sc = dd[(dd.outcome == "share_change")]
    pre_d = sc[(sc.spec == "h5") & (sc.term == "disc_pre")].iloc[0]
    pn["DdDiscPreChg"], pn["DdDiscPreChgP"] = f"{pre_d.est:.3f}", fmt_p(pre_d.p)
    bw = sc[sc.spec.str.match(r"^h\d+$") & (sc.term == "diff_in_disc")]
    pn["DdBwMin"], pn["DdBwMax"] = f"{bw.est.min():.3f}", f"{bw.est.max():.3f}"
    pn["DdInclTwentySixChg"] = f"{sc[(sc.spec == 'h5_incl2026') & (sc.term == 'diff_in_disc')].est.iloc[0]:.3f}"
    pn["DdDonutChg"] = f"{sc[(sc.spec == 'h5_donut0.5') & (sc.term == 'diff_in_disc')].est.iloc[0]:.3f}"
    pl = sc[sc.spec.str.startswith("placebo_cut") & (sc.term == "diff_in_disc")]
    pn["DdPlaceboCutMaxAbs"] = f"{pl.est.abs().max():.3f}"
    pn["DdPlaceboCutMinP"] = fmt_p(pl.p.min())
    py = sc[(sc.spec == "placebo_year2020") & (sc.term == "diff_in_disc")].iloc[0]
    pn["DdPlaceboYearChg"], pn["DdPlaceboYearChgP"] = f"{py.est:.3f}", fmt_p(py.p)
    rdc = rd[(rd.outcome == "share_change") & rd.est.notna()]
    pre_rd, post_rd = rdc[rdc.year <= 2022], rdc[rdc.year >= 2023]
    pn["DdRdPreMin"], pn["DdRdPreMax"] = f"{pre_rd.est.min():.3f}", f"{pre_rd.est.max():.3f}"
    pn["DdRdPostMin"], pn["DdRdPostMax"] = f"{post_rd.est.min():.3f}", f"{post_rd.est.max():.3f}"
    dens_df = pd.DataFrame(dens)
    if "p" in dens_df:
        pn["DdDensPMax"] = fmt_p(dens_df.p.max())
        pn["DdDensTMin"], pn["DdDensTMax"] = f"{dens_df['T'].min():.1f}", f"{dens_df['T'].max():.1f}"
    yr = s.groupby("year").share_change.mean()
    pn["DdShareChgPre"] = f"{yr[yr.index <= 2022].mean():.2f}"
    pn["DdShareChgPost"] = f"{yr[yr.index >= 2023].mean():.2f}"
    pn["DdNfarmYears"] = f"{len(s):,}"
    pn["DdNmain"] = f"{int(main_r.loc['share_change', 'n_obs']):,}"
    (OUT / "summary.json").write_text(json.dumps({"paper_numbers": pn}, indent=2))
    write_table(dd)
    print(main_r[["est", "se", "p", "lo", "hi", "n_obs"]])


if __name__ == "__main__":
    main()
