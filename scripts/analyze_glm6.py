#!/usr/bin/env python
"""GAEC 6 event studies and the 2024 exemption difference-in-discontinuities."""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

from capcausal.analysis import estimators as est
from capcausal.utils.paths import DATA_DIR, PROCESSED_DIR, RESULTS_DIR

log = logging.getLogger("glm6")
SPRING_NEXT = {"cereal_spring", "maize_silage", "maize_grain", "legume", "beet", "potatoes", "rapeseed_spring"}
HARVESTED = {"cereal_winter", "cereal_spring", "rapeseed_winter", "legume", "seed_grass_or_herb"}


def _try(fn, *args, **kw):
    """Run a secondary specification; log and skip it if it cannot be estimated."""
    try:
        return fn(*args, **kw)
    except Exception as e:
        log.warning("specification skipped: %s", str(e).strip().splitlines()[0][:150])
        return None


def _did(*args, **kw):
    r = _try(est.did, *args, **kw)
    return (r[0] if r else pd.DataFrame()), None


def _dd(*args, **kw):
    r = _try(est.diff_in_disc, *args, **kw)
    return r if r is not None else pd.DataFrame()


COLS = ["uid", "season", "cvr", "journalnr", "crop_group", "crop_group_next", "autumn_sown_next", "organic",
        "glm6_zone", "overlap_next", "tile", "x", "y", "area_ha", "agri_area_ha", "arable_area_ha", "n_augsep",
        "share_bare_augsep", "any_bare_augsep", "bare_spell_days", "share_bare_oct2to20"]


def load(seasons) -> pd.DataFrame:
    parts = []
    for s in seasons:
        x = pd.read_parquet(PROCESSED_DIR / f"parcel_season_{s}.parquet", columns=COLS)
        for c in x.select_dtypes("float64").columns:
            if c not in ("x", "y"):
                x[c] = x[c].astype("float32")
        for c in ("crop_group", "crop_group_next", "glm6_zone", "tile"):
            x[c] = x[c].astype("category")
        parts.append(x)
    d = pd.concat(parts, ignore_index=True)
    for c in ("crop_group", "crop_group_next", "glm6_zone", "tile"):
        d[c] = d[c].astype(str)
    d["cluster"] = d.cvr.fillna(d.journalnr).astype(str)
    d["organic"] = d.get("organic", pd.Series(0, index=d.index)).fillna(0)
    return d


def add_weather(d: pd.DataFrame) -> pd.DataFrame:
    """Aug-Sep precipitation (mm, ERA5 via Open-Meteo) of the nearest 0.5-degree cell, standardised."""
    f = DATA_DIR / "interim" / "weather" / "era5_dk_daily.parquet"
    if not f.exists():
        d["precip_z"] = np.nan
        return d
    from pyproj import Transformer
    w = pd.read_parquet(f)
    w = w[(w.date.dt.month.isin([8, 9]))]
    w["season"] = w.date.dt.year
    w = w.groupby(["lat", "lon", "season"]).precip_mm.sum().rename("precip_augsep").reset_index()
    lon, lat = Transformer.from_crs(25832, 4326, always_xy=True).transform(d.x.values, d.y.values)
    d = d.copy()
    d["lat"] = (np.round(np.asarray(lat) * 2) / 2).clip(w.lat.min(), w.lat.max())
    d["lon"] = (np.round(np.asarray(lon) * 2) / 2).clip(w.lon.min(), w.lon.max())
    d = d.merge(w, on=["lat", "lon", "season"], how="left")
    d["precip_z"] = (d.precip_augsep - d.precip_augsep.mean()) / d.precip_augsep.std()
    return d


def sample_a(d: pd.DataFrame, min_obs: int) -> pd.DataFrame:
    """Harvested arable parcel-seasons with a known successor (autumn- or spring-sown), both organic and conventional; `succ` in {autumn, spring}; `conventional` = 1 - organic."""
    s = d[d.crop_group.isin(HARVESTED) & (d.overlap_next >= 0.5) & (d.n_augsep >= min_obs)
          & (d.glm6_zone != "unknown")].copy()
    aut = s.autumn_sown_next.fillna(False).astype(bool)
    s["succ"] = np.where(aut, "autumn", np.where(s.crop_group_next.isin(SPRING_NEXT), "spring", "other"))
    s = s[s.succ != "other"].copy()
    s["conventional"] = (s.organic == 0).astype(int)
    s["post"] = (s.season >= 2023).astype(int)
    s["long_spell"] = (s.bare_spell_days >= 28).astype(int)
    return add_weather(s)


def _contrast(s: pd.DataFrame, treat: str, base: int, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Event study and DiD (with robustness) of the bare-soil outcomes for one treated/control contrast."""
    s = s.copy()
    s["treat"] = s[treat].astype(int)
    fe = "tile^season + crop_group^treat + glm6_zone^treat"
    ys = ["share_bare_augsep", "any_bare_augsep", "long_spell"]
    es = pd.concat([est.event_study(s, y, "treat", "season", base, fe, "cluster").assign(outcome=y) for y in ys])
    rows = []
    for y in ys:
        t, _ = est.did(s, y, "treat", "post", fe, "cluster")
        rows.append(t.assign(outcome=y, spec="main"))
        for name, sub in (("excl2022", s[s.season != 2022]),
                          ("excl2019_2021_biomass_pilot", s[~s.season.isin([2019, 2020, 2021])]),
                          ("high_obs_density", s[s.n_augsep >= s.n_augsep.median()])):
            t, _ = _did(sub, y, "treat", "post", fe, "cluster")
            rows.append(t.assign(outcome=y, spec=name))
        t, _ = _did(s, y, "treat", "post", fe, "cluster", extra="n_augsep")
        rows.append(t.assign(outcome=y, spec="ctrl_nobs"))
        t, _ = _did(s.dropna(subset=["precip_z"]), y, "treat", "post", fe, "cluster", extra="precip_z + precip_z:treat")
        rows.append(t.assign(outcome=y, spec="ctrl_weather"))
        pre = s[s.season <= 2022].copy()
        pre["fake_post"] = (pre.season >= 2021).astype(int)
        t, _ = _did(pre, y, "treat", "fake_post", fe, "cluster")
        rows.append(t.assign(outcome=y, spec="placebo_2021"))
        for z, g in s.groupby("glm6_zone"):
            t, _ = _did(g, y, "treat", "post", "tile^season + crop_group^treat", "cluster")
            rows.append(t.assign(outcome=y, spec=f"zone_{z}"))
    return es.assign(contrast=label), pd.concat(rows, ignore_index=True).assign(contrast=label)


def design_a(d: pd.DataFrame, base: int, min_obs: int, out) -> dict:
    s = sample_a(d, min_obs)
    out.mkdir(parents=True, exist_ok=True)
    s.groupby(["succ", "conventional", "season"]).agg(
        n=("uid", "size"), share_bare=("share_bare_augsep", "mean"), any_bare=("any_bare_augsep", "mean"),
        long_spell=("long_spell", "mean"), n_obs=("n_augsep", "mean")).to_csv(out / "descriptives.csv")
    contrasts = {
        "A1_spring_conv_vs_org": (s[s.succ == "spring"], "conventional"),
        "A2_autumn_conv_vs_org": (s[s.succ == "autumn"], "conventional"),
        "A0_conventional_autumn_vs_spring": (s[s.conventional == 1].assign(aut=lambda x: (x.succ == "autumn").astype(int)), "aut"),
    }
    es_all, did_all = [], []
    for label, (sub, tr) in contrasts.items():
        e, r = _contrast(sub, tr, base, label)
        es_all.append(e)
        did_all.append(r)
    es, res = pd.concat(es_all, ignore_index=True), pd.concat(did_all, ignore_index=True)
    es.to_csv(out / "event_study.csv", index=False)
    res.to_csv(out / "did.csv", index=False)
    try:
        sp = s[s.succ == "spring"]
        cs = est.cs_did(sp.assign(g=np.where(sp.conventional == 1, 2023, 0)), "share_bare_augsep", "g", "season")
        cs.to_csv(out / "cs_did_A1.csv", index=False)
    except Exception as e:
        log.warning("CS-DiD failed: %s", e)

    def fp(p):
        return "<0.001" if p < 0.001 else f"{p:.3f}"
    pn = {"AN": f"{len(s):,}", "AFarms": f"{s.cluster.nunique():,}",
          "AOrgShare": f"{(1 - s.conventional.mean()) * 100:.1f}"}
    for label, tag in (("A1_spring_conv_vs_org", "One"), ("A2_autumn_conv_vs_org", "Two"),
                       ("A0_conventional_autumn_vs_spring", "Zero")):
        m = res[(res.contrast == label) & (res.spec == "main")].set_index("outcome")
        r = m.loc["share_bare_augsep"]
        pn[f"A{tag}Eff"], pn[f"A{tag}Lo"], pn[f"A{tag}Hi"], pn[f"A{tag}P"] = (
            f"{r.est:.3f}", f"{r.lo:.3f}", f"{r.hi:.3f}", fp(r.p))
        r = m.loc["long_spell"]
        pn[f"A{tag}Spell"], pn[f"A{tag}SpellLo"], pn[f"A{tag}SpellHi"] = f"{r.est:.3f}", f"{r.lo:.3f}", f"{r.hi:.3f}"
        pre_es = es[(es.contrast == label) & (es.outcome == "share_bare_augsep") & (es.period < base)]
        pn[f"A{tag}PreMaxAbs"] = f"{pre_es.est.abs().max():.3f}"
        pn[f"A{tag}PreMinP"] = fp(pre_es.p.min())
        pl = res[(res.contrast == label) & (res.spec == "placebo_2021") & (res.outcome == "share_bare_augsep")]
        if len(pl):
            pn[f"A{tag}Placebo"], pn[f"A{tag}PlaceboP"] = f"{pl.est.iloc[0]:.3f}", fp(pl.p.iloc[0])
        rob = res[(res.contrast == label) & (res.outcome == "share_bare_augsep") &
                  res.spec.isin(["excl2022", "excl2019_2021_biomass_pilot", "high_obs_density", "ctrl_nobs", "ctrl_weather"])]
        pn[f"A{tag}RobMin"], pn[f"A{tag}RobMax"] = f"{rob.est.min():.3f}", f"{rob.est.max():.3f}"
    g = s.groupby(["succ", "conventional", s.season >= 2023]).share_bare_augsep.mean()
    for succ, st in (("spring", "Spr"), ("autumn", "Aut")):
        for conv, ct in ((1, "Conv"), (0, "Org")):
            pn[f"AMean{st}{ct}Pre"] = f"{g.loc[(succ, conv, False)]:.3f}"
            pn[f"AMean{st}{ct}Post"] = f"{g.loc[(succ, conv, True)]:.3f}"
    return pn


def design_b(seasons, ndti: float, base: int, out) -> dict:
    """Grain maize: sandy soils (GLM 6 cover until 1 Feb from 2023; nationally exempt before) vs other soils (GLM 6 period ends before the maize harvest)."""
    import yaml

    from capcausal.analysis import outcomes as oc
    from capcausal.analysis.panel import season_table
    from capcausal.utils.paths import ROOT
    root = DATA_DIR / "interim" / "s2_maize"
    parts = []
    for s_ in seasons:
        obs = oc.load_obs(s_, root=root)
        if obs.empty:
            continue
        obs = oc.classify_obs(obs, ndti_thr=ndti)
        m = oc.window_metrics(obs, s_, {"novjan": ("11-01", "01-31")})
        tab = season_table(s_)
        parts.append(tab.merge(m, on="uid").merge(obs.groupby("uid").tile.first().reset_index(), on="uid"))
    if not parts:
        return {}
    d = pd.concat(parts, ignore_index=True)
    d = d[(d.crop_group == "maize_grain") & (d.organic.fillna(0) == 0) & (d.glm6_zone != "unknown") & (d.n_novjan >= 2)].copy()
    d["treat"] = (d.glm6_zone == "sand_1feb").astype(int)
    d["post"] = (d.season >= 2023).astype(int)
    d["cluster"] = d.cvr.fillna(d.journalnr).astype(str)
    out.mkdir(parents=True, exist_ok=True)
    d.groupby(["season", "treat"]).agg(n=("uid", "size"), share_bare=("share_bare_novjan", "mean"),
                                       n_obs=("n_novjan", "mean")).to_csv(out / "descriptives.csv")
    fe = "tile^season + treat"
    es = est.event_study(d, "share_bare_novjan", "treat", "season", base, fe, "cluster")
    es.to_csv(out / "event_study.csv", index=False)
    t, _ = est.did(d, "share_bare_novjan", "treat", "post", fe, "cluster")
    t2, _ = _did(d, "any_bare_novjan", "treat", "post", fe, "cluster")
    pd.concat([t.assign(outcome="share_bare_novjan"), t2.assign(outcome="any_bare_novjan")]).to_csv(out / "did.csv", index=False)
    r = t.iloc[0]
    pre = es[(es.period < base) & es.p.notna()]
    return {"BN": f"{len(d):,}", "BEff": f"{r.est:.3f}", "BEffLo": f"{r.lo:.3f}", "BEffHi": f"{r.hi:.3f}",
            "BEffP": f"{r.p:.3f}" if r.p >= 0.001 else "<0.001",
            "BPreMaxAbs": f"{pre.est.abs().max():.3f}" if len(pre) else "nan",
            "BBaseSand": f"{d[(d.treat == 1) & (d.post == 0)].share_bare_novjan.mean():.3f}",
            "BBaseOther": f"{d[(d.treat == 0) & (d.post == 0)].share_bare_novjan.mean():.3f}"}


def farm_outcomes(d: pd.DataFrame, min_obs: int) -> pd.DataFrame:
    s = d[d.crop_group.isin(HARVESTED | {"maize_silage", "maize_grain"}) & (d.n_augsep >= min_obs)].copy()
    s["w"] = s.area_ha
    for c in ("share_bare_augsep", "any_bare_augsep", "share_bare_oct2to20"):
        s[c + "_w"] = s[c] * s.w
    g = s.groupby(["journalnr", "season"]).agg(cluster=("cluster", "first"), agri_area_ha=("agri_area_ha", "first"),
                                               arable_area_ha=("arable_area_ha", "first"), w=("w", "sum"),
                                               a=("share_bare_augsep_w", "sum"), b=("any_bare_augsep_w", "sum"),
                                               c=("share_bare_oct2to20_w", "sum"), n_parcels=("uid", "size"),
                                               organic=("organic", "mean"))
    g["share_bare_augsep"] = g.a / g.w
    g["any_bare_augsep"] = g.b / g.w
    g["share_bare_oct2to20"] = g.c / g.w
    return g.reset_index()


def design_c(d: pd.DataFrame, min_obs: int, out) -> dict:
    f = farm_outcomes(d, min_obs)
    f = f[(f.agri_area_ha > 0) & (f.organic < 0.5)]
    f["post"] = (f.season >= 2024).astype(int)
    out.mkdir(parents=True, exist_ok=True)
    ys = ["share_bare_augsep", "any_bare_augsep", "share_bare_oct2to20"]
    rd = pd.concat([est.rd_by_year(f, y, "agri_area_ha", 10.0, "season", cluster="cluster").assign(outcome=y)
                    for y in ys if f[y].notna().any()], ignore_index=True)
    rd.to_csv(out / "rd_by_year.csv", index=False)
    rows = []
    main_pre = f[f.season.isin([2023, 2024, 2025])]
    for y in ys:
        for h in (3.0, 5.0, 8.0):
            rows.append(_dd(main_pre, y, "agri_area_ha", 10.0, h, "post", "cluster", fe="season")
                        .assign(outcome=y, spec=f"pre2023_h{h:g}"))
            rows.append(_dd(f, y, "agri_area_ha", 10.0, h, "post", "cluster", fe="season")
                        .assign(outcome=y, spec=f"preall_h{h:g}"))
        donut = main_pre[(main_pre.agri_area_ha - 10).abs() > 0.5]
        rows.append(_dd(donut, y, "agri_area_ha", 10.0, 5.0, "post", "cluster", fe="season")
                    .assign(outcome=y, spec="pre2023_h5_donut"))
        for c in (7.0, 8.0, 13.0, 15.0):
            side = main_pre[(main_pre.agri_area_ha <= 10) if c < 10 else (main_pre.agri_area_ha > 10)]
            rows.append(_dd(side, y, "agri_area_ha", c, 2.5, "post", "cluster", fe="season")
                        .assign(outcome=y, spec=f"placebo_cut{c:g}"))
        pre = f[f.season <= 2023].copy()
        pre["fake_post"] = (pre.season >= 2021).astype(int)
        rows.append(_dd(pre, y, "agri_area_ha", 10.0, 5.0, "fake_post", "cluster", fe="season")
                    .assign(outcome=y, spec="placebo_year2021"))
    res = pd.concat(rows, ignore_index=True)
    res.to_csv(out / "diff_in_disc.csv", index=False)
    from rddensity import rddensity
    dens = []
    farms_all = pd.concat([pd.read_parquet(DATA_DIR / "interim" / "dk" / f"farms_{t}.parquet", columns=["agri_area_ha"])
                           .assign(year=t) for t in range(2018, 2026)])
    for t, g in farms_all.groupby("year"):
        x = g.agri_area_ha.values - 10.0
        x = x[np.abs(x) <= 8]
        try:
            r = rddensity(X=-x, c=0.0)
            dens.append({"year": t, "p": float(r.test["p_jk"]), "n": len(x),
                         "n_below": int((x <= 0).sum()), "n_above": int((x > 0).sum())})
        except Exception as e:
            dens.append({"year": t, "error": str(e)[:200], "n": len(x)})
    pd.DataFrame(dens).to_csv(out / "density_tests.csv", index=False)
    m = res[(res.spec == "pre2023_h5") & (res.term == "diff_in_disc")].set_index("outcome")
    all_ = res[(res.spec == "preall_h5") & (res.term == "diff_in_disc") & (res.outcome == "share_bare_augsep")].iloc[0]
    py_ = res[(res.spec == "placebo_year2021") & (res.term == "diff_in_disc") & (res.outcome == "share_bare_augsep")].iloc[0]
    return {"CEff": f"{m.loc['share_bare_augsep', 'est']:.3f}", "CEffLo": f"{m.loc['share_bare_augsep', 'lo']:.3f}",
            "CEffHi": f"{m.loc['share_bare_augsep', 'hi']:.3f}", "CEffP": f"{m.loc['share_bare_augsep', 'p']:.3f}",
            "CN": f"{int(m.loc['share_bare_augsep', 'n_obs']):,}",
            "CFarmsBelow": f"{int(((f.agri_area_ha <= 10) & (f.season == 2024)).sum()):,}",
            "CMde": f"{2.8 * m.loc['share_bare_augsep', 'se']:.3f}",
            "CEffAll": f"{all_.est:.3f}", "CEffAllLo": f"{all_.lo:.3f}", "CEffAllHi": f"{all_.hi:.3f}",
            "CEffOct": f"{m.loc['share_bare_oct2to20', 'est']:.3f}", "CEffOctP": f"{m.loc['share_bare_oct2to20', 'p']:.3f}",
            "CMeanBelow": f"{f[(f.agri_area_ha <= 10) & (f.agri_area_ha > 5)].share_bare_augsep.mean():.3f}",
            "CDensPMin": f"{pd.DataFrame(dens).p.min():.3f}", "CDensPLast": f"{pd.DataFrame(dens).set_index('year').p.get(2025, float('nan')):.3f}",
            "CPlaceboYear": f"{py_.est:.3f}", "CPlaceboYearP": f"{py_.p:.3f}"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", type=int, required=True)
    ap.add_argument("--base", type=int, default=2022)
    ap.add_argument("--min-obs", type=int, default=3)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    d = load(a.seasons)
    pa = design_a(d, a.base, a.min_obs, RESULTS_DIR / f"design_a{a.tag}")
    (RESULTS_DIR / f"design_a{a.tag}" / "summary.json").write_text(json.dumps({"paper_numbers": pa} if not a.tag else pa, indent=2))
    import yaml as _y
    from capcausal.utils.paths import ROOT as _R
    ndti = _y.safe_load((_R / "configs" / "default.yaml").read_text())["outcomes"]["bare_soil"]["ndti_max"]
    if (DATA_DIR / "interim" / "s2_maize").exists():
        pb = design_b(a.seasons, ndti, a.base, RESULTS_DIR / f"design_b{a.tag}")
        if pb:
            (RESULTS_DIR / f"design_b{a.tag}" / "summary.json").write_text(json.dumps({"paper_numbers": pb} if not a.tag else pb, indent=2))
    if d.season.max() >= 2024:
        pc = design_c(d, a.min_obs, RESULTS_DIR / f"design_c{a.tag}")
        (RESULTS_DIR / f"design_c{a.tag}" / "summary.json").write_text(json.dumps({"paper_numbers": pc} if not a.tag else pc, indent=2))
    print(pa)
