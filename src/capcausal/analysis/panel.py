"""Parcel-season table: crops, successor, soil zone, organic status and holding size."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd

from capcausal.iacs.dk_soil import glm6_zone, jb_at
from capcausal.utils.paths import DATA_DIR

DK = DATA_DIR / "interim" / "dk"
BASE_COLS = ["uid", "journalnr", "cvr", "marknr", "crop_code", "crop_group", "autumn_sown", "arable",
             "agricultural", "glm7_category", "area_ha"]


def _attrs(year: int, cols: list[str]) -> pd.DataFrame:
    return pd.read_parquet(DK / f"parcels_{year}.parquet", columns=cols)


def season_table(s: int) -> pd.DataFrame:
    out_f = DK / f"season_{s}.parquet"
    if out_f.exists():
        return pd.read_parquet(out_f)
    g = gpd.read_parquet(DK / f"parcels_{s}.parquet", columns=BASE_COLS + ["geometry"])
    pt = g.geometry.representative_point()
    g["x"], g["y"] = pt.x.values, pt.y.values
    g["jb"] = jb_at(g.x.values, g.y.values)
    g["glm6_zone"] = glm6_zone(g.jb.values)
    df = pd.DataFrame(g.drop(columns="geometry"))
    df["season"] = s

    nxt = DK / f"match_{s}_{s + 1}.parquet"
    if nxt.exists():
        m = pd.read_parquet(nxt).rename(columns={"uid_a": "uid", "uid_b": "uid_next", "overlap": "overlap_next"})
        a = _attrs(s + 1, ["uid", "crop_group", "autumn_sown", "glm7_category"]).rename(columns={
            "uid": "uid_next", "crop_group": "crop_group_next", "autumn_sown": "autumn_sown_next",
            "glm7_category": "glm7_category_next"})
        df = df.merge(m.merge(a, on="uid_next"), on="uid", how="left")
    prv = DK / f"match_{s - 1}_{s}.parquet"
    if prv.exists():
        m = pd.read_parquet(prv).rename(columns={"uid_a": "uid_prev", "uid_b": "uid", "overlap": "overlap_prev"})
        m = m.sort_values("overlap_prev", ascending=False).drop_duplicates("uid")
        a = _attrs(s - 1, ["uid", "crop_group", "glm7_category"]).rename(columns={
            "uid": "uid_prev", "crop_group": "crop_group_prev", "glm7_category": "glm7_category_prev"})
        df = df.merge(m.merge(a, on="uid_prev"), on="uid", how="left")

    farms = pd.read_parquet(DK / f"farms_{s}.parquet", columns=["journalnr", "agri_area_ha", "arable_area_ha",
                                                                  "grass_perm", "grass_rot", "fallow"])
    df = df.merge(farms, on="journalnr", how="left")
    org = DK / f"organic_{s}.parquet"
    if org.exists():
        o = pd.read_parquet(org)
        df = df.merge(o, on="uid", how="left")
        df["organic"] = df["organic"].fillna(0).astype("int8")
    df.to_parquet(out_f)
    return df


def harvested_arable(df: pd.DataFrame) -> pd.Series:
    """Parcels subject to GLM 6 after harvest: arable crops harvested in summer/autumn, excluding grass (exempt), potatoes (exempt), fallow/non-arable and vegetables/woody crops."""
    return df.crop_group.isin(["cereal_winter", "cereal_spring", "rapeseed_winter", "rapeseed_spring",
                               "legume", "seed_grass_or_herb", "maize_silage", "maize_grain", "beet"])


def organic_layer(year: int) -> pd.DataFrame:
    """Flag parcels (uid of `year`) whose representative point lies in an organic area polygon."""
    import zipfile
    raw = DATA_DIR / "raw" / "dk_lbst"
    z = raw / f"Oekologiske_arealer_{year}.zip"
    d = raw / f"Oekologiske_arealer_{year}"
    if not d.exists():
        with zipfile.ZipFile(z) as f:
            f.extractall(d)
    org = gpd.read_file(next(d.rglob("*.shp")))[["geometry"]].to_crs(25832)
    g = gpd.read_parquet(DK / f"parcels_{year}.parquet", columns=["uid", "geometry"])
    pts = gpd.GeoDataFrame({"uid": g.uid.values}, geometry=g.geometry.representative_point(), crs=25832)
    hit = gpd.sjoin(pts, org, predicate="within", how="inner").uid.unique()
    return pd.DataFrame({"uid": g.uid.values, "organic": np.isin(g.uid.values, hit).astype("int8")})
