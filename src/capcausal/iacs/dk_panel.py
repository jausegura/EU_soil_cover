"""Parcel-year and holding-year tables and cross-year parcel links for Denmark."""
from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio

from capcausal.iacs.dk_crops import classify_frame
from capcausal.utils.paths import DATA_DIR

log = logging.getLogger(__name__)
RAW = DATA_DIR / "raw" / "dk_lbst"
INV = DATA_DIR / "raw" / "iacs_inventory" / "DK"
OUT = DATA_DIR / "interim" / "dk"
CRS = 25832


def hcat_lookup() -> pd.DataFrame:
    """crop_code -> (EC_trans_n, EC_hcat_n) from the harmonised inventory (most frequent mapping)."""
    files = sorted(str(p).replace("\\", "/") for p in INV.glob("GSA-DK-*.geoparquet"))
    q = f"""select cast(crop_code as int) crop_code, EC_trans_n, EC_hcat_n, count(*) k
            from read_parquet({files}) where crop_code is not null group by all"""
    df = duckdb.sql(q).df().sort_values("k", ascending=False)
    return df.drop_duplicates("crop_code")[["crop_code", "EC_trans_n", "EC_hcat_n"]]


def read_marker(year: int) -> gpd.GeoDataFrame:
    z = RAW / f"Marker_{year}.zip"
    d = RAW / f"Marker_{year}"
    if not d.exists():
        with zipfile.ZipFile(z) as f:
            f.extractall(d)
    shp = next(d.rglob("*.shp"))
    gdf = pyogrio.read_dataframe(shp)
    if gdf.crs is None or gdf.crs.to_epsg() != CRS:
        gdf = gdf.to_crs(CRS)
    return gdf


def build_year(year: int, lookup: pd.DataFrame) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    g = read_marker(year)
    cols = {c.lower(): c for c in g.columns}
    g = g.rename(columns={cols.get("afgkode", "Afgkode"): "crop_code", cols.get("afgroede", "Afgroede"): "crop_name",
                          cols.get("journalnr", "Journalnr"): "journalnr", cols.get("cvr", "CVR"): "cvr",
                          cols.get("imk_areal", "IMK_areal"): "area_ha", cols.get("marknr", "Marknr"): "marknr"})
    g["crop_code"] = pd.to_numeric(g["crop_code"], errors="coerce").astype("Int64")
    g = g[g.geometry.notna() & ~g.geometry.is_empty].copy()
    g["geometry"] = g.geometry.make_valid()
    g = g.merge(lookup, on="crop_code", how="left")
    g = classify_frame(g.astype({"crop_code": "float64"}))
    g = gpd.GeoDataFrame(g, geometry="geometry", crs=CRS)
    g["geom_area_ha"] = g.geometry.area / 1e4
    g["area_ha"] = pd.to_numeric(g.get("area_ha"), errors="coerce").fillna(g["geom_area_ha"])
    g = g.reset_index(drop=True)
    g["uid"] = np.arange(1, len(g) + 1, dtype="int32")
    g["year"] = year
    ans = [c for c in g.columns if c.lower().startswith("ans")]
    for c in ans:
        g[c] = pd.to_numeric(g[c], errors="coerce").fillna(0).astype("int8")
    keep = ["uid", "year", "journalnr", "cvr", "marknr", "crop_code", "crop_name", "EC_hcat_n", "crop_group",
            "autumn_sown", "arable", "agricultural", "glm7_category", "area_ha", "geom_area_ha"] + ans + ["geometry"]
    g = g[[c for c in keep if c in g.columns]]

    farms = g.assign(agri=lambda d: d.area_ha.where(d.agricultural, 0.0),
                     arab=lambda d: d.area_ha.where(d.arable & d.agricultural, 0.0),
                     grass_perm=lambda d: d.area_ha.where(d.crop_group == "grass_permanent", 0.0),
                     grass_rot=lambda d: d.area_ha.where(d.crop_group == "grass_rotation", 0.0),
                     fallow=lambda d: d.area_ha.where(d.crop_group == "fallow", 0.0))
    agg = {"agri": "sum", "arab": "sum", "grass_perm": "sum", "grass_rot": "sum", "fallow": "sum",
           "uid": "count", "cvr": "first"}
    for c in ans:
        farms[c + "_ha"] = farms.area_ha * farms[c]
        agg[c + "_ha"] = "sum"
    f = farms.groupby("journalnr", dropna=True).agg(agg).rename(
        columns={"agri": "agri_area_ha", "arab": "arable_area_ha", "uid": "n_parcels"}).reset_index()
    f["year"] = year
    return g, f


def match_years(a: gpd.GeoDataFrame, b: gpd.GeoDataFrame, min_overlap: float = 0.5) -> pd.DataFrame:
    """Correspondence a(year t) -> b(year t+1): the b parcel containing a's representative point, accepted if intersection covers >= min_overlap of the a parcel area."""
    pts = gpd.GeoDataFrame({"uid_a": a.uid.values}, geometry=a.geometry.representative_point(), crs=a.crs)
    j = gpd.sjoin(pts, b[["uid", "geometry"]].rename(columns={"uid": "uid_b"}), predicate="within", how="inner")
    j = j.drop_duplicates("uid_a")
    ga = a.set_index("uid").geometry.loc[j.uid_a.values].values
    gb = b.set_index("uid").geometry.loc[j.uid_b.values].values
    inter = gpd.GeoSeries(ga, crs=a.crs).intersection(gpd.GeoSeries(gb, crs=a.crs), align=False).area.values
    ov = inter / gpd.GeoSeries(ga, crs=a.crs).area.values
    m = pd.DataFrame({"uid_a": j.uid_a.values, "uid_b": j.uid_b.values, "overlap": ov.astype("float32")})
    return m[m.overlap >= min_overlap].reset_index(drop=True)


def build_all(years: list[int]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lookup = hcat_lookup()
    prev = None
    for y in years:
        p_out, f_out = OUT / f"parcels_{y}.parquet", OUT / f"farms_{y}.parquet"
        if p_out.exists() and f_out.exists():
            g = gpd.read_parquet(p_out)
        else:
            g, f = build_year(y, lookup)
            g.to_parquet(p_out)
            f.to_parquet(f_out)
            log.info("year %d: %d parcels, %d farms", y, len(g), len(f))
        if prev is not None:
            m_out = OUT / f"match_{y - 1}_{y}.parquet"
            if not m_out.exists():
                match_years(prev, g).to_parquet(m_out)
        prev = g


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    build_all([int(y) for y in sys.argv[1:]])
