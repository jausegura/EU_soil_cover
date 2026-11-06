"""Holding-level crop-rotation outcomes from parcel declarations."""
from __future__ import annotations

import numpy as np
import pandas as pd

from capcausal.utils.paths import DATA_DIR

DK = DATA_DIR / "interim" / "dk"
NOT_SUBJECT = {"grass_rotation", "grass_permanent", "fallow", "woody_or_forest", "other_nonarable"}


def _cat(year: int) -> pd.DataFrame:
    return pd.read_parquet(DK / f"parcels_{year}.parquet",
                           columns=["uid", "journalnr", "cvr", "crop_group", "arable", "glm7_category", "area_ha"])


def _prev_map(t: int) -> pd.DataFrame:
    """uid_t -> uid_{t-1} (best overlap)."""
    m = pd.read_parquet(DK / f"match_{t - 1}_{t}.parquet")
    m = m.sort_values("overlap", ascending=False).drop_duplicates("uid_b")
    return m.rename(columns={"uid_a": "uid_prev", "uid_b": "uid"})[["uid", "uid_prev"]]


def farm_rotation(t: int) -> pd.DataFrame:
    cur = _cat(t)
    cur = cur[cur.arable & ~cur.crop_group.isin(NOT_SUBJECT)].copy()
    chain = cur[["uid"]].copy()
    cats = {0: cur.set_index("uid").glm7_category}
    uid_k = chain.uid
    ids = {0: uid_k}
    for k in (1, 2, 3):
        f = DK / f"match_{t - k}_{t - k + 1}.parquet"
        if not f.exists():
            break
        m = _prev_map(t - k + 1).set_index("uid").uid_prev
        prev_ids = ids[k - 1].map(m)
        c = _cat(t - k).set_index("uid").glm7_category
        ids[k] = prev_ids
        cats[k] = prev_ids.map(c)
    cur = cur.reset_index(drop=True)
    cur["cat_1"] = cats[1].values if 1 in cats else np.nan
    cur["matched"] = cur.cat_1.notna()
    cur["changed"] = cur.matched & (cur.glm7_category != cur.cat_1)
    if 3 in cats:
        same4 = (cur.glm7_category == cats[1].values) & (cats[1].values == cats[2].values) & (cats[2].values == cats[3].values)
        cur["same4"] = same4 & pd.Series(cats[3].values).notna().values
    else:
        cur["same4"] = np.nan
    cur["a_m"] = cur.area_ha * cur.matched
    cur["a_c"] = cur.area_ha * cur.changed
    cur["a_4"] = cur.area_ha * cur.same4.astype(float)
    g = cur.groupby("journalnr").agg(subject_ha=("area_ha", "sum"), matched_ha=("a_m", "sum"),
                                      changed_ha=("a_c", "sum"), same4_ha=("a_4", "sum"),
                                      n_subject=("uid", "size"))
    g["share_change"] = g.changed_ha / g.matched_ha.where(g.matched_ha > 0)
    g["meets35"] = (g.share_change >= 0.35).astype(float).where(g.share_change.notna())
    g["share_4yr"] = (g.same4_ha / g.matched_ha.where(g.matched_ha > 0)) if 3 in cats else np.nan
    g["match_rate"] = g.matched_ha / g.subject_ha
    farms = pd.read_parquet(DK / f"farms_{t}.parquet")
    out = farms.merge(g.reset_index(), on="journalnr", how="left")
    out["year"] = t
    return out
