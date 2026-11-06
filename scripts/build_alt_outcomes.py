#!/usr/bin/env python
"""August-September parcel metrics under alternative indicator definitions."""
from __future__ import annotations

import logging
import sys

import pandas as pd

from capcausal.analysis import outcomes as oc
from capcausal.utils.paths import PROCESSED_DIR

log = logging.getLogger("alt")
TAUS = {"tau06": "f_bare_ndti_lt06", "tau08": "f_bare_ndti_lt08", "base": "f_bare_ndti_lt10",
        "tau12": "f_bare_ndti_lt12", "tau14": "f_bare_ndti_lt14"}


def _agg(w: pd.DataFrame, flag: pd.Series, name: str) -> pd.DataFrame:
    g = flag.groupby(w.uid)
    return pd.DataFrame({f"n_{name}": g.size(), f"share_{name}": g.mean(), f"any_{name}": g.max().astype("int8")})


def build(season: int) -> pd.DataFrame:
    obs = oc.load_obs(season, min_valid_frac=0.3)
    obs = obs[(obs.date >= f"{season}-08-01") & (obs.date <= f"{season}-09-30")].copy()
    obs["vf"] = obs.n_valid / obs.n_pix
    parts = []
    main = obs[obs.vf >= 0.5]
    for name, col in TAUS.items():
        parts.append(_agg(main, main[col] >= 0.5, name))
    parts.append(_agg(main, main.f_ndvi_lt30 >= 0.5, "ndvi"))
    parts.append(_agg(main, main.f_bare_ndti_lt10 >= 0.3, "maj30"))
    parts.append(_agg(main, main.f_bare_ndti_lt10 >= 0.7, "maj70"))
    for v in (0.3, 0.7, 0.9):
        w = obs[obs.vf >= v]
        parts.append(_agg(w, w.f_bare_ndti_lt10 >= 0.5, f"vf{int(v * 100)}"))
    main = main.assign(is_aug=(main.date.dt.month == 8).astype("int16"), is_sep=(main.date.dt.month == 9).astype("int16"))
    g = main.groupby("uid")
    feats = pd.DataFrame({
        "n_aug": g.is_aug.sum(),
        "n_sep": g.is_sep.sum(),
        "mean_f_ndvi": g.f_ndvi_lt30.mean(), "max_f_ndvi": g.f_ndvi_lt30.max(),
        **{f"mean_{c[2:]}": g[c].mean() for c in TAUS.values()},
        **{f"max_{c[2:]}": g[c].max() for c in TAUS.values()},
        "mean_vf": g.vf.mean(),
    })
    out = pd.concat(parts + [feats], axis=1).reset_index().rename(columns={"index": "uid"})
    out["season"] = season
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    seasons = [int(s) for s in sys.argv[1:]] or list(range(2019, 2026))
    for s in seasons:
        f = PROCESSED_DIR / f"alt_augsep_{s}.parquet"
        df = build(s)
        df.to_parquet(f)
        log.info("season %d: %d parcels", s, len(df))
