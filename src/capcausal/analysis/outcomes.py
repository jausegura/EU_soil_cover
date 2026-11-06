"""Parcel-season bare-soil metrics from per-date parcel statistics."""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from capcausal.utils.paths import S2_DIR


def load_obs(season: int, tiles: list[str] | None = None, min_pix: int = 8,
             min_valid_frac: float = 0.5, root: Path | None = None) -> pd.DataFrame:
    """All valid parcel observations for one season (harvest year)."""
    root = root or S2_DIR
    tiles = tiles or sorted(p.name for p in root.iterdir() if p.is_dir() and len(p.name) == 5)
    out = []
    for t in tiles:
        files = glob.glob(str(root / t / str(season) / "*.parquet"))
        npx_f = root / t / f"npix_{season}.parquet"
        if not files or not npx_f.exists():
            continue
        d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        d = d.merge(pd.read_parquet(npx_f), on="uid")
        d = d[(d.n_pix >= min_pix) & (d.n_valid >= min_valid_frac * d.n_pix)]
        d["tile"] = t
        out.append(d)
    if not out:
        return pd.DataFrame()
    d = pd.concat(out, ignore_index=True)
    return d.sort_values("n_valid", ascending=False).drop_duplicates(["uid", "date"]).reset_index(drop=True)


def classify_obs(d: pd.DataFrame, ndti_thr: float = 0.10, maj: float = 0.5) -> pd.DataFrame:
    col = f"f_bare_ndti_lt{int(round(ndti_thr * 100)):02d}"
    d = d.copy()
    d["bare"] = d[col] >= maj
    d["low_ndvi"] = d["f_ndvi_lt30"] >= maj
    d["residue"] = d["low_ndvi"] & ~d["bare"]
    d["green"] = ~d["low_ndvi"]
    return d


def window_metrics(d: pd.DataFrame, season: int, windows: dict[str, tuple[str, str]]) -> pd.DataFrame:
    """Per-parcel metrics in each window {name: ("MM-DD", "MM-DD")}; windows may cross the new year."""
    res = []
    for name, (a, b) in windows.items():
        start = pd.Timestamp(f"{season}-{a}")
        end = pd.Timestamp(f"{season + (1 if b < a else 0)}-{b}")
        w = d[(d.date >= start) & (d.date <= end)]
        g = w.groupby("uid")
        m = pd.DataFrame({
            f"n_{name}": g.size(),
            f"share_bare_{name}": g.bare.mean(),
            f"any_bare_{name}": g.bare.max().astype("int8"),
            f"share_green_{name}": g.green.mean(),
        })
        if "ndvi" in w:
            m[f"ndvi_{name}"] = g.ndvi.mean()
        res.append(m)
    return pd.concat(res, axis=1).reset_index()


def bare_spell(d: pd.DataFrame, season: int, start: str = "07-15", end: str = "10-31") -> pd.DataFrame:
    """Days between the first and last bare observation without an intervening green observation (lower bound of the length of the longest bare spell), in [start, end]."""
    s0, s1 = pd.Timestamp(f"{season}-{start}"), pd.Timestamp(f"{season}-{end}")
    w = d[(d.date >= s0) & (d.date <= s1)].sort_values(["uid", "date"])
    out = {}
    for uid, g in w.groupby("uid", sort=False):
        best, cur0, prev = 0, None, None
        for dt, bare, green in zip(g.date.values, g.bare.values, g.green.values):
            if bare:
                cur0 = dt if cur0 is None else cur0
                prev = dt
                best = max(best, int((prev - cur0) / np.timedelta64(1, "D")))
            elif green:
                cur0 = None
        out[uid] = best
    return pd.DataFrame({"uid": list(out), "bare_spell_days": list(out.values())})
