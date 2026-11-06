"""Daily ERA5 precipitation through Open-Meteo on a 0.5 degree grid."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import requests

from capcausal.utils.paths import DATA_DIR

OUT = DATA_DIR / "interim" / "weather" / "era5_dk_daily.parquet"
URL = "https://archive-api.open-meteo.com/v1/archive"


def grid(lat=(54.5, 57.5), lon=(8.0, 12.5), step=0.5):
    la = np.arange(lat[0], lat[1] + 1e-9, step)
    lo = np.arange(lon[0], lon[1] + 1e-9, step)
    return [(round(a, 2), round(b, 2)) for a in la for b in lo]


def fetch(points, start="2017-07-01", end="2026-01-31") -> pd.DataFrame:
    rows = []
    for i in range(0, len(points), 20):
        chunk = points[i:i + 20]
        params = {"latitude": ",".join(str(p[0]) for p in chunk), "longitude": ",".join(str(p[1]) for p in chunk),
                  "start_date": start, "end_date": end, "daily": "precipitation_sum,temperature_2m_mean",
                  "models": "era5", "timezone": "UTC"}
        r = None
        for attempt in range(8):
            try:
                r = requests.get(URL, params=params, timeout=300)
                if r.status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(30 * (attempt + 1))
        if r is None or r.status_code != 200:
            raise RuntimeError(f"Open-Meteo request failed for chunk starting {chunk[0]}")
        js = r.json()
        js = js if isinstance(js, list) else [js]
        for (la, lo), j in zip(chunk, js):
            d = j["daily"]
            rows.append(pd.DataFrame({"lat": la, "lon": lo, "date": pd.to_datetime(d["time"]),
                                      "precip_mm": d["precipitation_sum"], "t2m_c": d["temperature_2m_mean"]}))
        time.sleep(5)
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    parts = []
    for season in range(2017, 2026):
        f = OUT.with_name(f"era5_dk_{season}.parquet")
        if not f.exists():
            fetch(grid(), start=f"{season}-08-01", end=f"{season}-10-31").dropna(subset=["precip_mm"]).to_parquet(f)
            time.sleep(60)
        parts.append(pd.read_parquet(f))
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(OUT)
    print(len(df), df.date.min(), df.date.max(), df[["lat", "lon"]].drop_duplicates().shape[0], "cells")
