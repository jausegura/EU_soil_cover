#!/usr/bin/env python
"""Organic status of parcels from the official organic-area layers."""
import sys

from capcausal.analysis.panel import organic_layer
from capcausal.utils.paths import DATA_DIR

for y in [int(a) for a in sys.argv[1:]]:
    out = DATA_DIR / "interim" / "dk" / f"organic_{y}.parquet"
    if out.exists() or not (DATA_DIR / "raw" / "dk_lbst" / f"Oekologiske_arealer_{y}.zip").exists():
        continue
    o = organic_layer(y)
    o.to_parquet(out)
    print(y, "organic share of parcels:", round(o.organic.mean(), 4))
