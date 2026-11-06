#!/usr/bin/env python
"""Agreement between the two Sentinel-2 processing chains."""
import glob
import json
from pathlib import Path

import pandas as pd

from capcausal.utils.paths import DATA_DIR, RESULTS_DIR, ROOT

OUT = RESULTS_DIR / "replication"


def agreement(seasons=range(2019, 2026), thr_col="f_bare_ndti_lt10"):
    """Per tile-date confusion counts between the chains."""
    rows = []
    for s in seasons:
        for tdir in sorted(glob.glob(str(DATA_DIR / "interim" / "s2_cdse" / "*" / str(s)))):
            tile = Path(tdir).parent.name
            npx_f = DATA_DIR / "interim" / "s2" / tile / f"npix_{s}.parquet"
            if not npx_f.exists():
                continue
            npx = pd.read_parquet(npx_f)
            npx = npx[npx.n_pix >= 8]
            pc = {}
            for f in glob.glob(str(DATA_DIR / "interim" / "s2" / tile / str(s) / "*.parquet")):
                d = pd.read_parquet(f, columns=["uid", "n_valid", thr_col, "f_ndvi_lt30", "date"])
                pc.setdefault(str(d.date.iloc[0].date()), []).append(d)
            for f in glob.glob(str(Path(tdir) / "*.parquet")):
                date = Path(f).stem
                if date not in pc:
                    continue
                a = pd.concat(pc[date]).sort_values("n_valid", ascending=False).drop_duplicates("uid")
                b = pd.read_parquet(f, columns=["uid", "n_valid", thr_col, "f_ndvi_lt30"])
                m = a.merge(b, on="uid", how="outer", suffixes=("_pc", "_cd")).merge(npx, on="uid")
                v_pc = m.n_valid_pc.fillna(0) >= 0.5 * m.n_pix
                v_cd = m.n_valid_cd.fillna(0) >= 0.5 * m.n_pix
                both = v_pc & v_cd
                bp = m[thr_col + "_pc"] >= 0.5
                bc = m[thr_col + "_cd"] >= 0.5
                rows.append({"season": s, "tile": tile, "date": date, "n_either": int((v_pc | v_cd).sum()),
                             "n": int(both.sum()), "valid_pc_only": int((v_pc & ~v_cd).sum()),
                             "valid_cd_only": int((v_cd & ~v_pc).sum()),
                             "a": int((both & bp & bc).sum()), "b": int((both & bp & ~bc).sum()),
                             "c": int((both & ~bp & bc).sum()), "d": int((both & ~bp & ~bc).sum())})
    r = pd.DataFrame(rows)
    r = r[r.n > 0]
    r["agree"] = (r.a + r.d) / r.n
    return r


def metrics(g):
    a, b, c, d = g.a.sum(), g.b.sum(), g.c.sum(), g.d.sum()
    n = a + b + c + d
    po = (a + d) / n
    pe = ((a + b) * (a + c) + (c + d) * (b + d)) / n ** 2
    return pd.Series({"n": n, "agree": po, "pos_agree": 2 * a / (2 * a + b + c), "kappa": (po - pe) / (1 - pe),
                      "prev_pc": (a + b) / n, "prev_cd": (a + c) / n,
                      "valid_both_share": n / g.n_either.sum(),
                      "valid_pc_only": g.valid_pc_only.sum() / g.n_either.sum(), "valid_cd_only": g.valid_cd_only.sum() / g.n_either.sum()})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ag = agreement()
    ag.to_csv(OUT / "agreement_by_tile_date.csv", index=False)
    by = ag.groupby("season").apply(metrics)
    tot = metrics(ag)
    by.to_csv(OUT / "agreement_by_season.csv")
    pn = {"RepAgreeAll": f"{tot.agree:.3f}", "RepAgreeMin": f"{by.agree.min():.3f}", "RepAgreeMax": f"{by.agree.max():.3f}",
          "RepNpairs": f"{int(tot.n):,}", "RepPosAgree": f"{tot.pos_agree:.3f}", "RepPosAgreeMin": f"{by.pos_agree.min():.3f}",
          "RepPosAgreeMax": f"{by.pos_agree.max():.3f}", "RepKappa": f"{tot.kappa:.3f}", "RepPrev": f"{tot.prev_cd:.3f}",
          "RepValidBoth": f"{tot.valid_both_share:.3f}", "RepValidBothMin": f"{by.valid_both_share.min():.3f}",
          "RepValidPcOnly": f"{tot.valid_pc_only:.3f}", "RepValidCdOnly": f"{tot.valid_cd_only:.3f}"}
    L = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
         r"Season & Pairs valid in both & Agreement & Positive agreement & Cohen's $\kappa$ & Bare share & Valid in both & Valid in one only \\", r"\midrule"]
    for s_, r in by.iterrows():
        L.append(f"{s_} & {int(r.n):,} & {r.agree:.3f} & {r.pos_agree:.3f} & {r.kappa:.3f} & {r.prev_cd:.3f} & {r.valid_both_share:.3f} & {r.valid_pc_only + r.valid_cd_only:.3f} \\\\".replace(",", "{,}"))
    L += [r"\bottomrule", r"\end{tabular}"]
    (RESULTS_DIR / "tables" / "supp_agreement.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    pcn = {}
    for f in glob.glob(str(ROOT / "results_pc" / "*" / "*.json")):
        pcn.update(json.loads(Path(f).read_text()).get("paper_numbers", {}))
    for k in ("ValSens", "ValSpec", "AOneEff", "AOneLo", "AOneHi", "AOnePlacebo", "ATwoEff", "CEff", "CEffLo", "CEffHi"):
        if k in pcn:
            pn["Pc" + k] = pcn[k]
    (OUT / "summary.json").write_text(json.dumps({"paper_numbers": pn}, indent=2))
    print(pn)
    print(by)


if __name__ == "__main__":
    main()
