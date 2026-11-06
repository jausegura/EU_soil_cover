"""Download files and record them in data/MANIFEST.csv."""
from __future__ import annotations

import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

import requests

from capcausal.data.zenodo import _append_manifest, _manifest_has, download


def _sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def fetch(url: str, outdir: Path, dataset: str, license_: str) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    dest = outdir / unquote(url.rsplit("/", 1)[-1])
    head = requests.head(url, allow_redirects=True, timeout=60)
    size = int(head.headers.get("Content-Length", 0))
    if dest.exists() and dest.stat().st_size == size:
        print(f"ok (cached) {dest.name}", flush=True)
    else:
        print(f"downloading {dest.name} ({size / 1e6:.0f} MB)", flush=True)
        download(url, dest, size, n_segments=8 if size > 50e6 else 1)
        if dest.stat().st_size != size:
            raise RuntimeError(f"size mismatch for {dest.name}")
    checksum = _sha256(dest)
    if not _manifest_has(checksum):
        _append_manifest({
            "dataset": dataset, "version": head.headers.get("Last-Modified", ""), "url": url,
            "downloaded_at": datetime.now(timezone.utc).isoformat(), "files": dest.name,
            "bytes": size, "checksum": f"sha256:{checksum}", "license": license_, "notes": "",
        })
    return dest


if __name__ == "__main__":
    out, ds, lic, *urls = sys.argv[1:]
    for u in urls:
        fetch(u, Path(out), ds, lic)
