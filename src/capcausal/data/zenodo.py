"""Download files from a Zenodo record and record them in data/MANIFEST.csv."""
from __future__ import annotations

import csv
import fnmatch
import hashlib
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "data" / "MANIFEST.csv"


def _md5(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _manifest_has(checksum: str) -> bool:
    if not MANIFEST.exists():
        return False
    return checksum in MANIFEST.read_text(encoding="utf-8")


def _append_manifest(row: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    new = not MANIFEST.exists() or MANIFEST.stat().st_size == 0
    fields = ["dataset", "version", "url", "downloaded_at", "files", "bytes", "checksum", "license", "notes"]
    with MANIFEST.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(row)


def _fetch_segment(url: str, part: Path, start: int, end: int, max_tries: int = 10) -> None:
    """Download bytes [start, end] into `part`, resuming from its current size."""
    want = end - start + 1
    for attempt in range(max_tries):
        have = part.stat().st_size if part.exists() else 0
        if have >= want:
            return
        try:
            headers = {"Range": f"bytes={start + have}-{end}"}
            with requests.get(url, headers=headers, stream=True, timeout=120) as r:
                if r.status_code != 206:
                    raise RuntimeError(f"HTTP {r.status_code}")
                with part.open("ab") as f:
                    for b in r.iter_content(1 << 20):
                        f.write(b)
        except Exception as e:
            wait = min(600, 5 * 2 ** attempt)
            print(f"  retry {attempt + 1} for {part.name} in {wait}s: {e}", flush=True)
            time.sleep(wait)
    if part.stat().st_size < want:
        raise RuntimeError(f"segment {part.name} incomplete")


def download(url: str, dest: Path, size: int, n_segments: int = 12) -> None:
    """Resumable multi-connection download with HTTP Range requests and exponential backoff."""
    from concurrent.futures import ThreadPoolExecutor

    seg = -(-size // n_segments)
    bounds = [(i * seg, min(size, (i + 1) * seg) - 1) for i in range(n_segments) if i * seg < size]
    parts = [dest.with_name(f"{dest.name}.seg{i:02d}") for i in range(len(bounds))]
    with ThreadPoolExecutor(len(bounds)) as ex:
        list(ex.map(lambda a: _fetch_segment(url, *a), [(p, s, e) for p, (s, e) in zip(parts, bounds)]))
    tmp = dest.with_suffix(dest.suffix + ".part")
    with tmp.open("wb") as out:
        for p in parts:
            with p.open("rb") as f:
                while b := f.read(1 << 24):
                    out.write(b)
    tmp.replace(dest)
    for p in parts:
        p.unlink()


def fetch_record(record_id: str, outdir: Path, patterns: list[str]) -> None:
    rec = requests.get(f"https://zenodo.org/api/records/{record_id}", timeout=60).json()
    meta = rec["metadata"]
    outdir.mkdir(parents=True, exist_ok=True)
    for f in rec["files"]:
        key = f["key"]
        if not any(fnmatch.fnmatch(key, p) for p in patterns):
            continue
        dest = outdir / key
        algo, checksum = f["checksum"].split(":", 1)
        if dest.exists() and dest.stat().st_size == f["size"] and _md5(dest) == checksum:
            print(f"ok (cached) {key}", flush=True)
        else:
            print(f"downloading {key} ({f['size'] / 1e6:.0f} MB)", flush=True)
            download(f["links"]["self"], dest, f["size"])
            got = _md5(dest)
            if got != checksum:
                dest.unlink()
                raise RuntimeError(f"checksum mismatch for {key}")
            print(f"verified {key}", flush=True)
        if not _manifest_has(checksum):
            _append_manifest({
                "dataset": meta["title"], "version": meta.get("version", ""),
                "url": f["links"]["self"], "downloaded_at": datetime.now(timezone.utc).isoformat(),
                "files": key, "bytes": f["size"], "checksum": f"{algo}:{checksum}",
                "license": (meta.get("license") or {}).get("id", ""), "notes": f"zenodo record {record_id}",
            })


if __name__ == "__main__":
    fetch_record(sys.argv[1], Path(sys.argv[2]), sys.argv[3:])
