"""Project paths and environment variables."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[3]
load_dotenv(ROOT / ".env", override=False)


def env(name: str, default: str | None = None, required: bool = False) -> str | None:
    val = os.environ.get(name, default)
    if required and not val:
        raise RuntimeError(f"Missing required environment variable {name} (fill it in .env)")
    return val


DATA_DIR = (ROOT / env("DATA_DIR", "./data")).resolve()
RESULTS_DIR = ROOT / "results"
LOGS_DIR = ROOT / "logs"
for _d in (DATA_DIR, RESULTS_DIR, LOGS_DIR, RESULTS_DIR / "tables", RESULTS_DIR / "figures"):
    _d.mkdir(parents=True, exist_ok=True)

S2_SOURCE = env("CAPCAUSAL_S2_SOURCE", "cdse")
S2_DIR = DATA_DIR / "interim" / ("s2_cdse" if S2_SOURCE == "cdse" else "s2")
PROCESSED_DIR = DATA_DIR / ("processed_cdse" if S2_SOURCE == "cdse" else "processed")
