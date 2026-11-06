# Post-harvest bare soil from Sentinel-2 and EU soil-cover rules

Core scripts of the analysis: parcel statistics, indicator, validation and policy tests.

## Setup

```bash
python -m venv .venv
.venv/bin/python -m pip install -e .
cp .env.example .env
```

`.env` needs a free Copernicus Data Space account. All input data are open; `data/MANIFEST.csv` lists every downloaded file with its URL and checksum.

## Input data

```bash
python -m capcausal.data.http_fetch data/raw/dk_lbst <dataset> <licence> <urls>
python -m capcausal.data.zenodo 18670815 data/raw/iacs_inventory _Documentation_v1.3.zip DK_years_2023-2025.zip
python -m capcausal.iacs.dk_panel 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026
python scripts/build_organic.py 2018 2019 2020 2021 2022 2023 2024 2025
python -c "from capcausal.iacs.dk_soil import jb_raster; jb_raster()"
python -m capcausal.data.weather
```

## Run order

```bash
S="2019 2020 2021 2022 2023 2024 2025"
python scripts/run_s2_extract.py --seasons $S
python scripts/run_cdse_extract.py --seasons $S --parallel 2
python scripts/build_outcomes.py $S
python scripts/validate_outcomes.py --seasons $S --calib 2019 2020 2021
python scripts/build_alt_outcomes.py $S
python scripts/validate_aligned.py
python scripts/revisit_simulation.py
python scripts/revisit_realistic.py
python scripts/analyze_rotation.py
python scripts/analyze_glm6.py --seasons $S
python scripts/inference_robust.py
python scripts/measurement_error.py
python scripts/measurement_error_diff.py
python scripts/rd_diagnostics.py
python scripts/robustness_a.py
python scripts/replication_pc.py
```

Results are written to `results/`. The random seed is 42. The calibrated NDTI threshold is set in `configs/default.yaml`.

`docs/environment.md` lists software versions and openEO job identifiers. `docs/crop_groups.csv` gives the crop-code grouping by year.
