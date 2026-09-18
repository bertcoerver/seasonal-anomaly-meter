"""End-to-end run for one whole WaPOR UTM tile, using NPP -- no command line.

Same run as ``npp_single_tile.py``, but the settings live in the SETTINGS block
below instead of in argparse, so this file can be executed top to bottom or
pasted into a notebook.

The two inputs come from WaPOR and Copernicus (``wapor_sources``, the only code
in this repository that uses ``lazy_dino``) and are fetched **once**, for the
full span both stages need, and cached as zarr on the tile's own grid. Every
later run -- and both stages of this one -- reads that store instead of the
remote archives.

The package itself takes the flux and the phenology as arrays and does no I/O,
so swapping in a different source means replacing ``load_flux``/
``load_phenology`` below and nothing else.
"""

from __future__ import annotations

import logging
import os
from datetime import date
from pathlib import Path

import pandas as pd
from wapor_sources import load_flux, load_phenology
from wapor_tiles import get_tile

from seasonal_anomaly_meter import (
    anomaly_encoding,
    baseline_encoding,
    check_packing_range,
    open_zarr,
    seasonal_anomalies,
    seasonal_baseline,
    write_zarr,
)

# ---- SETTINGS -- edit these ------------------------------------------

VARIABLE = "L1-UTM-AETI-D"
#: 36Q covers the Gezira scheme in Sudan -- irrigated, strongly seasonal, and
#: the area the original drought-depth work was tested on.
TILE = "36P"
DATES = [str(x.date()) for x in pd.date_range("2018-01-01", "2026-08-01") if x.day in [1, 11, 21]]

OUT_DIR = Path(os.path.expanduser("~/Local/sam"))

SCALE_FACTOR = 1.0
#: Spatial chunk, matching the one the pipeline uses, so nothing is reshuffled.
CHUNK = 256

#: WaPOR starts in 2018, so that is the earliest a baseline year can be. The
#: Copernicus phenology reaches back to 2014, but a baseline needs both.
WAPOR_START_YEAR = 2018

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("rasterio").setLevel(logging.ERROR)

tile = get_tile(VARIABLE, TILE)
OUT_DIR.mkdir(parents=True, exist_ok=True)
#: The current year is excluded from the baseline: a season still in progress
#: would contribute a truncated accumulation and drag every later slot down.
last_baseline_year = date.today().year - 1


# ---- Stage 0: the inputs (fetched once, then cached) ------------------

inputs_store = OUT_DIR / f"{VARIABLE}_{tile.code}_inputs.zarr"

if not inputs_store.exists():
    # The widest span either stage needs: the whole archive for the baseline,
    # up to the last query date for the anomalies.
    flux = load_flux(VARIABLE, tile, (f"{WAPOR_START_YEAR}-01-01", max(DATES)))
    # Phenology from the baseline anchor year, so season_indices places every
    # season on the same period axis in both stages.
    phenology = load_phenology(
        tile, range(WAPOR_START_YEAR, max(int(d[:4]) for d in DATES) + 1), example=flux
    )
    inputs = phenology.assign({VARIABLE: flux}).chunk(
        {"time": -1, "y": CHUNK, "x": CHUNK, "season": -1, "year": -1}
    )
    write_zarr(inputs, inputs_store, {})

inputs = open_zarr(inputs_store)
flux, phenology = inputs[VARIABLE], inputs[["SOSD", "EOSD", "QA"]]
print(f"tile {tile.code}  EPSG:{tile.epsg}  {tile.shape[0]}x{tile.shape[1]} px  {inputs_store}")


# ---- Stage 1: the baseline (slow, rebuilt rarely) ---------------------

baseline = seasonal_baseline(
    flux.sel(time=slice(None, f"{last_baseline_year}-12-31")),
    phenology,
    variable=VARIABLE,
).compute()

check_packing_range(baseline, SCALE_FACTOR)
baseline_store = OUT_DIR / f"{VARIABLE}_{tile.code}_baseline.zarr"
write_zarr(baseline, baseline_store, baseline_encoding(baseline, scale_factor=SCALE_FACTOR))
print(f"wrote {baseline_store}")


# ---- Stage 2: the anomalies (fast, rerun operationally) ---------------

# Read the baseline back rather than reusing the materialised one above. It is
# (season, pos, y, x) -- gigabytes for a whole tile -- and compute_anomaly puts
# it into the graph once per query date, so an in-memory array is embedded a few
# hundred times over and building the graph alone exhausts the machine. The
# store is chunked with season and pos whole, which is what the curve lookup
# wants, so reopening costs nothing and each chunk is read only as it is needed.
baseline = open_zarr(baseline_store)

# A Copernicus season can run ~500 days, so reach back far enough to contain
# the start of every season active on the query dates -- no further, the
# accumulation runs over whatever series it is given.
start = f"{min(int(d[:4]) for d in DATES) - 2}-01-01"
# Left lazy on purpose: this is one field per query date per variable, and for
# a whole tile over years of dates the materialised result runs to tens of GB.
# to_zarr walks the graph chunk by chunk, and anomaly_encoding already writes
# one date per file, so nothing larger than a chunk is ever held at once.
anomalies = seasonal_anomalies(
    flux.sel(time=slice(start, None)), phenology, baseline, DATES
)

anomaly_store = OUT_DIR / f"{VARIABLE}_{tile.code}_anomaly.zarr"
write_zarr(anomalies, anomaly_store, anomaly_encoding(anomalies))
print(f"wrote {anomaly_store}")
