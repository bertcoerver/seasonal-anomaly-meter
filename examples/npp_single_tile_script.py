import logging
import os
from datetime import date
from functools import partial
from pathlib import Path

import pandas as pd
from sources import get_wapor_tile, load_flux, load_phenology
from xr_utils import make_zarr_cached

from seasonal_anomaly_meter import (
    anomaly_encoding,
    baseline_encoding,
    seasonal_anomalies,
    seasonal_baseline,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("rasterio").setLevel(logging.ERROR)

# ---- SETTINGS -- edit these ------------------------------------------

VARIABLE = "L1-UTM-AETI-D"
TILE = "36P"
DATES = [str(x.date()) for x in pd.date_range("2018-01-01", "2026-08-01") if x.day in [1, 11, 21]]
WORKDIR = Path(os.path.expanduser("~/Local/sam"))
SCALE_FACTOR = 1.0
CHUNK = 256

tile = get_wapor_tile(VARIABLE, TILE)
last_baseline_year = date.today().year - 1
flux_range = ("2018-01-01", max(DATES))
phenology_years = list(range(2018, max(int(d[:4]) for d in DATES) + 1))

# ---- The four stages, each cached to a keyed store --------------------

cached_flux = make_zarr_cached(
    load_flux,
    cache_file_prefix=f"{VARIABLE}_{tile.code}_flux",
    select=VARIABLE,
    cache_directory=WORKDIR,
)
cached_phenology = make_zarr_cached(
    load_phenology,
    cache_file_prefix=f"{VARIABLE}_{tile.code}_phenology",
    select=["SOSD", "EOSD", "QA"],
    cache_directory=WORKDIR,
)
cached_baseline = make_zarr_cached(
    seasonal_baseline,
    cache_file_prefix=f"{VARIABLE}_{tile.code}_baseline",
    encoding=partial(baseline_encoding, scale_factor=SCALE_FACTOR),
    cache_directory=WORKDIR,
)
cached_anomalies = make_zarr_cached(
    seasonal_anomalies,
    cache_file_prefix=f"{VARIABLE}_{tile.code}_anomaly",
    encoding=partial(anomaly_encoding, scale_factor=SCALE_FACTOR),
    cache_directory=WORKDIR,
)

# ---- Stage 0: the inputs (fetched once, then cached) ------------------

flux = cached_flux(
    VARIABLE,
    tile,
    flux_range,
    chunk=CHUNK,
    key={"time_range": flux_range, "chunk": CHUNK},
)

phenology = cached_phenology(
    tile,
    phenology_years,
    example=flux,
    chunk=CHUNK,
    key={"years": phenology_years, "chunk": CHUNK, "flux": flux},
)

# ---- Stage 1: the baseline (slow, rebuilt rarely) ---------------------

baseline = cached_baseline(
    flux.sel(time=slice(None, f"{last_baseline_year}-12-31")),
    phenology,
    variable=VARIABLE,
    key={
        "flux": flux,
        "phenology": phenology,
        "last_year": last_baseline_year,
        "scale_factor": SCALE_FACTOR,
    },
)

# ---- Stage 2: the anomalies (fast, rerun operationally) ---------------

start = f"{min(int(d[:4]) for d in DATES) - 2}-01-01"

anomalies = cached_anomalies(
    flux.sel(time=slice(start, None)),
    phenology,
    baseline,
    DATES,
    key={
        "baseline": baseline,
        "flux": flux,
        "dates": DATES,
        "scale_factor": SCALE_FACTOR,
    },
)
