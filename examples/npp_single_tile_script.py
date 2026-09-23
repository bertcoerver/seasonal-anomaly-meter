import logging
import os
from datetime import date
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from sources import get_wapor_tile, load_flux, load_phenology
from xr_utils import (
    append_geozarr,
    make_zarr_cached,
    open_geozarr,
    store_path,
    write_geozarr,
)

from seasonal_anomaly_meter import (
    anomaly_encoding,
    baseline_encoding,
    required_flux_start,
    seasonal_anomalies,
    seasonal_baseline,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("rasterio").setLevel(logging.ERROR)
logger = logging.getLogger("npp_single_tile")

# ---- SETTINGS -- edit these ------------------------------------------

VARIABLE = "L1-UTM-NPP-D"
TILE = "36P"
# DATES = [str(x.date()) for x in pd.date_range("2018-01-01", "2026-08-01") if x.day in [1, 11, 21]]
DATES = [str(x.date()) for x in pd.date_range("2026-08-11", "2026-09-01") if x.day in [1, 11, 21]]
WORKDIR = Path(os.path.expanduser("~/Local/sam"))
SCALE_FACTOR = 1.0
CHUNK = 256

tile = get_wapor_tile(VARIABLE, TILE)
last_baseline_year = date.today().year - 1
flux_range = ("2018-01-01", max(DATES))
phenology_years = list(range(2018, max(int(d[:4]) for d in DATES) + 1))

GRID = {
    "epsg": tile.epsg,
    "x_range": tile.x_range,
    "y_range": tile.y_range,
    "pixel_size": tile.pixel_size,
}

# ---- The four stages, each cached to a keyed store --------------------

cached_flux = make_zarr_cached(
    load_flux,
    cache_file_prefix=f"{VARIABLE}_{tile.code}_flux",
    select=VARIABLE,
    cache_directory=WORKDIR,
)
cached_phenology = make_zarr_cached(
    load_phenology,
    cache_file_prefix=f"{tile.code}_phenology",
    select=["SOSD", "EOSD", "QA"],
    cache_directory=WORKDIR,
)
cached_baseline = make_zarr_cached(
    seasonal_baseline,
    cache_file_prefix=f"{VARIABLE}_{tile.code}_baseline",
    encoding=partial(baseline_encoding, scale_factor=SCALE_FACTOR),
    cache_directory=WORKDIR,
)
# Stage 2 is not wrapped in make_zarr_cached. That wrapper's premise is that a
# changed key means a rebuild, which is the opposite of what is wanted here:
# its store is appended to, one date at a time, as new flux is published.

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
    key={"years": phenology_years, "chunk": CHUNK, "grid": GRID},
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

# ---- Stage 2: the anomalies (append-only, rerun operationally) --------
#
# Re-running this after a new dekad is published computes that one date and
# appends it. Nothing already in the store is read back or recomputed --
# compute_anomaly carries no state between dates, so a date added later is
# identical to the same date computed in the first pass.
#
# The key deliberately holds neither DATES nor flux: both change every run
# (flux's own store name encodes its time_range), and keying on either would
# rename the store each time and rebuild every date. The baseline still keys
# it, so rebuilding Stage 1 correctly starts a fresh anomaly store.

anomaly_store = store_path(
    WORKDIR,
    f"{VARIABLE}_{tile.code}_anomaly",
    key={"baseline": baseline, "scale_factor": SCALE_FACTOR},
)

wanted = [np.datetime64(d, "ns") for d in DATES]
have = set(open_geozarr(anomaly_store)["time"].values) if anomaly_store.exists() else set()
todo = [d for d in wanted if d not in have]

if not todo:
    logger.info("%s is up to date (%d dates)", anomaly_store.name, len(have))
else:
    # Only as much flux as the seasons running on these dates actually need,
    # which for a single new dekad is a few months rather than the archive.
    flux_start = required_flux_start(phenology, todo)
    logger.info(
        "%d new date(s), %s to %s, from flux at %s",
        len(todo), todo[0], todo[-1], flux_start,
    )
    new = seasonal_anomalies(
        flux.sel(time=slice(flux_start, None)),
        phenology,
        baseline,
        todo,
        chunks=CHUNK,
    )
    if have:
        append_geozarr(new, anomaly_store)
    else:
        write_geozarr(
            new, anomaly_store, anomaly_encoding(new, scale_factor=SCALE_FACTOR)
        )

anomalies = open_geozarr(anomaly_store)
