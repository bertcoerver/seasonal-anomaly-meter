import logging
import os
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
    required_flux_start,
    seasonal_anomalies,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("rasterio").setLevel(logging.ERROR)
logger = logging.getLogger("npp_single_tile_anomalies")

# ---- SETTINGS -- edit these ------------------------------------------
#
# The baseline is built by npp_single_tile_baseline.py; this script only opens
# it. VARIABLE, TILE, BASELINE_YEARS, WORKDIR, SCALE_FACTOR and CHUNK must match
# that script, since together they name the baseline's store.

VARIABLE = "L1-UTM-NPP-D"
TILE = "36P"
BASELINE_YEARS = (2018, 2025)
DATES = [str(x.date()) for x in pd.date_range("2026-08-11", "2026-09-01") if x.day in [1, 11, 21]]
WORKDIR = Path(os.path.expanduser("~/Local/sam"))
SCALE_FACTOR = 1.0
CHUNK = 256

tile = get_wapor_tile(VARIABLE, TILE)
first_year, last_year = BASELINE_YEARS
# Only the seasons running on DATES matter, and one labelled with the previous
# year can still be running. Years not yet published are forward-filled.
query_years = [int(d[:4]) for d in DATES]
phenology_years = list(range(min(query_years) - 1, max(query_years) + 1))

GRID = {
    "epsg": tile.epsg,
    "x_range": tile.x_range,
    "y_range": tile.y_range,
    "pixel_size": tile.pixel_size,
}

# ---- Stage 1: open the baseline ---------------------------------------
#
# Rebuilds the store name npp_single_tile_baseline.py wrote to. Its key names
# the upstream flux and phenology stores, which store_key reduces to their
# store names, so the same names give the same digest here.

baseline_flux_range = (f"{first_year}-01-01", f"{last_year}-12-31")
baseline_phenology_years = list(range(first_year, last_year + 1))

baseline_flux_store = store_path(
    WORKDIR,
    f"{VARIABLE}_{tile.code}_flux",
    key={"time_range": baseline_flux_range, "chunk": CHUNK},
)
baseline_phenology_store = store_path(
    WORKDIR,
    f"{tile.code}_phenology",
    key={"years": baseline_phenology_years, "chunk": CHUNK, "grid": GRID},
)
baseline_store = store_path(
    WORKDIR,
    f"{VARIABLE}_{tile.code}_baseline",
    key={
        "flux": baseline_flux_store.name,
        "phenology": baseline_phenology_store.name,
        "last_year": last_year,
        "scale_factor": SCALE_FACTOR,
    },
)

if not baseline_store.exists():
    raise FileNotFoundError(
        f"no baseline at {baseline_store}. Run npp_single_tile_baseline.py "
        "first, with the same settings as this script."
    )
baseline = open_geozarr(baseline_store)
logger.info("baseline %d-%d: %s", first_year, last_year, baseline_store.name)

# ---- Stage 0: the inputs for the query dates (fetched, then cached) ---

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

# The phenology is warped onto the flux grid, but the flux window itself is
# decided by the phenology (required_flux_start, below). A lazy, uncached
# single-dekad flux breaks that loop: the warp reads only its grid, so no flux
# pixels are fetched for it.
grid_template = load_flux(VARIABLE, tile, (max(DATES), max(DATES)), chunk=CHUNK)

phenology = cached_phenology(
    tile,
    phenology_years,
    example=grid_template,
    chunk=CHUNK,
    key={"years": phenology_years, "chunk": CHUNK, "grid": GRID},
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
# it, so rebuilding the baseline correctly starts a fresh anomaly store.

anomaly_store = store_path(
    WORKDIR,
    f"{VARIABLE}_{tile.code}_anomaly",
    key={"baseline": baseline_store.name, "scale_factor": SCALE_FACTOR},
)

wanted = [np.datetime64(d, "ns") for d in DATES]
have = set(open_geozarr(anomaly_store)["time"].values) if anomaly_store.exists() else set()
todo = [d for d in wanted if d not in have]

if not todo:
    logger.info("%s is up to date (%d dates)", anomaly_store.name, len(have))
else:
    # Fetch only as much flux as the seasons running on these dates actually
    # need, which for a single new dekad is a few months rather than the archive.
    flux_start = required_flux_start(phenology, todo)
    flux_range = (
        str(np.datetime64(flux_start, "D")),
        str(np.datetime64(max(todo), "D")),
    )
    logger.info(
        "%d new date(s), %s to %s, from flux at %s",
        len(todo), todo[0], todo[-1], flux_start,
    )
    flux = cached_flux(
        VARIABLE,
        tile,
        flux_range,
        chunk=CHUNK,
        key={"time_range": flux_range, "chunk": CHUNK},
    )
    new = seasonal_anomalies(
        flux,
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
