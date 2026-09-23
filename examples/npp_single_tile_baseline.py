import logging
import os
from functools import partial
from pathlib import Path

from sources import get_wapor_tile, load_flux, load_phenology
from xr_utils import STORE_ATTR, make_zarr_cached

from seasonal_anomaly_meter import baseline_encoding, seasonal_baseline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("rasterio").setLevel(logging.ERROR)
logger = logging.getLogger("npp_single_tile_baseline")

# ---- SETTINGS -- edit these ------------------------------------------
#
# npp_single_tile_anomalies.py derives this baseline's store name from the same
# settings, so keep VARIABLE, TILE, BASELINE_YEARS, WORKDIR, SCALE_FACTOR and
# CHUNK identical in both scripts.

VARIABLE = "L1-UTM-NPP-D"
TILE = "36P"
BASELINE_YEARS = (2018, 2025)
WORKDIR = Path(os.path.expanduser("~/Local/sam"))
SCALE_FACTOR = 1.0
CHUNK = 256

tile = get_wapor_tile(VARIABLE, TILE)
first_year, last_year = BASELINE_YEARS
flux_range = (f"{first_year}-01-01", f"{last_year}-12-31")
phenology_years = list(range(first_year, last_year + 1))

GRID = {
    "epsg": tile.epsg,
    "x_range": tile.x_range,
    "y_range": tile.y_range,
    "pixel_size": tile.pixel_size,
}

# ---- The three stages, each cached to a keyed store -------------------

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
    flux,
    phenology,
    variable=VARIABLE,
    key={
        "flux": flux,
        "phenology": phenology,
        "last_year": last_year,
        "scale_factor": SCALE_FACTOR,
    },
)

logger.info("baseline %d-%d ready: %s", first_year, last_year, baseline.attrs[STORE_ATTR])
