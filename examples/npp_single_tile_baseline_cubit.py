"""npp_single_tile_baseline.py, with cubit fetching, aligning and caching.

One cubit plan (``npp_cubit_baseline_config.json``) holds WaPOR NPP on the
tile's native UTM grid -- read from that tile alone (``L1-UTM-NPP-D.<TILE>``),
none of its neighbours -- and Copernicus phenology warped onto it, nearest
neighbour. ``seasonal_baseline`` runs over that cube as the plan's
``derived:baseline`` slot.

``plan.materialize()`` writes the raw and aligned stores and then the baseline,
each keyed on what it was built from; a second run reads them all. The
phenology is static to cubit -- its years are a dimension, not a time axis --
so its stores do not depend on the period, and the anomalies script reuses
them.

Set ``AOI`` to a small box inside the tile for a quick run. The URL printed at
the start is a live page that follows the build.
"""

import logging
from pathlib import Path

import cubit
from cubit.external.wapor import WaporTile

import seasonal_anomaly_meter as sam

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("npp_single_tile_baseline_cubit")

# ---- 1. SETTINGS -- edit these ---------------------------------------
#
# npp_single_tile_anomalies_cubit.py opens the baseline this writes by the
# plan's name, the slot's name and the extent, so keep VARIABLE, TILE, AOI and
# WORKDIR identical in both scripts.

VARIABLE = "L1-UTM-NPP-D"
TILE = "36P"
BASELINE_YEARS = (2018, 2025)
AOI = None          # e.g. {"x_range": (33.0, 33.3), "y_range": (14.0, 14.3)}
WORKDIR = Path("~/Local/sam_cubit").expanduser()
CHUNK = 256

# How a raw store is written; neither changes what is in it. The raw WaPOR
# store is written RESUME_BATCH time steps per compute -- one progress bar, and
# the most an interruption can lose. 36 is one year of dekads.
RESUME_BATCH = 36
PROGRESS = True     # the terminal progress bars and per-slot byte reports

cubit.live()

# ---- 2. THE PLAN -----------------------------------------------------
#
# The extent is a WaPOR tile, or the AOI inside it: WaPOR is read from that
# tile alone, so its grid is exactly the tile's. Everything else sees the area
# -- without an AOI the tile's footprint, a polygon in lon/lat -- which is what
# the phenology is read and warped over, and what the cube is masked to.
#
# The raw stores are resumable, which matters for eight years of a tile; the
# aligned ones are what the cube is read from. Except aligned:npp: on its own
# tile's grid and dekads it is the raw WaPOR store renamed, so cubit leaves it
# a view over that one rather than write the flux twice. Not "cube": it would
# be a second copy of the flux, and nothing reads it that the stores below it
# do not serve as well.

plan = cubit.plan(
    Path(__file__).with_name("npp_cubit_baseline_config.json"),
    name="npp_baseline",
    extent=WaporTile.lookup(VARIABLE, TILE).within(AOI),
    period=(f"{BASELINE_YEARS[0]}-01-01", f"{BASELINE_YEARS[1]}-12-31"),
    workdir=WORKDIR,
    cache=["raw", "aligned"],
    options={"resume_batch": RESUME_BATCH, "progress": PROGRESS},
)

# ---- 3. THE MODEL ----------------------------------------------------
#
# The cube already speaks the package's names -- the config keys are npp, SOSD,
# EOSD and QA -- so the model is the package's own call, with the phenology cut
# to the baseline years. cubit keys the slot on the lambda's source, on
# BASELINE_YEARS and CHUNK, and on the package's version. static=True because
# the baseline has no time axis -- its dims are (season, pos, y, x) -- which is
# what lets the anomalies config read it with "time_dim": null.

baseline = plan.derive(
    "baseline",
    lambda cube: sam.seasonal_baseline(
        cube["npp"],
        cube[["SOSD", "EOSD", "QA"]].sel(year=slice(*BASELINE_YEARS)),
        chunks=CHUNK,
    ),
    variables=["acc_mean", "acc_std", "acc_count"],
    static=True,
)

plan.materialize()

logger.info(
    "baseline %d-%d ready: %s, %s",
    *BASELINE_YEARS, baseline.path(resolve=True), dict(baseline.dataset["acc_mean"].sizes),
)
# Again, now that the stores exist, so the diagram shows what is cached.
plan.to_html(WORKDIR / "npp_baseline_plan.html")
