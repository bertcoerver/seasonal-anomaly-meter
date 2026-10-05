"""npp_single_tile_anomalies.py, with cubit fetching, aligning and caching.

One cubit plan, reading another's output:

* the **baseline** is the ``derived:baseline`` slot the plan ``npp_baseline``
  wrote -- npp_single_tile_baseline_cubit.py -- opened from the workdir by
  name. It must already be there; this script does not build a baseline.
* **npp_anomalies** -- ``npp_cubit_anomalies_config.json``: the flux over only
  the current seasons, the phenology, and that baseline as the upstream
  product ``{"upstream": "baseline"}``. ``seasonal_anomalies`` runs over that
  cube as its ``derived:anomalies`` slot.

The flux window comes from the phenology (``required_flux_start``), and the
phenology is static to cubit, so the stores the baseline run aligned already
hold it: reading it is free, and it decides the plan's period.

``derived:anomalies`` is keyed on what its lambda names -- DATES among them --
so a run with other dates writes a new store rather than appending to the last
one.

The URL printed before the build is a live page that follows it.
"""

import logging
from pathlib import Path

import cubit
import pandas as pd
from cubit.external.wapor import WaporTile

import seasonal_anomaly_meter as sam

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("npp_single_tile_anomalies_cubit")

# ---- 1. SETTINGS -- edit these ---------------------------------------
#
# The baseline is built by npp_single_tile_baseline_cubit.py; this script only
# reads it. VARIABLE, TILE, AOI and WORKDIR must match that script, since the
# baseline is the one written there for this extent -- the last one, if it
# was built more than once.

VARIABLE = "L1-UTM-NPP-D"
TILE = "36P"
DATES = [str(x.date()) for x in pd.date_range("2026-08-11", "2026-09-01") if x.day in [1, 11, 21]]
AOI = None          # e.g. {"x_range": (33.0, 33.3), "y_range": (14.0, 14.3)}
WORKDIR = Path("~/Local/sam_cubit").expanduser()
CHUNK = 256

# The phenology years DATES need: only the seasons running on DATES matter, and
# one labelled with the previous year can still be running. Years not yet
# published are forward-filled.
PHENOLOGY_YEARS = slice(int(min(DATES)[:4]) - 1, int(max(DATES)[:4]))

# ---- 2. THE PLAN -----------------------------------------------------
#
# The extent is a WaPOR tile, or the AOI inside it: WaPOR is read from that
# tile alone, on the tile's own grid.

config = Path(__file__).with_name("npp_cubit_anomalies_config.json")
extent = WaporTile.lookup(VARIABLE, TILE).within(AOI)
settings = {
    "name": "npp_anomalies",
    "extent": extent,
    "workdir": WORKDIR,
    "cache": ["raw", "aligned"],
    "upstream": [cubit.open_derived(WORKDIR, "npp_baseline", "baseline", extent=extent)],
}

# The period starts where the phenology says: only as much flux as the seasons
# running on DATES need, which for a single new dekad is a few months rather
# than the archive. The phenology does not depend on the period, so a first
# plan over DATES alone is enough to read it.
phenology = cubit.plan(config, period=(min(DATES), max(DATES)), **settings).cube[
    ["SOSD", "EOSD", "QA"]
].sel(year=PHENOLOGY_YEARS)
flux_start = sam.required_flux_start(phenology, DATES)
logger.info("%d date(s), %s to %s, from flux at %s",
            len(DATES), DATES[0], DATES[-1], flux_start)

cubit.live()                            # from here on: the plan below
plan = cubit.plan(config, period=(str(flux_start), max(DATES)), **settings)

# ---- 3. THE MODEL ----------------------------------------------------
#
# The package's own call over the cube: the baseline is the cube's acc_*
# variables, read from upstream.

anomalies = plan.derive(
    "anomalies",
    lambda cube: sam.seasonal_anomalies(
        cube["npp"],
        cube[["SOSD", "EOSD", "QA"]].sel(year=PHENOLOGY_YEARS),
        cube[["acc_mean", "acc_std", "acc_count"]],
        DATES,
        chunks=CHUNK,
    ),
    variables=["DOS", "season", "acc", "acc_baseline", "anomaly_abs", "anomaly_rel", "anomaly_z"],
)
print(plan)

plan.materialize()

logger.info("anomalies ready: %s, %s", anomalies.path(resolve=True), dict(anomalies.dataset.sizes))
# Again, now that the stores exist, so the diagrams show what is cached.
plan.to_html(WORKDIR / "npp_anomalies_plan.html")
plan.to_html(WORKDIR / "npp_anomalies_plan_full.html", expand_upstream=True)
