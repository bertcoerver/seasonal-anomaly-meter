"""Seasonal, day-of-season-aligned anomalies for gridded fluxes.

For every pixel this integrates a flux from the start of that pixel's own
growing season -- taken from a phenology product such as Copernicus LSP -- up to
a query date, and compares it against what the same pixel usually accumulates by
the same *day of season*. The baseline day can fall on a quite different
calendar date, which is the whole reason for pairing phenology with the flux.

You supply both inputs as xarray objects on a shared grid; this package does no
I/O of its own::

    baseline  = seasonal_baseline(flux, phenology)                  # rarely
    anomalies = seasonal_anomalies(flux, phenology, baseline, date) # every new period

``flux`` is a per-day rate on ``(time, y, x)`` and ``phenology`` carries
``SOSD``/``EOSD`` day-of-year codes on ``(season, year, y, x)``; see
:mod:`seasonal_anomaly_meter.inputs` for the full contract, and
``examples/wapor_sources.py`` for fetching them from WaPOR and Copernicus.

The flux grid is the example grid -- it is never resampled, and
:func:`align_phenology` moves the phenology onto it instead. Work on an area
that stays in one CRS; for WaPOR's UTM mosaicsets that is one tile.
"""

from importlib.metadata import PackageNotFoundError, version

from seasonal_anomaly_meter.accumulate import (
    FluxSeries,
    accumulate_by_slot,
    accumulate_to_date,
    interpolate_slot,
)
from seasonal_anomaly_meter.anomaly import compute_anomaly
from seasonal_anomaly_meter.baseline import build_baseline
from seasonal_anomaly_meter.calendar import DEKADAL, MONTHLY, infer_resolution
from seasonal_anomaly_meter.inputs import (
    PHENOLOGY_VARS,
    align_phenology,
    as_flux,
    check_phenology,
    check_same_grid,
)
from seasonal_anomaly_meter.io import (
    anomaly_encoding,
    baseline_encoding,
    check_packing_range,
)
from seasonal_anomaly_meter.pipeline import (
    required_flux_start,
    seasonal_anomalies,
    seasonal_baseline,
)
from seasonal_anomaly_meter.season import (
    MAX_POS,
    forward_fill_phenology,
    season_indices,
    select_season,
)

try:
    __version__ = version("seasonal-anomaly-meter")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    # pipeline -- start here
    "seasonal_baseline",
    "seasonal_anomalies",
    "required_flux_start",
    # inputs
    "align_phenology",
    "as_flux",
    "check_phenology",
    "check_same_grid",
    "PHENOLOGY_VARS",
    # methodology
    "season_indices",
    "select_season",
    "forward_fill_phenology",
    "build_baseline",
    "compute_anomaly",
    "MAX_POS",
    # accumulation kernel
    "FluxSeries",
    "accumulate_to_date",
    "accumulate_by_slot",
    "interpolate_slot",
    # calendar
    "DEKADAL",
    "MONTHLY",
    "infer_resolution",
    # io -- encodings only; the writing and reading is xr_utils.geozarr's
    # write_geozarr / open_geozarr
    "baseline_encoding",
    "anomaly_encoding",
    "check_packing_range",
    "__version__",
]
