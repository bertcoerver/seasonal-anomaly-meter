"""The two entry points: build a baseline once, then compute anomalies often.

The split is the point. Building the baseline reads years of flux and is
expensive; computing an anomaly reads the current season and the stored
baseline, and is cheap enough to re-run whenever a new period is published.
Stage 2 never touches the historical archive.

Both take the flux and the phenology as arrays you have already loaded, on a
shared grid, and neither knows or cares where they came from -- see
:mod:`seasonal_anomaly_meter.inputs` for that contract. Work on an area small
enough to stay in a single CRS, so nothing is ever reprojected: for WaPOR's UTM
mosaicsets that is one tile.
"""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr

from seasonal_anomaly_meter.anomaly import compute_anomaly
from seasonal_anomaly_meter.baseline import DEFAULT_MIN_YEARS, build_baseline
from seasonal_anomaly_meter.calendar import TemporalResolution, infer_resolution
from seasonal_anomaly_meter.inputs import (
    as_flux,
    check_phenology,
    check_same_grid,
)
from seasonal_anomaly_meter.season import (
    MAX_POS,
    forward_fill_phenology,
    season_indices,
)

logger = logging.getLogger(__name__)

__all__ = ["seasonal_baseline", "seasonal_anomalies"]


def seasonal_baseline(
    flux: xr.DataArray | xr.Dataset,
    phenology: xr.Dataset,
    *,
    variable: str | None = None,
    resolution: TemporalResolution | None = None,
    year_min: int | None = None,
    max_pos: int = MAX_POS,
    min_years: int = DEFAULT_MIN_YEARS,
    chunks: int | None = 256,
) -> xr.Dataset:
    """Stage 1: accumulation statistics over the years the inputs cover.

    ``flux`` is a per-day rate on ``(time, y, x)``; ``phenology`` carries
    ``SOSD``/``EOSD`` (and optionally ``QA``) on ``(season, year, y, x)``, on
    the same pixels as the flux. Give the flux every year you want in the
    baseline, and the phenology at least those years.

    Returns a lazy Dataset of ``acc_mean``/``acc_std``/``acc_count``; write it
    with :func:`~seasonal_anomaly_meter.io.write_zarr`. ``year_min`` is recorded in
    its attributes and reused by :func:`seasonal_anomalies`, so the two stages
    cannot drift onto different period axes.

    ``resolution`` is inferred from the time axis when omitted. ``chunks`` sets
    the spatial chunk size; the ``time`` axis is always kept whole, because the
    kernel builds one prefix sum over the full series and a split time axis
    would force a shuffle for every day-of-season slot. Pass ``chunks=None`` to
    leave the inputs chunked however you handed them over.
    """
    flux = as_flux(flux, variable)
    check_phenology(phenology, flux)
    resolution = resolution or infer_resolution(flux["time"].values)
    year_min = year_min if year_min is not None else _first_year(flux)

    first, last = _year_span(flux)
    _check_overlapping_years(phenology, first, last, min_years)

    flux, phenology = _chunk(flux, phenology, chunks)
    seasons = season_indices(phenology, year_min, resolution)

    logger.info(
        "baseline over %d-%d (%s), %d x %d px, %s flux steps",
        first,
        last,
        resolution.name,
        flux.sizes["y"],
        flux.sizes["x"],
        flux.sizes["time"],
    )
    baseline = build_baseline(
        flux,
        seasons,
        year_min,
        resolution=resolution,
        max_pos=max_pos,
        min_years=min_years,
    )
    rate_units = flux.attrs.get("units", "")
    baseline.attrs.update(
        variable=variable or flux.name or "",
        # The store holds accumulations, so its units are the flux's with the
        # per-day part removed -- gC/m2, not gC/m2/day.
        units=rate_units.replace("/day", ""),
        rate_units=rate_units,
    )
    for key in ("crs_wkt", "epsg"):
        if key in flux.attrs:
            baseline.attrs[key] = flux.attrs[key]
    return baseline


def seasonal_anomalies(
    flux: xr.DataArray | xr.Dataset,
    phenology: xr.Dataset,
    baseline: xr.Dataset,
    dates,
    *,
    variable: str | None = None,
    resolution: TemporalResolution | None = None,
    forward_fill: bool = True,
    chunks: int | None = 256,
) -> xr.Dataset:
    """Stage 2: accumulated flux and its anomalies at each of ``dates``.

    ``baseline`` is a Stage 1 result, typically reopened from zarr with
    :func:`~seasonal_anomaly_meter.io.open_zarr`. It fixes ``year_min``, and its
    grid is checked against the flux -- a baseline built for a different area
    fails here rather than producing plausible nonsense.

    ``flux`` need only reach back far enough to contain the start of every
    season active on ``dates``, not the whole archive; Copernicus seasons run up
    to roughly 500 days, so two years is a safe span. That is what keeps an
    operational rerun cheap.

    With ``forward_fill`` (the default) the phenology is extended to the latest
    query year by repeating its last year: Copernicus publishes a year's seasons
    a year or more in arrears, so an operational run is always asking about a
    season whose parameters are not yet out. The assumed years are flagged with
    a ``forward_filled`` coordinate.
    """
    flux = as_flux(flux, variable)
    check_phenology(phenology, flux)
    check_same_grid(flux, baseline, "flux", "baseline")

    resolution = resolution or infer_resolution(flux["time"].values)
    try:
        year_min = int(baseline.attrs["year_min"])
    except KeyError:
        raise ValueError(
            "the baseline carries no 'year_min' attribute, so its period axis "
            "cannot be matched to the flux. Rebuild it with seasonal_baseline, "
            "or open the store with io.open_zarr, which keeps attributes."
        ) from None

    dates = [np.datetime64(d, "D") for d in np.atleast_1d(dates)]
    query_year = max(int(str(d)[:4]) for d in dates)
    if forward_fill:
        phenology = forward_fill_phenology(phenology, query_year)

    flux, phenology = _chunk(flux, phenology, chunks)
    seasons = season_indices(phenology, year_min, resolution)

    # The stored coordinates are float32 after a zarr round-trip while the flux
    # keeps float64, so they compare equal but do not align. check_same_grid has
    # already established they describe the same pixels; adopt the flux's copy
    # so xarray does not treat the difference as a reindex.
    baseline = baseline.assign_coords(y=flux["y"], x=flux["x"])

    logger.info("%d query date(s), %s to %s", len(dates), dates[0], dates[-1])
    per_date = [
        compute_anomaly(flux, seasons, baseline, date, year_min, resolution=resolution)
        for date in dates
    ]
    out = xr.concat(per_date, dim="time")
    out.attrs.update(
        variable=baseline.attrs.get("variable", variable or flux.name or ""),
        units=flux.attrs.get("units", ""),
        baseline_years=baseline.attrs.get("baseline_years", ""),
        year_min=year_min,
    )
    return out


def _chunk(
    flux: xr.DataArray, phenology: xr.Dataset, chunks: int | None
) -> tuple[xr.DataArray, xr.Dataset]:
    """Put both inputs on matching spatial chunks, with time and season whole.

    Leaving this to the caller is how the two stages end up on chunk grids that
    do not line up, which xarray resolves by rechunking mid-graph -- correct,
    but it turns a windowed read into a full one.
    """
    if chunks is None:
        return flux, phenology
    flux = flux.chunk({"time": -1, "y": chunks, "x": chunks})
    phenology = phenology.chunk({"y": chunks, "x": chunks, "season": -1, "year": -1})
    return flux, phenology


def _check_overlapping_years(
    phenology: xr.Dataset, first: int, last: int, min_years: int
) -> None:
    """Raise when too few phenology years overlap the flux to build a baseline.

    A season contributes to the baseline only where both inputs cover its year,
    so the years that matter are the intersection, not the phenology's own axis.
    Below ``min_years`` of them every pixel fails the count test and the store
    comes out empty -- a silent, expensive nothing, and one that looks like a
    real baseline until someone reads its values. Worth naming up front: the
    usual cause is a phenology source that quietly returned fewer years than it
    was asked for.
    """
    years = [int(y) for y in np.atleast_1d(phenology["year"].values)]
    overlap = sorted(y for y in years if first <= y <= last)
    if len(overlap) < min_years:
        raise ValueError(
            f"the phenology covers {overlap or 'no year'} within the flux's "
            f"{first}-{last}, which is fewer than the {min_years} years "
            "min_years requires, so every pixel's baseline would be empty. "
            f"The phenology's own year axis is {years}. Supply more years, or "
            "lower min_years if that few is genuinely what you want."
        )


def _first_year(flux: xr.DataArray) -> int:
    return int(str(np.datetime64(flux["time"].values.min(), "D"))[:4])


def _year_span(flux: xr.DataArray) -> tuple[int, int]:
    times = flux["time"].values
    return (
        int(str(np.datetime64(times.min(), "D"))[:4]),
        int(str(np.datetime64(times.max(), "D"))[:4]),
    )
