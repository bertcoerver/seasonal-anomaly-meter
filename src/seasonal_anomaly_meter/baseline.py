"""Stage 1: the multi-year baseline, in accumulation space.

For every pixel, season label and day-of-season slot, this records what the
accumulated flux *usually* is by that point in the season::

    acc_mean (season, pos, y, x)   mean across years of the accumulated flux
    acc_std  (season, pos, y, x)   its standard deviation  -> z-scores
    acc_count(season, pos, y, x)   how many years contributed -> reliability

The statistics are of the **accumulation**, not of the per-period rate. That is
the substantive difference from the original implementation, which stored a
mean daily rate per slot and re-accumulated it on demand. A mean survives that
treatment, but a standard deviation does not: the spread of a seasonal total is
not recoverable from the spreads of its parts without their covariance, and
within a growing season that covariance is large. Storing accumulations also
means the live stage does two array lookups instead of re-integrating a season.

No separate mean-rate array is stored. Interpolating between two consecutive
slots is arithmetically identical to adding ``rate * days_into_slot`` (see
:func:`~seasonal_anomaly_meter.accumulate.interpolate_slot`), so such an array
would be redundant -- and at ~1.3 GB per tile, expensively so.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from seasonal_anomaly_meter.accumulate import FluxSeries, accumulate_by_slot
from seasonal_anomaly_meter.calendar import (
    DEKADAL,
    TemporalResolution,
    dates_to_abs_index,
)
from seasonal_anomaly_meter.inputs import check_same_grid
from seasonal_anomaly_meter.season import MAX_POS

__all__ = ["build_baseline", "DEFAULT_MIN_YEARS"]

#: Slots seen in fewer years than this are dropped. The original stored every
#: slot regardless, so a day-of-season reached in one year out of eight looked
#: exactly as authoritative as one reached in all eight.
DEFAULT_MIN_YEARS = 3


def build_baseline(
    flux: xr.DataArray,
    seasons: xr.Dataset,
    year_min: int,
    *,
    resolution: TemporalResolution = DEKADAL,
    max_pos: int = MAX_POS,
    min_years: int = DEFAULT_MIN_YEARS,
) -> xr.Dataset:
    """Accumulation statistics per season label and day-of-season slot.

    ``flux`` is ``(time, y, x)`` in units per day on the example grid;
    ``seasons`` is :func:`~seasonal_anomaly_meter.season.season_indices` output
    with dims ``(season, year, y, x)``. Both must already share the grid; see
    :func:`~seasonal_anomaly_meter.inputs.align_phenology`.

    The result is lazy when ``flux`` is. Chunk ``flux`` spatially but keep its
    ``time`` axis whole: the kernel needs the full series to build one prefix
    sum, and a chunked time axis would force a shuffle per slot.
    """
    period_index = dates_to_abs_index(flux["time"].values, year_min, resolution)
    check_same_grid(flux, seasons, "flux", "phenology")

    mean, std, count = xr.apply_ufunc(
        _baseline_kernel,
        flux,
        seasons["start_idx"],
        seasons["start_date"],
        seasons["end_idx"],
        input_core_dims=[["time"], ["season", "year"], ["season", "year"], ["season", "year"]],
        output_core_dims=[["season", "pos"], ["season", "pos"], ["season", "pos"]],
        dask="parallelized",
        output_dtypes=[np.float32, np.float32, np.uint8],
        dask_gufunc_kwargs={
            "output_sizes": {"season": seasons.sizes["season"], "pos": max_pos},
            "allow_rechunk": True,
        },
        kwargs={
            "period_index": period_index,
            "max_pos": max_pos,
            "year_min": year_min,
            "resolution": resolution,
            "min_years": min_years,
        },
    )

    out = xr.Dataset(
        {"acc_mean": mean, "acc_std": std, "acc_count": count},
        coords={
            "season": seasons["season"].values,
            "pos": np.arange(1, max_pos + 1, dtype=np.int16),
        },
    )
    # apply_ufunc leaves the core dims trailing; fix a canonical order so the
    # store's layout does not depend on how the inputs happened to be arranged.
    out = out.transpose("season", "pos", "y", "x")
    out["acc_mean"].attrs = {
        "long_name": "mean accumulated flux at day-of-season slot",
        "units": flux.attrs.get("units", "").replace("/day", ""),
    }
    out["acc_std"].attrs = {"long_name": "standard deviation across baseline years"}
    out["acc_count"].attrs = {
        "long_name": "baseline years contributing to this slot",
        "min_years": min_years,
    }
    out.attrs.update(
        year_min=year_min,
        temporal_resolution=resolution.name,
        min_years=min_years,
        baseline_years=f"{int(seasons['year'].min())}-{int(seasons['year'].max())}",
    )
    return out


def _baseline_kernel(
    rate: np.ndarray,
    start_idx: np.ndarray,
    start_date: np.ndarray,
    end_idx: np.ndarray,
    *,
    period_index: np.ndarray,
    max_pos: int,
    year_min: int,
    resolution: TemporalResolution,
    min_years: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-chunk statistics. Layout follows ``apply_ufunc``: core dims trail.

    ``rate`` is ``(*spatial, time)`` and the season fields are
    ``(*spatial, season, year)``; the outputs are ``(*spatial, season, pos)``.

    Years are folded in one at a time rather than stacked, so peak memory is
    one ``(max_pos, *spatial)`` array instead of one per year -- the difference
    between ~13 MB and ~100 MB for a 256x256 chunk.
    """
    spatial = rate.shape[:-1]
    n_season, n_year = start_idx.shape[-2:]

    series = FluxSeries.from_rate(
        np.moveaxis(rate, -1, 0), period_index, year_min, resolution
    )
    si = np.moveaxis(start_idx, (-2, -1), (0, 1))
    sd = np.moveaxis(start_date, (-2, -1), (0, 1))
    ei = np.moveaxis(end_idx, (-2, -1), (0, 1))

    mean = np.full((n_season, max_pos, *spatial), np.nan, dtype=np.float32)
    std = np.full((n_season, max_pos, *spatial), np.nan, dtype=np.float32)
    count = np.zeros((n_season, max_pos, *spatial), dtype=np.uint8)

    for s in range(n_season):
        # float64 accumulators: with ~8 years the variance is computed as
        # E[x^2] - E[x]^2, which loses relative precision when the mean dwarfs
        # the spread. float64 keeps that error ~1e-14 instead of ~1e-5.
        total = np.zeros((max_pos, *spatial), dtype=np.float64)
        total_sq = np.zeros((max_pos, *spatial), dtype=np.float64)
        n = np.zeros((max_pos, *spatial), dtype=np.int32)

        for y in range(n_year):
            acc = accumulate_by_slot(series, si[s, y], sd[s, y], ei[s, y], max_pos)
            present = np.isfinite(acc)
            if not present.any():
                continue
            values = np.where(present, acc, 0.0).astype(np.float64)
            total += values
            total_sq += values * values
            n += present

        enough = n >= max(min_years, 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.where(enough, total / np.maximum(n, 1), np.nan)
            # Sample (ddof=1) variance: these years are a sample of the
            # climatology, not the whole population.
            var = (total_sq - n * m * m) / np.maximum(n - 1, 1)
        mean[s] = m.astype(np.float32)
        std[s] = np.where(enough & (n > 1), np.sqrt(np.maximum(var, 0.0)), np.nan)
        count[s] = np.minimum(n, 255).astype(np.uint8)

    to_trailing = lambda a: np.moveaxis(a, (0, 1), (-2, -1))  # noqa: E731
    return to_trailing(mean), to_trailing(std), to_trailing(count)
