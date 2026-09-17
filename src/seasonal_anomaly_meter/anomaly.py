"""Stage 2: where each pixel stands against its own baseline, right now.

For a query date this produces, per pixel, how far into its season it is, how
much flux it has accumulated since that season began, what the baseline says it
should have accumulated by the same *day of season*, and the gap between them
in three forms.

The comparison is day-of-season aligned, not day-of-year aligned: a pixel 40
days into its season is compared with the baseline's day 40, which in a given
year may fall on a quite different calendar date. That is the whole point of
pairing phenology with the flux, and it is why the baseline is indexed by slot.

This stage is the operational one -- it runs whenever a new dekad lands -- so it
reads the stored baseline and never touches the historical archive.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from seasonal_anomaly_meter.accumulate import (
    FluxSeries,
    accumulate_to_date,
    interpolate_slot,
)
from seasonal_anomaly_meter.calendar import (
    DEKADAL,
    TemporalResolution,
    dates_to_abs_index,
    period_bounds,
)
from seasonal_anomaly_meter.season import select_season

__all__ = ["compute_anomaly", "BASELINE_FLOOR"]

#: Baseline accumulations at or below this are too small to compare against.
#: A few days into a season a pixel has accumulated almost nothing, so both the
#: percentage and the z-score blow up: on real Gezira NPP the pixels with
#: ``|z| > 10`` had a median baseline of 0.19 gC/m2 against 17.6 for everything
#: else, and a handful reached ``z = -126`` purely because the denominator was
#: negligible. Note that a *relative* guard (std as a fraction of the mean) does
#: not catch these -- their spread is proportionate, it is the magnitude that is
#: meaningless -- so the floor is on the baseline itself, in the variable's own
#: accumulated units.
BASELINE_FLOOR = 1.0


def compute_anomaly(
    flux: xr.DataArray,
    seasons: xr.Dataset,
    baseline: xr.Dataset,
    query_date,
    year_min: int,
    *,
    resolution: TemporalResolution = DEKADAL,
    baseline_floor: float = BASELINE_FLOOR,
) -> xr.Dataset:
    """Accumulated flux and its anomalies at ``query_date``.

    ``flux`` need only cover the current season (plus the periods it started
    in), not the whole archive -- that is what makes this cheap to re-run.
    ``baseline`` is :func:`~seasonal_anomaly_meter.baseline.build_baseline`
    output; ``seasons`` is
    :func:`~seasonal_anomaly_meter.season.season_indices` output, forward-filled
    to the query year.

    Returns ``DOS``, ``season``, ``acc``, ``acc_baseline``, ``anomaly_abs``,
    ``anomaly_rel`` and ``anomaly_z``. Everything is NaN (or 0 for the two
    integer masks) outside a season.
    """
    query_date = np.datetime64(query_date, "D")
    query_idx = int(dates_to_abs_index([query_date], year_min, resolution)[0])

    picked = select_season(seasons, query_date)
    start_idx = picked["start_idx"]

    # Where in its season the query sits: 1-based slot, plus how far into that
    # slot, so the baseline curve can be read between two stored points.
    slot = (query_idx - start_idx + 1).astype("float32")
    fraction = _slot_fraction(
        picked["start_date"], query_date, query_idx, year_min, resolution
    )

    period_index = dates_to_abs_index(flux["time"].values, year_min, resolution)
    acc = xr.apply_ufunc(
        _accumulate_kernel,
        flux,
        start_idx,
        picked["start_date"],
        input_core_dims=[["time"], [], []],
        dask="parallelized",
        output_dtypes=[np.float32],
        kwargs={
            "period_index": period_index,
            "query_date": query_date,
            "query_idx": query_idx,
            "year_min": year_min,
            "resolution": resolution,
        },
    )

    # Each pixel reads the curve for the season label it was assigned, at its
    # own fractional slot. Both the season pick and the slot lookup happen
    # inside the kernel: expressing them as xarray indexing would rely on
    # apply_ufunc's broadcasting to line up a per-pixel label with a per-season
    # curve, which is exactly the kind of implicit alignment that goes wrong
    # quietly.
    labels = baseline["season"].values
    acc_baseline = _read_curve(baseline["acc_mean"], slot, picked["season"], fraction, labels)
    baseline_std = _read_curve(baseline["acc_std"], slot, picked["season"], fraction, labels)

    in_season = picked["in_season"]
    acc = acc.where(in_season)
    acc_baseline = acc_baseline.where(in_season)
    baseline_std = baseline_std.where(in_season)

    difference = acc - acc_baseline
    # Both ratios are only meaningful once there is an accumulation worth
    # comparing against, so both take the same floor -- see BASELINE_FLOOR.
    comparable = np.abs(acc_baseline) > baseline_floor
    relative = 100.0 * difference / acc_baseline.where(comparable)
    z_score = difference / baseline_std.where(comparable & (baseline_std > 0))

    days_of_season = (
        (query_date - picked["start_date"]) / np.timedelta64(1, "D") + 1
    ).where(in_season, 0)

    out = xr.Dataset(
        {
            "DOS": days_of_season.fillna(0).clip(0, 65535).astype("uint16"),
            "season": picked["season"],
            "acc": acc.astype("float32"),
            "acc_baseline": acc_baseline.astype("float32"),
            "anomaly_abs": difference.astype("float32"),
            "anomaly_rel": relative.astype("float32"),
            "anomaly_z": z_score.astype("float32"),
        }
    )
    out = out.expand_dims(time=[query_date.astype("datetime64[ns]")])
    out = out.transpose("time", "y", "x")
    _stamp_attrs(out, flux)
    return out


def _accumulate_kernel(
    rate, start_idx, start_date, *, period_index, query_date, query_idx, year_min, resolution
):
    series = FluxSeries.from_rate(
        np.moveaxis(rate, -1, 0), period_index, year_min, resolution
    )
    return accumulate_to_date(series, start_idx, start_date, query_date, query_idx)


def _slot_fraction(
    start_date: xr.DataArray,
    query_date: np.datetime64,
    query_idx: int,
    year_min: int,
    resolution: TemporalResolution,
) -> xr.DataArray:
    """How far the query date sits into its day-of-season slot, in ``[0, 1]``.

    Measured within the **slot**, which is not always the whole period. Slot 1
    begins at the season start, typically part-way through its dekad, so a
    query on the season's first day is one day into a 6-day slot -- not one day
    into a 10-day dekad. Using the period instead would overstate the baseline
    for every pixel in its first slot, by more the later in the dekad its
    season began.

    Later slots do span a whole period, and the expression below reduces to
    that case on its own.
    """
    period_start, period_end = period_bounds(
        np.array([query_idx]), year_min, resolution
    )
    first = np.datetime64(period_start[0], "ns")
    last = np.datetime64(period_end[0], "ns")
    day = np.timedelta64(1, "D")

    slot_start = xr.where(start_date > first, start_date, first)
    elapsed = (np.datetime64(query_date, "ns") - slot_start) / day + 1
    length = (last - slot_start) / day + 1
    return (elapsed / length).clip(0.0, 1.0)


def _read_curve(
    curve: xr.DataArray,
    slot: xr.DataArray,
    label: xr.DataArray,
    fraction: xr.DataArray,
    labels: np.ndarray,
) -> xr.DataArray:
    """Sample a stored ``(season, pos, y, x)`` curve per pixel.

    Picks each pixel's own season label, then reads that season's cumulative
    curve at the pixel's fractional day-of-season position.
    """
    return xr.apply_ufunc(
        _read_curve_kernel,
        curve,
        slot,
        label,
        fraction,
        input_core_dims=[["season", "pos"], [], [], []],
        dask="parallelized",
        output_dtypes=[np.float32],
        kwargs={"labels": labels},
    )


def _read_curve_kernel(curve, slot, label, fraction, *, labels):
    """``curve`` arrives as ``(*spatial, season, pos)``; returns ``(*spatial,)``."""
    curve = np.moveaxis(curve, (-2, -1), (0, 1))  # (season, pos, *spatial)
    # Map the season label (1, 2, ... ; 0 means "no season") onto its position
    # along the season axis. Pixels with no season read season 0's curve and
    # are masked out by the caller, so the value never escapes.
    lookup = {int(v): i for i, v in enumerate(labels)}
    index = np.zeros(label.shape, dtype=np.int64)
    for value, position in lookup.items():
        index[label == value] = position

    picked = np.take_along_axis(curve, index[np.newaxis, np.newaxis], axis=0)[0]
    return interpolate_slot(picked, slot, np.asarray(fraction, dtype=np.float32))


def _stamp_attrs(out: xr.Dataset, flux: xr.DataArray) -> None:
    amount_units = flux.attrs.get("units", "").replace("/day", "")
    out["DOS"].attrs = {
        "long_name": "days since start of season",
        "units": "days",
        "comment": "0 marks a pixel that is not in season; it is the in-season mask.",
    }
    out["season"].attrs = {
        "long_name": "phenology season label in effect",
        "comment": "0 where no season is active.",
    }
    out["acc"].attrs = {
        "long_name": "flux accumulated since start of season",
        "units": amount_units,
    }
    out["acc_baseline"].attrs = {
        "long_name": "baseline accumulation at the same day of season",
        "units": amount_units,
    }
    out["anomaly_abs"].attrs = {
        "long_name": "accumulated flux minus baseline",
        "units": amount_units,
        "comment": "Signed: negative is a deficit. Never store as an unsigned type.",
    }
    out["anomaly_rel"].attrs = {
        "long_name": "accumulated flux relative to baseline",
        "units": "%",
    }
    out["anomaly_z"].attrs = {
        "long_name": "standardised anomaly",
        "units": "1",
        "comment": "Difference divided by the baseline standard deviation across years.",
    }
