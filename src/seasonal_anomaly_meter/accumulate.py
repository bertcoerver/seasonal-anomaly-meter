"""The one accumulation kernel, used by both the baseline and the live stage.

The flux is a **rate** (mm/day, gC/m2/day) that is constant across each
dekad. Accumulating it between two dates is therefore a weighted sum of whole
periods with two partial ones at the ends::

    amount = sum over periods p of  rate[p] * (days of p inside [start, end])

Computed naively that is a loop over periods per pixel. Instead the whole
series is turned into a prefix sum once, so any interval becomes two lookups
plus two boundary corrections:

    amount = C[end_period + 1] - C[start_period]          whole periods
             - rate[start_period] * days_before_start     trim the first
             - rate[end_period]   * days_after_end        trim the last

Both stages call the same functions, so a historical baseline and a live value
are produced by identical arithmetic. That is deliberate: the original code
accumulated the *mean rate* for the baseline but the *observed rate* for the
current season, which are only equal when every period has the same length --
and dekad D3 is 8 to 11 days.

The rate, the per-period amount and the prefix sum are carried together in a
:class:`FluxSeries` rather than as loose arrays: they have identical shapes but
different units, so passing one where another belongs is an easy mistake that
no amount of care at the call site would catch.

Every function here is pure NumPy so it can be unit tested without dask or
network, and wrapped for chunked execution by the callers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from seasonal_anomaly_meter.calendar import (
    DEKADAL,
    TemporalResolution,
    period_bounds,
    period_lengths,
)

__all__ = [
    "FluxSeries",
    "accumulate_to_date",
    "accumulate_by_slot",
    "interpolate_slot",
]


@dataclass(frozen=True)
class FluxSeries:
    """A flux time series prepared for interval accumulation.

    ``rate`` is the flux as published (units per day, ``(period, ...)``);
    ``amount`` is that rate times each period's true length; ``cumulative`` is
    the exclusive prefix sum of ``amount`` with a leading zero, so
    ``cumulative[b] - cumulative[a]`` sums periods ``a .. b-1``.
    """

    rate: np.ndarray
    amount: np.ndarray
    cumulative: np.ndarray
    period_index: np.ndarray
    year_min: int
    resolution: TemporalResolution = DEKADAL
    _positions: dict[int, int] = field(repr=False, default_factory=dict)

    @classmethod
    def from_rate(
        cls,
        rate: np.ndarray,
        period_index: np.ndarray,
        year_min: int,
        resolution: TemporalResolution = DEKADAL,
    ) -> "FluxSeries":
        """Weight each period by its length and build the prefix sum.

        NaN rates contribute zero rather than poisoning every later cumulative
        value; callers mask on season validity instead, so a single missing
        dekad degrades one season rather than the whole series.
        """
        lengths = period_lengths(period_index, year_min, resolution)
        shape = (-1,) + (1,) * (rate.ndim - 1)
        amount = (np.nan_to_num(rate, nan=0.0) * lengths.reshape(shape)).astype(np.float32)
        cumulative = np.zeros((amount.shape[0] + 1, *amount.shape[1:]), dtype=np.float32)
        np.cumsum(amount, axis=0, dtype=np.float32, out=cumulative[1:])
        return cls(
            rate=np.asarray(rate, dtype=np.float32),
            amount=amount,
            cumulative=cumulative,
            period_index=np.asarray(period_index),
            year_min=year_min,
            resolution=resolution,
            _positions={int(p): i for i, p in enumerate(np.asarray(period_index))},
        )

    @property
    def n_periods(self) -> int:
        return int(self.rate.shape[0])

    def position_of(self, abs_index: int) -> int | None:
        """Where an absolute period index sits on this series' leading axis."""
        return self._positions.get(int(abs_index))

    def _positions_for(self, start_idx: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """Gather index of each pixel's start period, ``-1`` when not covered."""
        flat = np.full(start_idx.shape, -1, dtype=np.int64)
        if valid.any():
            flat[valid] = [
                self._positions.get(int(w), -1) for w in start_idx[valid].astype(np.int64)
            ]
        return flat[np.newaxis]

    def _head_trim(
        self, start_idx: np.ndarray, start_date: np.ndarray, positions: np.ndarray
    ) -> np.ndarray:
        """Amount to subtract for the part of the first period before the season."""
        valid = np.isfinite(start_idx)
        first_start, _ = period_bounds(
            np.where(valid, start_idx, 0).astype(np.int64), self.year_min, self.resolution
        )
        days_before = np.where(
            valid, (start_date - first_start) / np.timedelta64(1, "D"), 0
        )
        days_before = np.clip(np.nan_to_num(days_before), 0, None).astype(np.float32)
        rate_at_start = np.take_along_axis(self.rate, np.maximum(positions, 0), axis=0)[0]
        return rate_at_start * days_before


def accumulate_to_date(
    series: FluxSeries,
    start_idx: np.ndarray,
    start_date: np.ndarray,
    query_date: np.datetime64,
    query_idx: int,
) -> np.ndarray:
    """Accumulate from each pixel's ``start_date`` up to ``query_date``.

    ``start_idx``/``start_date`` are per-pixel (any trailing shape); NaN/NaT
    marks a pixel with no season, which yields NaN, as does a season that has
    not started by ``query_date``.

    The two corrections are what make this exact for a query landing mid-period:
    the first period is trimmed by the days before ``start_date``, and the last
    by the days after ``query_date``.
    """
    positions = series._positions_for(start_idx, np.isfinite(start_idx))
    valid = np.isfinite(start_idx) & (start_idx <= query_idx) & (positions[0] >= 0)
    if not valid.any():
        return np.full(start_idx.shape, np.nan, dtype=np.float32)

    query_pos = series.position_of(query_idx)
    if query_pos is None:
        raise ValueError(
            f"period {query_idx} is not in the supplied flux series (covers "
            f"{series.period_index.min()}..{series.period_index.max()})."
        )

    total = series.cumulative[query_pos + 1]
    before = np.take_along_axis(series.cumulative, np.maximum(positions, 0), axis=0)[0]
    acc = total - before - series._head_trim(start_idx, start_date, positions)

    _, query_end = period_bounds(
        np.array([query_idx]), series.year_min, series.resolution
    )
    days_after = int((query_end[0] - query_date) / np.timedelta64(1, "D"))
    if days_after > 0:
        acc = acc - series.rate[query_pos] * np.float32(days_after)

    return np.where(valid, acc, np.nan).astype(np.float32)


def accumulate_by_slot(
    series: FluxSeries,
    start_idx: np.ndarray,
    start_date: np.ndarray,
    end_idx: np.ndarray,
    max_pos: int,
) -> np.ndarray:
    """Accumulate to the end of every day-of-season slot, for one season/year.

    Slot ``k`` (1-based) ends with the ``k``-th period of the season, so the
    result is the season's cumulative curve sampled once per period. Returns
    ``(max_pos, *start_idx.shape)``; slots past the season's end, or on pixels
    without a season, are NaN.

    This is what the baseline is built from: run it per historical year, then
    take mean/std/count across years.
    """
    out = np.full((max_pos, *start_idx.shape), np.nan, dtype=np.float32)
    valid = np.isfinite(start_idx)
    if not valid.any():
        return out

    positions = series._positions_for(start_idx, valid)
    covered = valid & (positions[0] >= 0)
    if not covered.any():
        return out

    before = np.take_along_axis(series.cumulative, np.maximum(positions, 0), axis=0)[0]
    head_trim = series._head_trim(start_idx, start_date, positions)
    season_length = end_idx - start_idx  # slot k covers offset k-1

    for k in range(1, max_pos + 1):
        end_pos = positions[0] + k  # exclusive bound into `cumulative`
        in_range = covered & (end_pos <= series.n_periods) & (k - 1 <= season_length)
        if not in_range.any():
            continue
        gather = np.clip(end_pos, 0, series.n_periods)[np.newaxis]
        total = np.take_along_axis(series.cumulative, gather, axis=0)[0]
        out[k - 1] = np.where(in_range, total - before - head_trim, np.nan)
    return out


def interpolate_slot(
    acc_by_slot: np.ndarray,
    slot: np.ndarray,
    fraction: np.ndarray,
) -> np.ndarray:
    """Sample a per-slot cumulative curve at a fractional slot position.

    ``acc_by_slot`` is ``(max_pos, ...)`` from :func:`accumulate_by_slot` (or
    the baseline's stored mean); ``slot`` is the 1-based slot the query falls
    in and ``fraction`` how far into it the query date sits, in ``[0, 1]``.

    Interpolating between slot ``k-1`` and slot ``k`` is exactly equivalent to
    adding ``mean_rate_in_slot_k * days_into_slot_k``, because the difference
    between consecutive cumulative slots *is* that slot's amount. That is why
    the baseline stores no separate mean-rate array: it would be redundant.
    """
    max_pos = acc_by_slot.shape[0]
    k = np.clip(np.nan_to_num(slot, nan=1).astype(np.int64), 1, max_pos)

    upper = np.take_along_axis(acc_by_slot, (k - 1)[np.newaxis], axis=0)[0]
    lower = np.where(
        k > 1,
        np.take_along_axis(acc_by_slot, np.maximum(k - 2, 0)[np.newaxis], axis=0)[0],
        0.0,
    )
    frac = np.clip(np.nan_to_num(fraction), 0.0, 1.0).astype(np.float32)
    result = lower + (upper - lower) * frac
    return np.where(np.isfinite(slot), result, np.nan).astype(np.float32)
