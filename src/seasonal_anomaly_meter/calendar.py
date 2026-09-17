"""Absolute period indexing shared by the phenology and the flux series.

Every pixel's season is defined by day-of-year codes, while the flux arrives as
dekads (or months). To line the two up, both are mapped onto a single monotonic
integer axis anchored at ``year_min``::

    abs_index = (year - year_min) * periods_per_year + period_of_year

``year_min`` is therefore load-bearing: the phenology and the flux *must* be
indexed against the same anchor, or every season silently shifts by a multiple
of ``periods_per_year``. It is an explicit argument everywhere in this module
rather than a module-level default, so a mismatch is a visible call-site bug.

A dekad is a 10-day period; each calendar month has exactly three (D1: days
1-10, D2: 11-20, D3: 21 to month end). D3 is therefore 8-11 days long, which is
why accumulation weights every period by its true length instead of assuming 10.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "DEKADAL",
    "MONTHLY",
    "TemporalResolution",
    "dates_to_abs_index",
    "doy_year_to_abs_index",
    "infer_resolution",
    "period_bounds",
    "period_lengths",
]


class TemporalResolution:
    """How a variable's time axis divides the calendar year."""

    def __init__(self, name: str, periods_per_year: int):
        self.name = name
        self.periods_per_year = periods_per_year

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"TemporalResolution({self.name!r}, {self.periods_per_year})"


DEKADAL = TemporalResolution("dekadal", 36)
MONTHLY = TemporalResolution("monthly", 12)

def infer_resolution(time) -> TemporalResolution:
    """The :class:`TemporalResolution` implied by a time coordinate's spacing.

    Reads the data rather than a filename, so it works for any caller-supplied
    series regardless of where it came from. Dekads average 10.15 days and
    months 30.4, which are far enough apart that the median step separates them
    unambiguously; anything else raises rather than guessing, because picking
    the wrong ``periods_per_year`` would misplace every season by a multiple of
    a year without any visible symptom.

    A series with a gap is fine -- the median ignores it -- but an irregular
    series that is neither dekadal nor monthly is rejected.
    """
    days = np.asarray(time, dtype="datetime64[D]")
    if days.size < 2:
        raise ValueError(
            "cannot infer a temporal resolution from fewer than two time steps; "
            "pass resolution=DEKADAL or resolution=MONTHLY explicitly."
        )
    steps = np.diff(days) / np.timedelta64(1, "D")
    if np.any(steps <= 0):
        raise ValueError(
            "the time axis is not strictly increasing; sort it with "
            ".sortby('time') before use."
        )
    step = float(np.median(steps))
    if 8.0 <= step <= 11.5:
        return DEKADAL
    if 27.0 <= step <= 31.5:
        return MONTHLY
    raise ValueError(
        f"the time steps are {step:.1f} days apart on average, which is neither "
        "dekadal (~10) nor monthly (~30). The seasonal anomaly methodology needs "
        "a period short enough to resolve a season but long enough that a season "
        "spans several; pass resolution= explicitly if you are sure."
    )


def doy_year_to_abs_index(
    doy: np.ndarray,
    year: np.ndarray,
    year_min: int,
    resolution: TemporalResolution = DEKADAL,
) -> np.ndarray:
    """Convert ``(day-of-year, year)`` to an absolute period index.

    Shapes are preserved and the result is float so callers can keep NaN for
    "no season here" rather than inventing a sentinel. ``doy`` is 1-based, as
    Copernicus SOSD/EOSD are.

    A day-of-year past the end of its year (Copernicus encodes a season that
    runs into the next calendar year that way) rolls over naturally, because
    the conversion goes through a real date rather than arithmetic on the
    year/month fields.
    """
    dates = (
        year.ravel().astype("int64").astype(str).astype("datetime64[D]")
        + (doy.ravel().astype("int64") - 1).astype("timedelta64[D]")
    )
    y = dates.astype("datetime64[Y]").astype(int) + 1970
    m = dates.astype("datetime64[M]").astype(int) % 12 + 1

    if resolution is MONTHLY:
        period_of_year = m - 1
    else:
        day_of_month = (dates - dates.astype("datetime64[M]")).astype(int) + 1
        period_of_year = (
            (m - 1) * 3 + (day_of_month > 10).astype(int) + (day_of_month > 20).astype(int)
        )

    idx = (y - year_min) * resolution.periods_per_year + period_of_year
    return idx.astype(float).reshape(doy.shape)


def dates_to_abs_index(
    dates,
    year_min: int,
    resolution: TemporalResolution = DEKADAL,
) -> np.ndarray:
    """Absolute period index of each calendar date, as int64.

    Used to place a flux series' ``time`` coordinate -- and a query date -- on
    the same axis as the phenology. Any date inside a period maps to that
    period, so it does not matter whether the source stamps a period by its
    first day or its middle.
    """
    days = np.asarray(dates, dtype="datetime64[D]")
    year = days.astype("datetime64[Y]").astype(int) + 1970
    doy = (days - days.astype("datetime64[Y]")).astype("timedelta64[D]").astype(int) + 1
    return doy_year_to_abs_index(doy, year, year_min, resolution).astype(np.int64)


def period_bounds(
    abs_index: np.ndarray,
    year_min: int,
    resolution: TemporalResolution = DEKADAL,
) -> tuple[np.ndarray, np.ndarray]:
    """First and last calendar day of each absolute period, as ``datetime64[D]``.

    Both bounds are inclusive, so a period's length in days is
    ``(end - start) + 1``. Vectorised over any shape.
    """
    abs_index = np.asarray(abs_index)
    ppy = resolution.periods_per_year
    year = year_min + abs_index // ppy
    period_of_year = abs_index % ppy

    if resolution is MONTHLY:
        month = period_of_year
        start = _ym_to_date(year, month)
        end = _ym_to_date(year, month + 1) - np.timedelta64(1, "D")
        return start, end

    month, dekad_in_month = np.divmod(period_of_year, 3)
    month_start = _ym_to_date(year, month)
    start = month_start + (dekad_in_month * 10).astype("timedelta64[D]")
    # D3 runs to the end of the month, so its end is the next month's start - 1.
    month_end = _ym_to_date(year, month + 1) - np.timedelta64(1, "D")
    end = np.where(
        dekad_in_month == 2,
        month_end,
        month_start + ((dekad_in_month + 1) * 10 - 1).astype("timedelta64[D]"),
    )
    return start, end.astype("datetime64[D]")


def period_lengths(
    abs_index: np.ndarray,
    year_min: int,
    resolution: TemporalResolution = DEKADAL,
) -> np.ndarray:
    """Length in days of each absolute period, as float32.

    This is the weight that turns a per-day rate (mm/day, gC/m2/day) into an
    amount over the period. Never assume 10 for a dekad -- D3 is 8-11 days.
    """
    start, end = period_bounds(abs_index, year_min, resolution)
    days = ((end - start) / np.timedelta64(1, "D")).astype(np.int64) + 1
    return days.astype(np.float32)


def _ym_to_date(year: np.ndarray, month: np.ndarray) -> np.ndarray:
    """``(year, month)`` -> first day of that month, with month overflow.

    ``month`` is 0-based and may equal ``12`` (or more), which rolls into the
    next year -- that is what makes the "next month's start minus one day"
    trick work for December.
    """
    months_since_epoch = (np.asarray(year) - 1970) * 12 + np.asarray(month)
    return months_since_epoch.astype("datetime64[M]").astype("datetime64[D]")
