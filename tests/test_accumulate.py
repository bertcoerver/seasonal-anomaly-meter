"""The accumulation kernel.

A constant rate of 1 unit/day is the workhorse fixture here: with it, every
accumulation must come out as an exact whole number of days, so the partial
first/last period corrections have nowhere to hide.
"""

import numpy as np
import pytest

from seasonal_anomaly_meter.accumulate import (
    FluxSeries,
    accumulate_by_slot,
    accumulate_to_date,
    interpolate_slot,
)
from seasonal_anomaly_meter.calendar import doy_year_to_abs_index, period_bounds

YEAR_MIN = 2018
PERIODS = np.arange(72)  # 2018-2019, dekadal


@pytest.fixture
def unit_rate():
    """1 unit/day over two years, on a 1x1 raster."""
    return np.ones((72, 1, 1), dtype=np.float32)


@pytest.fixture
def season():
    """2018-03-15 .. 2018-08-20, as (start_idx, start_date, end_idx)."""
    start_idx = doy_year_to_abs_index(np.array([74]), np.array([2018]), YEAR_MIN)
    end_idx = doy_year_to_abs_index(np.array([232]), np.array([2018]), YEAR_MIN)
    start_date = np.array(["2018-03-15"], dtype="datetime64[D]")
    return start_idx.reshape(1, 1), start_date.reshape(1, 1), end_idx.reshape(1, 1)


def _query(date: str):
    """(np.datetime64 date, absolute period index) for a calendar date."""
    date = np.datetime64(date, "D")
    year = date.astype("datetime64[Y]").astype(int) + 1970
    doy = (date - date.astype("datetime64[Y]")).astype(int) + 1
    idx = int(doy_year_to_abs_index(np.array([doy]), np.array([year]), YEAR_MIN)[0])
    return date, idx


def test_prefix_sum_of_unit_rate_counts_days(unit_rate):
    series = FluxSeries.from_rate(unit_rate, PERIODS, YEAR_MIN)
    assert series.cumulative[0, 0, 0] == 0.0
    assert series.cumulative[-1, 0, 0] == 730.0  # 365 + 365


@pytest.mark.parametrize(
    "query_date, expected_days",
    [
        ("2018-03-15", 1),  # the season's first day
        ("2018-03-20", 6),  # end of the season's first dekad
        ("2018-06-15", 93),  # mid-dekad, the interesting case
        ("2018-08-20", 159),  # the season's last day
    ],
)
def test_accumulate_to_date_is_exact(unit_rate, season, query_date, expected_days):
    """At 1 unit/day the accumulation *is* the inclusive day count."""
    start_idx, start_date, _ = season
    series = FluxSeries.from_rate(unit_rate, PERIODS, YEAR_MIN)
    date, idx = _query(query_date)

    acc = accumulate_to_date(series, start_idx, start_date, date, idx)
    assert acc[0, 0] == pytest.approx(expected_days)


def test_accumulate_by_slot_follows_the_season_curve(unit_rate, season):
    start_idx, start_date, end_idx = season
    series = FluxSeries.from_rate(unit_rate, PERIODS, YEAR_MIN)

    by_slot = accumulate_by_slot(series, start_idx, start_date, end_idx, 49)

    # Slot k ends with the k-th period of the season.
    first = int(start_idx[0, 0])
    _, ends = period_bounds(np.arange(first, first + 5), YEAR_MIN)
    expected = [(ends[k] - start_date[0, 0]).astype(int) + 1 for k in range(5)]
    assert list(by_slot[:5, 0, 0]) == pytest.approx(expected)

    # The curve stops at the end of the season and never wraps into the next.
    n_finite = int(np.isfinite(by_slot[:, 0, 0]).sum())
    assert n_finite == int(end_idx[0, 0] - start_idx[0, 0]) + 1


def test_interpolation_replaces_a_stored_mean_rate(unit_rate, season):
    """The plan's load-bearing claim: interpolating between two cumulative
    slots equals adding mean_rate * days_into_slot, so no rate array is stored.
    """
    start_idx, start_date, end_idx = season
    series = FluxSeries.from_rate(unit_rate, PERIODS, YEAR_MIN)
    by_slot = accumulate_by_slot(series, start_idx, start_date, end_idx, 49)
    date, idx = _query("2018-06-15")

    slot = np.array([[idx - int(start_idx[0, 0]) + 1]])
    p_start, p_end = period_bounds(np.array([idx]), YEAR_MIN)
    fraction = np.array(
        [[((date - p_start[0]).astype(int) + 1) / ((p_end[0] - p_start[0]).astype(int) + 1)]]
    )

    interpolated = interpolate_slot(by_slot, slot, fraction)
    direct = accumulate_to_date(series, start_idx, start_date, date, idx)
    assert interpolated[0, 0] == pytest.approx(direct[0, 0])
    assert interpolated[0, 0] == pytest.approx(93.0)


def test_pixels_without_a_season_are_nan(unit_rate):
    series = FluxSeries.from_rate(unit_rate, PERIODS, YEAR_MIN)
    start_idx = np.array([[np.nan]])
    start_date = np.array([["NaT"]], dtype="datetime64[D]")
    date, idx = _query("2018-06-15")

    acc = accumulate_to_date(series, start_idx, start_date, date, idx)
    assert np.isnan(acc[0, 0])


def test_query_before_season_start_is_nan(unit_rate, season):
    """A season that has not begun yet accumulates nothing, not a negative."""
    start_idx, start_date, _ = season
    series = FluxSeries.from_rate(unit_rate, PERIODS, YEAR_MIN)
    date, idx = _query("2018-01-15")

    acc = accumulate_to_date(series, start_idx, start_date, date, idx)
    assert np.isnan(acc[0, 0])


def test_nan_rates_do_not_poison_the_sum(season):
    """A gap in the flux contributes zero rather than NaN-ing the whole season."""
    start_idx, start_date, _ = season
    rate = np.ones((72, 1, 1), dtype=np.float32)
    rate[10] = np.nan
    series = FluxSeries.from_rate(rate, PERIODS, YEAR_MIN)
    date, idx = _query("2018-06-15")

    acc = accumulate_to_date(series, start_idx, start_date, date, idx)
    # Period 10 is 2018-04-11..20, ten days, all inside the window.
    assert acc[0, 0] == pytest.approx(93.0 - 10.0)
