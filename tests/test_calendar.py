"""Period arithmetic: the axis both the phenology and the flux are mapped onto."""

import numpy as np
import pytest

from seasonal_anomaly_meter.calendar import (
    DEKADAL,
    MONTHLY,
    doy_year_to_abs_index,
    period_bounds,
    period_lengths,
)

YEAR_MIN = 2018


def _idx(doy, year, resolution=DEKADAL):
    return doy_year_to_abs_index(
        np.array([doy]), np.array([year]), YEAR_MIN, resolution
    )[0]


@pytest.mark.parametrize(
    "doy, year, expected",
    [
        (1, 2018, 0),  # 2018-01-D1, the anchor
        (11, 2018, 1),
        (21, 2018, 2),
        (32, 2018, 3),  # 2018-02-D1
        (355, 2018, 35),  # 2018-12-D3
        (1, 2019, 36),  # one whole year on
    ],
)
def test_dekad_index(doy, year, expected):
    assert _idx(doy, year) == expected


def test_monthly_index():
    assert _idx(1, 2018, MONTHLY) == 0
    assert _idx(335, 2018, MONTHLY) == 11
    assert _idx(1, 2019, MONTHLY) == 12


def test_dekads_tile_the_year_exactly():
    """Every day of the year belongs to exactly one dekad, leap years included."""
    for year, expected in ((2018, 365), (2020, 366)):
        offset = (year - YEAR_MIN) * 36
        assert period_lengths(np.arange(offset, offset + 36), YEAR_MIN).sum() == expected


def test_third_dekad_length_varies():
    """D3 is 8-11 days -- assuming 10 would bias every accumulation."""
    lengths = period_lengths(np.arange(36), YEAR_MIN)
    assert lengths[2] == 11  # January
    assert lengths[5] == 8  # February, non-leap
    assert period_lengths(np.array([2 + 36 * 2]), YEAR_MIN)[0] == 11  # Jan 2020
    assert period_lengths(np.array([5 + 36 * 2]), YEAR_MIN)[0] == 9  # Feb 2020, leap


def test_period_bounds_are_inclusive_and_contiguous():
    start, end = period_bounds(np.arange(36), YEAR_MIN)
    assert start[0] == np.datetime64("2018-01-01")
    assert end[35] == np.datetime64("2018-12-31")
    # No gaps and no overlaps between consecutive periods.
    assert np.all(start[1:] - end[:-1] == np.timedelta64(1, "D"))


def test_month_bounds_handle_december_rollover():
    start, end = period_bounds(np.array([11]), YEAR_MIN, MONTHLY)
    assert start[0] == np.datetime64("2018-12-01")
    assert end[0] == np.datetime64("2018-12-31")


@pytest.mark.parametrize(
    "doy, year, expected_period",
    [
        (-61, 2020, ("2019-10-21", "2019-10-31")),  # season began the previous year
        (416, 2020, ("2021-02-11", "2021-02-20")),  # ... and ends in the next one
    ],
)
def test_out_of_year_day_of_year_rolls_over(doy, year, expected_period):
    """Copernicus really does publish SOSD < 1 and EOSD > 365 for cross-year seasons."""
    idx = _idx(doy, year)
    start, end = period_bounds(np.array([int(idx)]), YEAR_MIN)
    assert (str(start[0]), str(end[0])) == expected_period
