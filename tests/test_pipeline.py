"""Baseline and anomaly together, on synthetic data with a known answer.

Nothing here touches the network. The flux is a constant 1 unit/day, so every
accumulation is an exact day count and the anomalies have arithmetic that can
be checked by hand.
"""

import numpy as np
import pytest
import rioxarray  # noqa: F401  -- registers .rio, used for the CRS round-trip
import xarray as xr

from seasonal_anomaly_meter.anomaly import compute_anomaly
from seasonal_anomaly_meter.baseline import build_baseline
from seasonal_anomaly_meter.calendar import period_bounds
from seasonal_anomaly_meter.io import (
    anomaly_encoding,
    baseline_encoding,
    check_packing_range,
    open_zarr,
    write_zarr,
)
from seasonal_anomaly_meter.season import season_indices

YEAR_MIN = 2018
YEARS = [2018, 2019, 2020, 2021]
SHAPE = (4, 4)
#: Day-of-year 74 is 15 March in a common year and 14 March in a leap one, so a
#: season pinned to it is one day longer in 2020 -- which is why the baseline
#: below lands on 93.25 rather than a round 93.
SOSD, EOSD = 74, 232
QUERY = "2021-06-15"
QUERY_DAYS = 93


@pytest.fixture
def flux():
    n_periods = len(YEARS) * 36
    starts, _ = period_bounds(np.arange(n_periods), YEAR_MIN)
    data = np.ones((n_periods, *SHAPE), dtype="float32")
    return xr.DataArray(
        data,
        dims=("time", "y", "x"),
        coords={
            "time": starts.astype("datetime64[ns]"),
            "y": np.arange(SHAPE[0], dtype=float) * 300 + 100,
            "x": np.arange(SHAPE[1], dtype=float) * 300 + 100,
        },
        attrs={"units": "gC/m2/day"},
    ).rio.write_crs("EPSG:32636")


@pytest.fixture
def seasons(flux):
    dims = ("season", "year", "y", "x")
    full = (1, len(YEARS), *SHAPE)
    phenology = xr.Dataset(
        {
            "SOSD": (dims, np.full(full, SOSD, dtype="float32")),
            "EOSD": (dims, np.full(full, EOSD, dtype="float32")),
            "QA": (dims, np.zeros(full, dtype="uint8")),
        },
        coords={"season": [1], "year": YEARS, "y": flux["y"], "x": flux["x"]},
    )
    return season_indices(phenology, YEAR_MIN)


@pytest.fixture
def baseline(flux, seasons):
    return build_baseline(flux, seasons, YEAR_MIN, min_years=3).compute()


def test_baseline_curve_counts_days(baseline):
    """At 1 unit/day each slot holds the days elapsed by the end of that slot."""
    curve = baseline["acc_mean"].isel(season=0, y=0, x=0).values
    # Slot 1 ends 20 March; 15->20 March is 6 days, and 7 in the leap year.
    assert curve[0] == pytest.approx(6.25)
    assert curve[1] == pytest.approx(17.25)
    assert np.isfinite(curve).sum() == 16  # a 16-dekad season


def test_baseline_spread_reflects_the_leap_year(baseline):
    """One of four years runs a day longer, so the spread is a real 0.5."""
    std = baseline["acc_std"].isel(season=0, y=0, x=0).values
    assert std[0] == pytest.approx(0.5)


def test_baseline_counts_contributing_years(baseline):
    counts = baseline["acc_count"].isel(season=0, y=0, x=0).values
    assert counts[0] == len(YEARS)
    assert counts[-1] == 0  # past the end of the season


def test_thin_slots_are_masked(flux):
    """A slot reached in too few years must not look as solid as a full one."""
    dims = ("season", "year", "y", "x")
    full = (1, len(YEARS), *SHAPE)
    sosd = np.full(full, SOSD, dtype="float32")
    eosd = np.full(full, EOSD, dtype="float32")
    # Only the last year has a long season; its extra slots are single-sample.
    eosd[0, :-1] = 120
    phenology = xr.Dataset(
        {
            "SOSD": (dims, sosd),
            "EOSD": (dims, eosd),
            "QA": (dims, np.zeros(full, dtype="uint8")),
        },
        coords={"season": [1], "year": YEARS, "y": flux["y"], "x": flux["x"]},
    )
    seasons = season_indices(phenology, YEAR_MIN)

    lenient = build_baseline(flux, seasons, YEAR_MIN, min_years=1).compute()
    strict = build_baseline(flux, seasons, YEAR_MIN, min_years=3).compute()

    counts = strict["acc_count"].isel(season=0, y=0, x=0).values
    thin = counts == 1
    assert thin.any(), "fixture should produce single-year slots"
    assert np.isfinite(lenient["acc_mean"].isel(season=0, y=0, x=0).values[thin]).all()
    assert np.isnan(strict["acc_mean"].isel(season=0, y=0, x=0).values[thin]).all()


def test_anomaly_against_its_own_baseline_is_near_zero(flux, seasons, baseline):
    result = compute_anomaly(flux, seasons, baseline, QUERY, YEAR_MIN).compute()
    point = result.isel(time=0, y=0, x=0)

    assert int(point["DOS"]) == QUERY_DAYS
    assert int(point["season"]) == 1
    assert float(point["acc"]) == pytest.approx(QUERY_DAYS)
    # The baseline averages in the leap year, so it sits a quarter-day above.
    assert float(point["acc_baseline"]) == pytest.approx(93.25)
    assert float(point["anomaly_abs"]) == pytest.approx(-0.25)
    assert float(point["anomaly_z"]) == pytest.approx(-0.5)


def test_doubling_the_flux_doubles_the_accumulation(flux, seasons, baseline):
    result = compute_anomaly(flux * 2, seasons, baseline, QUERY, YEAR_MIN).compute()
    point = result.isel(time=0, y=0, x=0)

    assert float(point["acc"]) == pytest.approx(2 * QUERY_DAYS)
    assert float(point["anomaly_rel"]) == pytest.approx(99.46, abs=0.1)
    assert float(point["anomaly_abs"]) > 0


def test_out_of_season_pixels_are_masked(flux, seasons, baseline):
    result = compute_anomaly(flux, seasons, baseline, "2021-01-15", YEAR_MIN).compute()
    point = result.isel(time=0, y=0, x=0)

    assert int(point["DOS"]) == 0
    assert int(point["season"]) == 0
    assert np.isnan(float(point["acc"]))
    assert np.isnan(float(point["anomaly_abs"]))


def test_dos_is_the_in_season_mask(flux, seasons, baseline):
    """The dashboard relies on DOS == 0 meaning "this band did not happen"."""
    result = compute_anomaly(flux, seasons, baseline, QUERY, YEAR_MIN).compute()
    np.testing.assert_array_equal(
        result["DOS"].values == 0, result["season"].values == 0
    )


def test_anomaly_is_self_consistent(flux, seasons, baseline):
    result = compute_anomaly(flux * 3, seasons, baseline, QUERY, YEAR_MIN).compute()
    difference = result["acc"] - result["acc_baseline"]
    np.testing.assert_allclose(
        difference.values, result["anomaly_abs"].values, rtol=1e-5
    )


def test_lazy_and_eager_paths_agree(flux, seasons):
    """The pipeline runs chunked; the tests mostly do not."""
    chunked_flux = flux.chunk({"time": -1, "y": 2, "x": 2})
    chunked_seasons = seasons.chunk({"y": 2, "x": 2})

    eager = build_baseline(flux, seasons, YEAR_MIN, min_years=3).compute()
    lazy = build_baseline(chunked_flux, chunked_seasons, YEAR_MIN, min_years=3).compute()
    np.testing.assert_allclose(
        eager["acc_mean"].values, lazy["acc_mean"].values, equal_nan=True
    )

    eager_anomaly = compute_anomaly(flux, seasons, eager, QUERY, YEAR_MIN).compute()
    lazy_anomaly = compute_anomaly(
        chunked_flux, chunked_seasons, lazy, QUERY, YEAR_MIN
    ).compute()
    np.testing.assert_allclose(
        eager_anomaly["anomaly_abs"].values,
        lazy_anomaly["anomaly_abs"].values,
        equal_nan=True,
    )


def test_mismatched_grids_are_rejected(flux, seasons):
    """A phenology on a different grid must fail loudly, not be coerced."""
    shifted = seasons.assign_coords(x=seasons["x"] + 1000.0)
    with pytest.raises(ValueError, match="disagree on the x axis"):
        build_baseline(flux, shifted, YEAR_MIN)


def test_stores_round_trip_with_their_crs(tmp_path, flux, seasons, baseline):
    result = compute_anomaly(flux, seasons, baseline, QUERY, YEAR_MIN).compute()

    check_packing_range(baseline, 0.1)
    write_zarr(baseline, tmp_path / "baseline.zarr", baseline_encoding(baseline, scale_factor=0.1))
    write_zarr(result, tmp_path / "anomaly.zarr", anomaly_encoding(result))

    reopened_baseline = open_zarr(tmp_path / "baseline.zarr")
    reopened_anomaly = open_zarr(tmp_path / "anomaly.zarr")

    assert reopened_baseline.rio.crs.to_epsg() == 32636
    assert reopened_anomaly.rio.crs.to_epsg() == 32636
    np.testing.assert_allclose(
        reopened_baseline["acc_mean"].values,
        baseline["acc_mean"].values,
        atol=0.05,  # int16 at scale_factor 0.1
        equal_nan=True,
    )


def test_packing_refuses_to_saturate(baseline):
    """Silent int16 saturation is the failure that produced wrapped values before."""
    too_big = baseline.copy()
    too_big["acc_mean"] = too_big["acc_mean"] * 1000
    with pytest.raises(ValueError, match="int16"):
        check_packing_range(too_big, 0.1)


def test_negative_anomalies_survive_the_store(tmp_path, flux, seasons, baseline):
    """The reference stored these unsigned; deficits wrapped to ~65529."""
    result = compute_anomaly(flux * 0.5, seasons, baseline, QUERY, YEAR_MIN).compute()
    assert float(result["anomaly_abs"].isel(time=0, y=0, x=0)) < 0

    write_zarr(result, tmp_path / "anomaly.zarr", anomaly_encoding(result))
    reopened = open_zarr(tmp_path / "anomaly.zarr")
    assert float(reopened["anomaly_abs"].isel(time=0, y=0, x=0)) < 0


def test_dos_and_season_survive_the_store_as_integers(tmp_path, flux, seasons, baseline):
    """Zero means "not in season" here -- a real value, not missing data.

    Declaring it as ``_FillValue`` would make xarray decode it back to NaN and
    promote both masks to float, which silently destroys the in-season mask the
    rest of the product (and the dashboard) keys off.
    """
    result = compute_anomaly(flux, seasons, baseline, "2021-01-15", YEAR_MIN).compute()
    assert int(result["DOS"].max()) == 0  # nothing is in season on this date

    write_zarr(result, tmp_path / "anomaly.zarr", anomaly_encoding(result))
    reopened = open_zarr(tmp_path / "anomaly.zarr")

    assert reopened["DOS"].dtype == np.uint16
    assert reopened["season"].dtype == np.uint8
    assert not np.isnan(reopened["DOS"].values).any()
    np.testing.assert_array_equal(
        reopened["DOS"].values == 0, reopened["season"].values == 0
    )


def test_ratios_are_masked_below_the_baseline_floor(flux, seasons, baseline):
    """Dividing by a negligible baseline is what produced |z| up to 126 on real
    Gezira NPP: those pixels had a median baseline of 0.19 gC/m2 against 17.6
    for everything else. The absolute value of the accumulation is still
    reported -- only the two ratios are withheld.
    """
    point = dict(time=0, y=0, x=0)
    guarded = compute_anomaly(
        flux, seasons, baseline, "2021-03-17", YEAR_MIN, baseline_floor=10.0
    ).compute().isel(**point)
    allowed = compute_anomaly(
        flux, seasons, baseline, "2021-03-17", YEAR_MIN, baseline_floor=0.0
    ).compute().isel(**point)

    assert float(allowed["acc_baseline"]) < 10.0  # below the floor we imposed
    assert np.isnan(float(guarded["anomaly_rel"]))
    assert np.isnan(float(guarded["anomaly_z"]))
    assert np.isfinite(float(allowed["anomaly_rel"]))
    assert np.isfinite(float(allowed["anomaly_z"]))

    # The accumulation itself is never withheld -- only the ratios.
    assert float(guarded["acc"]) == pytest.approx(float(allowed["acc"]))
    assert np.isfinite(float(guarded["anomaly_abs"]))


def test_first_slot_fraction_is_measured_from_the_season_start(flux, seasons, baseline):
    """Slot 1 begins mid-dekad, at SOSD, so the fraction must not use the dekad.

    Measuring within the period instead would put the query 5/10 of the way
    through slot 1 rather than 1/6, overstating the baseline for every pixel in
    its first slot -- by more the later in its dekad the season began.
    """
    result = compute_anomaly(
        flux, seasons, baseline, "2021-03-15", YEAR_MIN, baseline_floor=0.0
    ).compute()
    first_day = float(result["acc_baseline"].isel(time=0, y=0, x=0))

    slot_one_total = float(baseline["acc_mean"].isel(season=0, pos=0, y=0, x=0))
    assert first_day == pytest.approx(slot_one_total / 6, abs=0.01)
    # And the observed accumulation on day one is exactly one day's flux.
    assert float(result["acc"].isel(time=0, y=0, x=0)) == pytest.approx(1.0)


def test_baseline_tracks_the_observation_through_the_first_slot(flux, seasons, baseline):
    """At a constant rate the two curves should stay within a fraction of a day."""
    for date, expected_days in [
        ("2021-03-15", 1),
        ("2021-03-17", 3),
        ("2021-03-20", 6),
        ("2021-03-25", 11),
    ]:
        result = compute_anomaly(
            flux, seasons, baseline, date, YEAR_MIN, baseline_floor=0.0
        ).compute()
        point = result.isel(time=0, y=0, x=0)
        assert float(point["acc"]) == pytest.approx(expected_days)
        assert float(point["acc_baseline"]) == pytest.approx(expected_days, abs=0.3)


def test_season_starting_later_in_the_query_dekad_is_not_yet_active(flux):
    """A season beginning 18 March shares its dekad with a 15 March query.

    Testing activity on period indices would call it active three days early,
    which produced a zero day-of-season next to a non-zero season label -- and
    an accumulation that trimmed more off both ends of the dekad than the dekad
    contains, coming out negative.
    """
    dims = ("season", "year", "y", "x")
    full = (1, len(YEARS), *SHAPE)
    phenology = xr.Dataset(
        {
            "SOSD": (dims, np.full(full, 77, dtype="float32")),  # 18 March
            "EOSD": (dims, np.full(full, EOSD, dtype="float32")),
            "QA": (dims, np.zeros(full, dtype="uint8")),
        },
        coords={"season": [1], "year": YEARS, "y": flux["y"], "x": flux["x"]},
    )
    seasons = season_indices(phenology, YEAR_MIN)
    baseline = build_baseline(flux, seasons, YEAR_MIN, min_years=3).compute()

    # 15 March: same dekad as the season start, but three days before it.
    early = compute_anomaly(flux, seasons, baseline, "2021-03-15", YEAR_MIN).compute()
    point = early.isel(time=0, y=0, x=0)
    assert int(point["season"]) == 0
    assert int(point["DOS"]) == 0
    assert np.isnan(float(point["acc"]))

    # 18 March: the season has begun, and one day has accumulated.
    started = compute_anomaly(flux, seasons, baseline, "2021-03-18", YEAR_MIN).compute()
    point = started.isel(time=0, y=0, x=0)
    assert int(point["season"]) == 1
    assert int(point["DOS"]) == 1
    assert float(point["acc"]) == pytest.approx(1.0)


def test_accumulation_is_never_negative_for_a_non_negative_flux(flux, seasons, baseline):
    """A guard on the trimming arithmetic, which subtracts from both ends."""
    for date in ("2021-03-15", "2021-03-16", "2021-03-20", "2021-06-15", "2021-08-20"):
        result = compute_anomaly(flux, seasons, baseline, date, YEAR_MIN).compute()
        values = result["acc"].values
        finite = values[np.isfinite(values)]
        assert (finite >= 0).all(), f"negative accumulation on {date}"
