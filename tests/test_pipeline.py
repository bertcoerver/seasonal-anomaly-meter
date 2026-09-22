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
from seasonal_anomaly_meter.calendar import dates_to_abs_index, period_bounds
from xr_utils import append_geozarr, open_geozarr, write_geozarr

from seasonal_anomaly_meter.io import (
    anomaly_encoding,
    baseline_encoding,
    check_packing_range,
)
from seasonal_anomaly_meter.pipeline import required_flux_start, seasonal_anomalies
from seasonal_anomaly_meter.season import MAX_POS, season_indices

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


def _phenology(flux, sosd=SOSD, eosd=EOSD):
    dims = ("season", "year", "y", "x")
    full = (1, len(YEARS), *SHAPE)
    return xr.Dataset(
        {
            "SOSD": (dims, np.full(full, sosd, dtype="float32")),
            "EOSD": (dims, np.full(full, eosd, dtype="float32")),
            "QA": (dims, np.zeros(full, dtype="uint8")),
        },
        coords={"season": [1], "year": YEARS, "y": flux["y"], "x": flux["x"]},
    )


@pytest.fixture
def phenology(flux):
    return _phenology(flux)


@pytest.fixture
def seasons(phenology):
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
    write_geozarr(baseline, tmp_path / "baseline.zarr", baseline_encoding(baseline, scale_factor=0.1))
    write_geozarr(result, tmp_path / "anomaly.zarr", anomaly_encoding(result))

    reopened_baseline = open_geozarr(tmp_path / "baseline.zarr")
    reopened_anomaly = open_geozarr(tmp_path / "anomaly.zarr")

    assert reopened_baseline.rio.crs.to_epsg() == 32636
    assert reopened_anomaly.rio.crs.to_epsg() == 32636
    np.testing.assert_allclose(
        reopened_baseline["acc_mean"].values,
        baseline["acc_mean"].values,
        atol=0.05,  # int16 at scale_factor 0.1
        equal_nan=True,
    )


def test_lazy_write_survives_dask_chunks_off_the_zarr_grid(tmp_path, flux, seasons, baseline):
    """A lazy result inherits its input stores' chunks, which need not match.

    Left alone, ``to_zarr`` refuses the whole write ("would overlap multiple
    Dask chunks") -- at the end of the run, after every earlier stage has been
    paid for.
    """
    result = compute_anomaly(flux, seasons, baseline, QUERY, YEAR_MIN)
    misaligned = result.chunk({"y": (1, 1, 2), "x": 4})
    assert misaligned["acc"].chunks[1] == (1, 1, 2)  # off the grid the encoding sets

    encoding = anomaly_encoding(misaligned, chunks=(2, 4))
    write_geozarr(misaligned, tmp_path / "anomaly.zarr", encoding)

    reopened = open_geozarr(tmp_path / "anomaly.zarr")
    np.testing.assert_allclose(
        reopened["acc"].values, result["acc"].compute().values, equal_nan=True
    )


def test_anomaly_packing_leaves_the_unitless_fields_alone(tmp_path, flux, seasons, baseline):
    """A flux scale factor applies to the flux's units and to nothing else.

    ``anomaly_rel`` is a percentage and ``anomaly_z`` a count of standard
    deviations; packing those at the flux's resolution would round a z-score to
    whole sigma.
    """
    result = compute_anomaly(flux, seasons, baseline, QUERY, YEAR_MIN).compute()

    encoding = anomaly_encoding(result, scale_factor=0.1)

    assert encoding["anomaly_abs"]["dtype"] == "int16"
    assert encoding["anomaly_abs"]["scale_factor"] == 0.1
    assert encoding["anomaly_rel"]["dtype"] == "float32"
    assert encoding["anomaly_z"]["dtype"] == "float32"
    assert "scale_factor" not in encoding["anomaly_z"]
    # The two masks stay integer, and unpacked: zero means "not in season".
    assert encoding["DOS"]["dtype"] == "uint16"
    assert encoding["season"]["dtype"] == "uint8"

    write_geozarr(result, tmp_path / "anomaly.zarr", encoding)
    reopened = open_geozarr(tmp_path / "anomaly.zarr")
    np.testing.assert_allclose(
        reopened["anomaly_abs"].values,
        result["anomaly_abs"].values,
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

    write_geozarr(result, tmp_path / "anomaly.zarr", anomaly_encoding(result))
    reopened = open_geozarr(tmp_path / "anomaly.zarr")
    assert float(reopened["anomaly_abs"].isel(time=0, y=0, x=0)) < 0


def test_dos_and_season_survive_the_store_as_integers(tmp_path, flux, seasons, baseline):
    """Zero means "not in season" here -- a real value, not missing data.

    Declaring it as ``_FillValue`` would make xarray decode it back to NaN and
    promote both masks to float, which silently destroys the in-season mask the
    rest of the product (and the dashboard) keys off.
    """
    result = compute_anomaly(flux, seasons, baseline, "2021-01-15", YEAR_MIN).compute()
    assert int(result["DOS"].max()) == 0  # nothing is in season on this date

    write_geozarr(result, tmp_path / "anomaly.zarr", anomaly_encoding(result))
    reopened = open_geozarr(tmp_path / "anomaly.zarr")

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


# --- required_flux_start: how little flux an operational rerun needs ---------


def test_required_flux_start_follows_the_phenology(phenology, baseline):
    """The window begins at the period the active season started in.

    DOY 74 is 15 March in 2021, which falls in that month's second dekad -- so
    the flux is needed from 11 March, not from some fixed lookback.
    """
    start = required_flux_start(phenology, baseline, [QUERY])

    assert start == np.datetime64("2021-03-11")


def test_required_flux_start_spans_every_query_date(phenology, baseline):
    """Several dates take the earliest start among them, not the last one's."""
    start = required_flux_start(phenology, baseline, ["2021-06-15", "2021-08-01"])

    assert start == np.datetime64("2021-03-11")


def test_required_flux_start_is_bounded_by_the_stored_slots(flux, baseline, caplog):
    """A season older than the baseline's slot axis cannot be compared anyway.

    ``interpolate_slot`` clips past MAX_POS, so the curve flatlines however much
    flux is supplied. The window stops there and says so.
    """
    # SOSD -400 puts the season start well over a year before its nominal year.
    ancient = _phenology(flux, sosd=-400, eosd=232)

    with caplog.at_level("WARNING"):
        start = required_flux_start(ancient, baseline, [QUERY])

    query_idx = int(dates_to_abs_index([np.datetime64(QUERY, "D")], YEAR_MIN)[0])
    floor, _ = period_bounds(np.array([query_idx - (MAX_POS - 1)]), YEAR_MIN)
    assert start == floor[0]
    assert "clamping the flux window" in caplog.text


def test_required_flux_start_handles_a_date_with_no_season(phenology, baseline):
    """Out of season everywhere: still returns a date the caller can slice on."""
    start = required_flux_start(phenology, baseline, ["2021-01-05"])

    assert isinstance(start, np.datetime64)


def test_required_flux_start_needs_the_baseline_anchor_year(phenology, baseline):
    with pytest.raises(ValueError, match="no 'year_min' attribute"):
        required_flux_start(phenology, baseline.drop_attrs(), [QUERY])


def test_a_trimmed_flux_window_gives_the_same_answer(flux, phenology, baseline):
    """The load-bearing claim: trimming the flux changes nothing it computes.

    This is what lets an operational rerun open a few months of flux instead of
    the archive. If it ever fails, the windowing is wrong, not the trimming.
    """
    dates = ["2021-06-15", "2021-07-01"]

    whole = seasonal_anomalies(flux, phenology, baseline, dates).compute()

    start = required_flux_start(phenology, baseline, dates)
    trimmed = seasonal_anomalies(
        flux.sel(time=slice(start, None)), phenology, baseline, dates
    ).compute()

    # The trim has to be a real one, or the test proves nothing.
    assert flux.sel(time=slice(start, None)).sizes["time"] < flux.sizes["time"] / 4
    xr.testing.assert_allclose(whole, trimmed)


def test_an_operationally_appended_date_matches_a_single_pass(
    tmp_path, flux, phenology, baseline
):
    """The whole Stage 2 loop: build a store, then grow it a date at a time.

    Each later date is computed from a flux window trimmed to what its own
    season needs -- the operational case -- and must land exactly as it would
    have in a single pass over every date at once.
    """
    dates = ["2021-06-15", "2021-07-01", "2021-07-11"]

    at_once = tmp_path / "at_once.zarr"
    whole = seasonal_anomalies(flux, phenology, baseline, dates)
    write_geozarr(
        whole, at_once, anomaly_encoding(whole, scale_factor=0.1), progress=False
    )

    staged = tmp_path / "staged.zarr"
    first = seasonal_anomalies(flux, phenology, baseline, dates[:1])
    write_geozarr(
        first, staged, anomaly_encoding(first, scale_factor=0.1), progress=False
    )
    for date in dates[1:]:
        start = required_flux_start(phenology, baseline, [date])
        later = seasonal_anomalies(
            flux.sel(time=slice(start, None)), phenology, baseline, [date]
        )
        append_geozarr(later, staged, progress=False)

    xr.testing.assert_allclose(open_geozarr(at_once), open_geozarr(staged))
    assert open_geozarr(staged).rio.crs.to_epsg() == 32636
