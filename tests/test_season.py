"""Season windows: converting phenology to flux-axis indices and picking one."""

import numpy as np
import pytest
import xarray as xr

from seasonal_anomaly_meter.calendar import doy_year_to_abs_index
from seasonal_anomaly_meter.season import (
    forward_fill_phenology,
    season_indices,
    select_season,
)

YEAR_MIN = 2018
YEARS = [2018, 2019, 2020]


def phenology(sosd, eosd, qa=0, *, seasons=(1,), years=YEARS, shape=(2, 2)):
    """A phenology Dataset from per-(season, year) day-of-year scalars or arrays."""
    dims = ("season", "year", "y", "x")
    full = (len(seasons), len(years), *shape)

    def field(value, dtype):
        array = np.asarray(value, dtype=dtype)
        if array.ndim == 0:
            return np.full(full, array, dtype=dtype)
        # (season, year) given -- broadcast over space.
        return np.broadcast_to(array[..., None, None], full).astype(dtype)

    return xr.Dataset(
        {
            "SOSD": (dims, field(sosd, "float32")),
            "EOSD": (dims, field(eosd, "float32")),
            "QA": (dims, field(qa, "uint8")),
        },
        coords={
            "season": list(seasons),
            "year": list(years),
            "y": np.arange(shape[0], dtype=float),
            "x": np.arange(shape[1], dtype=float),
        },
    )


def test_season_indices_match_the_calendar():
    seasons = season_indices(phenology(74, 232), YEAR_MIN)
    expected_start = doy_year_to_abs_index(np.array([74]), np.array([2018]), YEAR_MIN)[0]
    assert seasons["start_idx"].isel(season=0, year=0, y=0, x=0) == expected_start
    assert str(seasons["start_date"].isel(season=0, year=0, y=0, x=0).values)[:10] == "2018-03-15"


def test_missing_phenology_yields_no_season():
    seasons = season_indices(phenology(np.nan, np.nan), YEAR_MIN)
    assert bool(seasons["start_idx"].isnull().all())
    assert bool(seasons["start_date"].isnull().all())


def test_season_ending_before_it_starts_is_dropped():
    """The source data contains these; left in, they accumulate nothing silently."""
    seasons = season_indices(phenology(232, 74), YEAR_MIN)
    assert bool(seasons["start_idx"].isnull().all())


def test_cross_year_season_spans_the_boundary():
    """SOSD in the previous year, EOSD in the next -- a real Copernicus pattern."""
    seasons = season_indices(phenology(-61, 120), YEAR_MIN)
    start = seasons["start_idx"].isel(season=0, year=1, y=0, x=0)
    end = seasons["end_idx"].isel(season=0, year=1, y=0, x=0)
    assert float(start) < float(end)
    assert str(seasons["start_date"].isel(season=0, year=1, y=0, x=0).values)[:10] == "2018-10-31"


def test_select_picks_the_active_season():
    seasons = season_indices(phenology(74, 232), YEAR_MIN)
    picked = select_season(seasons, np.datetime64("2019-05-30"))
    assert bool(picked["in_season"].all())
    assert int(picked["season"].isel(y=0, x=0)) == 1
    assert str(picked["start_date"].isel(y=0, x=0).values)[:10] == "2019-03-15"


def test_select_reports_no_season_between_seasons():
    seasons = season_indices(phenology(74, 232), YEAR_MIN)
    picked = select_season(seasons, np.datetime64("2019-01-05"))
    assert not bool(picked["in_season"].any())
    assert int(picked["season"].isel(y=0, x=0)) == 0
    assert bool(picked["start_idx"].isnull().all())


def test_overlapping_seasons_resolved_by_quality_then_duration():
    """Two labels active at once: lowest QA wins, longer season breaks ties."""
    query = np.datetime64("2019-05-30")

    # season 1: long but poor quality; season 2: shorter but clean.
    sosd = np.array([[74, 74, 74], [100, 100, 100]], dtype="float32")
    eosd = np.array([[300, 300, 300], [200, 200, 200]], dtype="float32")
    qa = np.array([[50, 50, 50], [10, 10, 10]], dtype="uint8")
    picked = select_season(
        season_indices(phenology(sosd, eosd, qa, seasons=(1, 2)), YEAR_MIN), query
    )
    assert int(picked["season"].isel(y=0, x=0)) == 2  # better QA wins

    # With equal QA the longer season wins instead.
    equal_qa = np.array([[10, 10, 10], [10, 10, 10]], dtype="uint8")
    picked = select_season(
        season_indices(phenology(sosd, eosd, equal_qa, seasons=(1, 2)), YEAR_MIN), query
    )
    assert int(picked["season"].isel(y=0, x=0)) == 1


def test_select_matches_between_numpy_and_dask():
    """The kernel runs chunked in the pipeline and unchunked in tests."""
    seasons = season_indices(phenology(74, 232), YEAR_MIN)
    query = np.datetime64("2019-05-30")

    eager = select_season(seasons, query)
    lazy = select_season(seasons.chunk({"y": 1, "x": 1}), query).compute()
    for name in ("start_idx", "end_idx", "season", "in_season"):
        np.testing.assert_array_equal(eager[name].values, lazy[name].values)


def test_forward_fill_extends_and_flags():
    filled = forward_fill_phenology(phenology(74, 232), 2022)
    assert filled["year"].values.tolist() == [2018, 2019, 2020, 2021, 2022]
    assert filled["forward_filled"].values.tolist() == [False, False, False, True, True]
    # The assumed years repeat the last real one's day-of-year values.
    assert float(filled["SOSD"].sel(year=2022).isel(season=0, y=0, x=0)) == 74


def test_forward_fill_is_a_noop_when_already_covered():
    filled = forward_fill_phenology(phenology(74, 232), 2019)
    assert filled["year"].values.tolist() == YEARS
    assert not filled["forward_filled"].values.any()


@pytest.mark.parametrize("target", [2021, 2023])
def test_forward_fill_keeps_year_in_one_chunk(target):
    """Concatenating year-by-year otherwise leaves `year` chunked inconsistently."""
    filled = forward_fill_phenology(phenology(74, 232).chunk({"y": 1}), target)
    for variable in filled.data_vars.values():
        assert len(variable.chunksizes.get("year", (1,))) == 1
