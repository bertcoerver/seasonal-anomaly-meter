"""The input contract, and the two entry points that enforce it.

These are the checks that replaced the loader. When the package fetched its own
data the shape of these arrays was guaranteed; now a caller supplies them, and
the ways that can go wrong are mostly silent -- a mismatched grid pairs every
pixel with a stranger's season, a monthly series read as dekadal shifts every
date by years. Each of those gets a test here.
"""

import sys

import numpy as np
import pytest
import rioxarray  # noqa: F401  -- registers .rio
import xarray as xr

from seasonal_anomaly_meter import (
    DEKADAL,
    MONTHLY,
    as_flux,
    check_phenology,
    infer_resolution,
    seasonal_anomalies,
    seasonal_baseline,
)
from seasonal_anomaly_meter.calendar import period_bounds

YEAR_MIN = 2018
YEARS = [2018, 2019, 2020, 2021]
SHAPE = (4, 4)
SOSD, EOSD = 74, 232


def make_flux(units="gC/m2/day"):
    n_periods = len(YEARS) * 36
    starts, _ = period_bounds(np.arange(n_periods), YEAR_MIN)
    return xr.DataArray(
        np.ones((n_periods, *SHAPE), dtype="float32"),
        dims=("time", "y", "x"),
        coords={
            "time": starts.astype("datetime64[ns]"),
            "y": np.arange(SHAPE[0], dtype=float) * 300 + 100,
            "x": np.arange(SHAPE[1], dtype=float) * 300 + 100,
        },
        attrs={"units": units},
        name="NPP",
    ).rio.write_crs("EPSG:32636")


def make_phenology(flux, sosd=SOSD, eosd=EOSD):
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
def flux():
    return make_flux()


@pytest.fixture
def phenology(flux):
    return make_phenology(flux)


# ---- the package stands alone -------------------------------------------


def test_the_package_works_without_lazy_dino_or_xr_utils():
    """The point of the split: both unpublished siblings are optional.

    Run in a subprocess with an import blocker in front of ``sys.meta_path``,
    because the modules under test are already imported in this one -- an
    in-process check would pass whether or not the imports were still there.
    """
    import subprocess
    import textwrap

    script = textwrap.dedent(
        """
        import sys

        class Blocker:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in {"lazy_dino", "xr_utils"}:
                    raise ImportError(name + " is blocked")
                return None

        sys.meta_path.insert(0, Blocker())

        import numpy as np, rioxarray, xarray as xr
        sys.path.insert(0, %r)
        from test_inputs import make_flux, make_phenology
        from seasonal_anomaly_meter import seasonal_anomalies, seasonal_baseline

        flux = make_flux()
        phenology = make_phenology(flux)
        baseline = seasonal_baseline(flux, phenology).compute()
        anomalies = seasonal_anomalies(flux, phenology, baseline, "2021-06-15").compute()
        assert float(np.nanmax(np.abs(anomalies["anomaly_abs"].values))) < 1.0
        print("ok")
        """
    ) % str(__file__.rsplit("/", 1)[0])

    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr
    assert "ok" in done.stdout


# ---- as_flux -------------------------------------------------------------


def test_a_single_variable_dataset_is_accepted(flux):
    assert as_flux(flux.to_dataset()).equals(flux)


def test_an_ambiguous_dataset_asks_which_variable(flux):
    ds = flux.to_dataset().assign(other=flux)
    with pytest.raises(ValueError, match="pass variable="):
        as_flux(ds)
    assert as_flux(ds, "other").equals(flux)


def test_misnamed_dimensions_are_reported(flux):
    with pytest.raises(ValueError, match="lat"):
        as_flux(flux.rename({"y": "lat"}))


def test_a_non_datetime_time_axis_is_rejected(flux):
    renamed = flux.assign_coords(time=np.arange(flux.sizes["time"]))
    with pytest.raises(TypeError, match="datetime64"):
        as_flux(renamed)


def test_per_period_totals_are_warned_about(caplog):
    """Accumulation multiplies by days, so a total would be ~10x too big."""
    with caplog.at_level("WARNING"):
        as_flux(make_flux(units="gC/m2"))
    assert "per-day rate" in caplog.text

    caplog.clear()
    with caplog.at_level("WARNING"):
        as_flux(make_flux(units="gC/m2/day"))
    assert caplog.text == ""


# ---- check_phenology -----------------------------------------------------


def test_missing_phenology_variables_are_named(flux, phenology):
    with pytest.raises(ValueError, match="EOSD"):
        check_phenology(phenology.drop_vars("EOSD"), flux)


def test_a_missing_season_axis_is_reported(flux, phenology):
    with pytest.raises(ValueError, match="season"):
        check_phenology(phenology.isel(season=0, drop=True), flux)


def test_dates_in_place_of_day_of_year_are_rejected(flux, phenology):
    as_dates = phenology.assign(
        SOSD=phenology["SOSD"].astype("int64").astype("datetime64[D]")
    )
    with pytest.raises(TypeError, match="day-of-year"):
        check_phenology(as_dates, flux)


def test_out_of_range_day_of_year_is_rejected(flux, phenology):
    scaled = phenology.assign(SOSD=phenology["SOSD"] * 100)
    with pytest.raises(ValueError, match="day-of-year"):
        check_phenology(scaled, flux)


def test_a_cross_year_season_is_accepted(flux):
    """SOSD -61 and EOSD 416 are ordinary Copernicus values, not errors."""
    check_phenology(make_phenology(flux, sosd=-61, eosd=416), flux)


def test_a_different_grid_is_rejected(flux, phenology):
    shifted = phenology.assign_coords(x=phenology["x"] + 1e5)
    with pytest.raises(ValueError, match="align_phenology"):
        check_phenology(shifted, flux)


def test_same_shape_different_place_is_still_rejected(flux, phenology):
    """The failure the original missed: matching sizes, unrelated pixels."""
    elsewhere = phenology.assign_coords(
        y=phenology["y"].values + 4e6, x=phenology["x"].values + 2e5
    )
    with pytest.raises(ValueError, match="disagree on the"):
        check_phenology(elsewhere, flux)


# ---- resolution inference ------------------------------------------------


def test_dekadal_and_monthly_series_are_told_apart(flux):
    assert infer_resolution(flux["time"].values) is DEKADAL
    months = np.arange("2018-01", "2021-01", dtype="datetime64[M]")
    assert infer_resolution(months.astype("datetime64[D]")) is MONTHLY


def test_an_unsupported_cadence_is_refused():
    daily = np.arange("2018-01-01", "2018-03-01", dtype="datetime64[D]")
    with pytest.raises(ValueError, match="neither dekadal"):
        infer_resolution(daily)


def test_an_unsorted_time_axis_is_refused(flux):
    with pytest.raises(ValueError, match="strictly increasing"):
        infer_resolution(flux["time"].values[::-1])


# ---- the two entry points, end to end ------------------------------------


def test_baseline_and_anomalies_round_trip_from_plain_datasets(flux, phenology):
    """A user with two xarray objects and no loader gets the whole product."""
    baseline = seasonal_baseline(flux, phenology).compute()
    assert set(baseline.data_vars) == {"acc_mean", "acc_std", "acc_count"}
    assert baseline.attrs["year_min"] == YEAR_MIN
    assert baseline.attrs["units"] == "gC/m2"

    anomalies = seasonal_anomalies(flux, phenology, baseline, "2021-06-15").compute()
    # Flux is a constant 1 unit/day in every year, so every year accumulates
    # the same amount and the anomaly is zero to within the leap-day wobble.
    assert np.nanmax(np.abs(anomalies["anomaly_abs"].values)) < 1.0
    assert anomalies["DOS"].isel(time=0).values.min() > 0


def test_the_flux_is_never_resampled(flux, phenology):
    """WaPOR is the example grid: the output sits on the input's own pixels."""
    baseline = seasonal_baseline(flux, phenology)
    np.testing.assert_array_equal(baseline["y"].values, flux["y"].values)
    np.testing.assert_array_equal(baseline["x"].values, flux["x"].values)


def test_a_dataset_flux_works_throughout(flux, phenology):
    baseline = seasonal_baseline(flux.to_dataset(), phenology).compute()
    anomalies = seasonal_anomalies(flux.to_dataset(), phenology, baseline, "2021-06-15")
    assert anomalies.sizes["time"] == 1


def test_a_baseline_from_elsewhere_is_rejected(flux, phenology):
    baseline = seasonal_baseline(flux, phenology).compute()
    elsewhere = baseline.assign_coords(x=baseline["x"].values + 2e5)
    with pytest.raises(ValueError, match="flux and baseline"):
        seasonal_anomalies(flux, phenology, elsewhere, "2021-06-15")


def test_a_baseline_without_its_anchor_year_is_rejected(flux, phenology):
    """year_min ties both stages to one period axis; losing it shifts seasons."""
    baseline = seasonal_baseline(flux, phenology).compute()
    del baseline.attrs["year_min"]
    with pytest.raises(ValueError, match="year_min"):
        seasonal_anomalies(flux, phenology, baseline, "2021-06-15")


def test_phenology_is_forward_filled_to_the_query_year(flux, phenology):
    """Copernicus lags, so an operational date has no published season yet."""
    baseline = seasonal_baseline(flux, phenology).compute()
    stale = phenology.sel(year=slice(None, 2020))

    assumed = seasonal_anomalies(flux, stale, baseline, "2021-06-15").compute()
    assert bool((assumed["DOS"].isel(time=0) > 0).all())

    literal = seasonal_anomalies(
        flux, stale, baseline, "2021-06-15", forward_fill=False
    ).compute()
    assert bool((literal["DOS"].isel(time=0) == 0).all())


def test_a_phenology_too_thin_for_the_baseline_is_refused(flux, phenology):
    """The failure that produced an all-NaN store has to be loud.

    A phenology source that quietly returns fewer years than asked (a catalog
    whose item timestamps collapse a dozen annual products onto three) leaves
    every pixel below min_years, so the store is written, valid and empty.
    """
    thin = phenology.sel(year=[2018, 2019])
    with pytest.raises(ValueError, match="fewer than the 3 years"):
        seasonal_baseline(flux, thin)

    # ...and it is the *overlap* that counts, not the phenology's own axis: a
    # long phenology whose years sit outside the flux is just as empty.
    elsewhere = phenology.assign_coords(year=phenology["year"].values - 10)
    with pytest.raises(ValueError, match="no year within the flux"):
        seasonal_baseline(flux, elsewhere)


def test_a_thin_phenology_is_allowed_when_min_years_says_so(flux, phenology):
    """The guard states the rule, it does not add one: lowering min_years works."""
    thin = phenology.sel(year=[2018, 2019])
    baseline = seasonal_baseline(flux, thin, min_years=2).compute()
    assert bool((baseline["acc_count"] >= 2).any())
