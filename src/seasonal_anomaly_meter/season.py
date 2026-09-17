"""Turning phenology into season windows on the flux time axis.

A phenology product publishes a season as two day-of-year numbers per (season,
year, pixel): ``SOSD`` and ``EOSD``. Those numbers routinely fall **outside** their
nominal year -- a SOSD of ``-61`` means the season began in late October of the
previous year, an EOSD of ``416`` means it ends in February of the next one.
:func:`season_indices` converts them to absolute period indices on the same
axis as the flux, where a cross-year season is just a larger number and needs
no special casing.

The original implementation materialised a ``(season, dekad, y, x)`` "progress
of season" cube to drive a ``flox`` groupby. Nothing here does: the accumulation
kernel indexes the flux directly from the season's start and end, which avoids
building a cube whose size is ``n_periods`` times the raster.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from seasonal_anomaly_meter.calendar import (
    DEKADAL,
    TemporalResolution,
    doy_year_to_abs_index,
    period_bounds,
)

__all__ = [
    "MAX_POS",
    "forward_fill_phenology",
    "season_indices",
    "select_season",
]

#: Highest day-of-season slot the baseline stores. Copernicus seasons are
#: bounded at ~36 dekads and a cross-year season adds a few more, so 49 leaves
#: headroom while keeping the POS axis a fixed, known size -- which means the
#: baseline store's shape never depends on reading the data first.
MAX_POS = 49


def forward_fill_phenology(ds: xr.Dataset, target_year: int) -> xr.Dataset:
    """Extend ``ds``'s ``year`` axis up to ``target_year`` by repeating the last year.

    Phenology lags the flux: a flux product typically publishes a period within
    weeks, while the season parameters for a year appear a year or more later
    (Copernicus LSP and WaPOR are the worked example). To run
    operationally we assume the current season starts and ends on the same
    days of the year as the most recent year we have.

    Existing years are untouched, and a ``target_year`` already covered is a
    no-op. The appended years carry a ``forward_filled`` coordinate flag so a
    downstream consumer can tell a real season from an assumed one.
    """
    max_year = int(ds["year"].max())
    known = ds["year"].values.tolist()
    if target_year <= max_year:
        return ds.assign_coords(
            forward_filled=("year", np.zeros(len(known), dtype=bool))
        )

    last = ds.sel(year=max_year, drop=True)
    filled = [ds] + [
        last.expand_dims(year=[yr]) for yr in range(max_year + 1, target_year + 1)
    ]
    out = xr.concat(filled, dim="year")
    # Each appended year arrives as its own chunk, which leaves `year` chunked
    # inconsistently against the rest of the dataset. Collapse it back to one
    # chunk -- it is a handful of elements either way, and consumers that
    # reduce over it need it whole.
    if any(v.chunks is not None for v in out.data_vars.values()):
        out = out.chunk({"year": -1})
    flags = np.array([yr > max_year for yr in out["year"].values], dtype=bool)
    return out.assign_coords(forward_filled=("year", flags))


def season_indices(
    phenology: xr.Dataset,
    year_min: int,
    resolution: TemporalResolution = DEKADAL,
) -> xr.Dataset:
    """Absolute period index and calendar date of each season's start and end.

    Takes the caller-supplied ``SOSD``/``EOSD`` day-of-year fields (see
    :mod:`seasonal_anomaly_meter.inputs`) and returns
    ``start_idx``/``end_idx`` (float, NaN where the pixel has no
    season that year) plus ``start_date``/``end_date`` (``datetime64[D]``,
    NaT likewise).

    A season is dropped when either bound is missing, or when it ends before it
    starts -- a combination that occurs in the source data and would otherwise
    produce a negative-length season that silently accumulates nothing.
    """
    year = phenology["year"].broadcast_like(phenology["SOSD"])

    start_idx = _to_abs_index(phenology["SOSD"], year, year_min, resolution)
    end_idx = _to_abs_index(phenology["EOSD"], year, year_min, resolution)

    valid = start_idx.notnull() & end_idx.notnull() & (end_idx >= start_idx)
    start_idx = start_idx.where(valid)
    end_idx = end_idx.where(valid)

    start_date = _doy_to_date(phenology["SOSD"], year).where(valid)
    end_date = _doy_to_date(phenology["EOSD"], year).where(valid)

    out = xr.Dataset(
        {
            "start_idx": start_idx,
            "end_idx": end_idx,
            "start_date": start_date,
            "end_date": end_date,
        }
    )
    if "QA" in phenology:
        out["QA"] = phenology["QA"]
    return out


def select_season(
    seasons: xr.Dataset,
    query_date,
    *,
    use_qa: bool = True,
) -> xr.Dataset:
    """Pick the one season active on ``query_date`` for every pixel.

    Activity is tested against the season's **dates**, not its period indices.
    A season starting on 18 March shares its dekad with a query on 15 March, so
    an index-level test would call it active three days before it begins --
    yielding a zero day-of-season alongside a non-zero season label, and an
    accumulation that trims more off both ends of the dekad than the dekad
    holds, which comes out negative.

    A pixel can have several candidate seasons at one moment: two Copernicus
    season labels (``s1``/``s2``) may overlap, and -- once phenology has been
    forward-filled -- a real season and an assumed one can both be active. The
    reference resolved this as: the **latest year** that brackets the query
    wins per season label, then the label with the **lowest QA** wins, with the
    **longer** season breaking ties. That ordering is preserved here.

    Returns ``start_idx``/``start_date``/``end_idx``/``season``/``in_season``
    reduced over the ``season`` and ``year`` dimensions. ``season`` is 0 and
    ``in_season`` is False where no season is active.
    """
    # The kernel reduces over season and year together, so both must be whole
    # in each chunk. They are tiny (2 labels, a handful of years), so this
    # never costs anything -- unlike leaving it to `allow_rechunk`, which would
    # also silently permit a rechunk of the spatial dims.
    if any(v.chunks is not None for v in seasons.data_vars.values()):
        seasons = seasons.chunk({"season": -1, "year": -1})

    qa = seasons["QA"] if (use_qa and "QA" in seasons) else xr.zeros_like(
        seasons["start_idx"], dtype=np.uint8
    )

    # Expressed as one kernel over the (season, year) core dims rather than as
    # xarray `.isel` with computed indexers: selecting with a dask-backed index
    # array is not supported, and materialising the indexer just to pick two
    # numbers per pixel would defeat the lazy read.
    start_idx, start_date, end_idx, label, in_season = xr.apply_ufunc(
        _select_kernel,
        seasons["start_idx"],
        seasons["start_date"],
        seasons["end_idx"],
        seasons["end_date"],
        qa,
        input_core_dims=[["season", "year"]] * 5,
        output_core_dims=[[], [], [], [], []],
        dask="parallelized",
        output_dtypes=[np.float32, "datetime64[ns]", np.float32, np.uint8, bool],
        kwargs={
            "query_date": np.datetime64(query_date, "ns"),
            "labels": seasons["season"].values,
        },
    )

    return xr.Dataset(
        {
            "start_idx": start_idx,
            "start_date": start_date,
            "end_idx": end_idx,
            "season": label,
            "in_season": in_season,
        }
    )


def _select_kernel(start_idx, start_date, end_idx, end_date, qa, *, query_date, labels):
    """Pick one (season, year) per pixel. Core dims trail, as apply_ufunc wants."""
    to_lead = lambda a: np.moveaxis(a, (-2, -1), (0, 1))  # noqa: E731
    start_idx, end_idx = to_lead(start_idx), to_lead(end_idx)
    start_date, end_date, qa = to_lead(start_date), to_lead(end_date), to_lead(qa)

    # NaT compares False against everything, so pixels with no season fall out
    # here without a separate validity mask.
    active = (start_date <= query_date) & (query_date <= end_date)  # (season, year, *sp)

    # Latest active year per season label. argmax on the reversed year axis
    # finds the last True, so a forward-filled current season beats the same
    # label's historical years.
    active_any_year = active.any(axis=1)
    last_active = active.shape[1] - 1 - active[:, ::-1].argmax(axis=1)
    gather_year = last_active[:, np.newaxis]

    pick = lambda a: np.take_along_axis(a, gather_year, axis=1)[:, 0]  # noqa: E731
    s_idx, e_idx = pick(start_idx), pick(end_idx)
    s_date, s_qa = pick(start_date), pick(qa)

    # Lowest QA wins, longest season breaks the tie -- combined into one score
    # so a single argmax does both. 255 is the QA product's nodata.
    duration = np.where(active_any_year, e_idx - s_idx, -np.inf)
    quality = np.where(active_any_year, s_qa.astype(np.float64), 255.0)
    score = np.where(active_any_year, -quality * 1e6 + duration, -np.inf)

    season_pos = score.argmax(axis=0)[np.newaxis]
    take = lambda a: np.take_along_axis(a, season_pos, axis=0)[0]  # noqa: E731
    in_season = take(active_any_year)

    label = np.asarray(labels)[season_pos[0]]
    return (
        np.where(in_season, take(s_idx), np.nan).astype(np.float32),
        np.where(in_season, take(s_date), np.datetime64("NaT", "ns")).astype("datetime64[ns]"),
        np.where(in_season, take(e_idx), np.nan).astype(np.float32),
        np.where(in_season, label, 0).astype(np.uint8),
        in_season,
    )


def _to_abs_index(doy, year, year_min: int, resolution: TemporalResolution):
    """Vectorised day-of-year -> absolute period index, keeping NaN as NaN."""

    def _kernel(d, y):
        out = np.full(d.shape, np.nan, dtype=float)
        ok = np.isfinite(d)
        if ok.any():
            out[ok] = doy_year_to_abs_index(
                d[ok], np.broadcast_to(y, d.shape)[ok], year_min, resolution
            )
        return out

    return xr.apply_ufunc(
        _kernel, doy, year, dask="parallelized", output_dtypes=[float]
    )


def _doy_to_date(doy, year):
    """Vectorised ``(day-of-year, year)`` -> ``datetime64[D]``, NaN -> NaT."""

    def _kernel(d, y):
        out = np.full(d.shape, np.datetime64("NaT", "D"), dtype="datetime64[D]")
        ok = np.isfinite(d)
        if ok.any():
            years = np.broadcast_to(y, d.shape)[ok].astype("int64")
            out[ok] = years.astype(str).astype("datetime64[D]") + (
                d[ok].astype("int64") - 1
            ).astype("timedelta64[D]")
        return out

    return xr.apply_ufunc(
        _kernel, doy, year, dask="parallelized", output_dtypes=["datetime64[D]"]
    )


def period_day_bounds(idx, year_min: int, resolution: TemporalResolution = DEKADAL):
    """``period_bounds`` for a possibly-NaN float index array, NaN -> NaT."""
    idx = np.asarray(idx)
    ok = np.isfinite(idx)
    start = np.full(idx.shape, np.datetime64("NaT", "D"), dtype="datetime64[D]")
    end = np.full(idx.shape, np.datetime64("NaT", "D"), dtype="datetime64[D]")
    if ok.any():
        s, e = period_bounds(idx[ok].astype(np.int64), year_min, resolution)
        start[ok], end[ok] = s, e
    return start, end
