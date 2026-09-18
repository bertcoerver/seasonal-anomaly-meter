"""Producing the package's two inputs from WaPOR and Copernicus, via lazy_dino.

**This is example code, not part of the package.** ``seasonal_anomaly_meter``
takes the flux and the phenology as arrays you hand it, and does not know where
they came from; this module is one way to fetch them, and the one the
methodology was developed against. Keeping it here is what lets the package
itself depend on nothing but PyPI.

``lazy_dino`` does the discovery, credential plumbing and windowed lazy reads;
nothing here opens a file itself. A run starts at :func:`get_wapor_tile`, which
resolves one tile of WaPOR's UTM grid, and both loaders take that tile.

**The flux never moves.** WaPOR is the example grid, so :func:`load_flux`
returns the tile on its native UTM pixels and the phenology is warped onto it
(by the package's ``align_phenology``, using nearest neighbour). That is the
reverse of the original methodology, which interpolated every dekad of flux
onto the Copernicus grid.

Both loaders return a Dataset, chunked by their own ``chunk`` argument. Which
axes have to stay whole is a property of the product and of what the pipeline
does with it, not of any one script, so it lives here rather than in a
``.chunk`` call at the call site.
"""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr
from lazy_dino import CDSE, WAPOR

from seasonal_anomaly_meter import align_phenology
from wapor_tiles import Tile, list_tiles

logger = logging.getLogger(__name__)

__all__ = ["get_wapor_tile", "load_flux", "load_phenology", "LSP_COLLECTION"]

#: Copernicus Land Surface Phenology, 300 m global, yearly. Season start/end are
#: published as day-of-year, two seasons (``s1``/``s2``) per year, from 2014.
LSP_COLLECTION = "clms_lsp_global_300m_yearly_v2_cog"

#: Degrees of halo added to the phenology request. ``lazy_dino``'s pixel-based
#: ``spatial_buffer`` raises for LSP variables (their catalog entries declare no
#: pixel size), so the halo is expressed in degrees instead. 0.02 deg is ~2 km,
#: comfortably more than the one source pixel a nearest-neighbour warp needs.
PHENOLOGY_PAD_DEG = 0.02


def get_wapor_tile(variable: str, code: str) -> Tile:
    """One WaPOR UTM tile by its MGRS grid-zone code.

    ``get_wapor_tile("L1-UTM-NPP-D", "36Q")`` -- the grid every run starts from,
    since WaPOR publishes one file per tile and a tile is the largest area that
    stays in a single CRS. What comes back is the tile's extent and grid, not
    its data; :func:`load_flux` turns it into arrays.

    The grid is enumerated from WaPOR's catalog, so this needs network access
    the first time it is asked about a variable (``list_tiles`` caches it).
    """
    tiles = list_tiles(variable)
    try:
        return tiles[code]
    except KeyError:
        raise KeyError(
            f"{code!r} is not a tile of {variable!r}'s grid "
            f"({len(tiles)} tiles available)."
        ) from None


def load_flux(
    variable: str,
    tile: Tile,
    time_range: tuple[str, str],
    *,
    chunk: int | None = None,
    reader=None,
) -> xr.Dataset:
    """One WaPOR variable over one UTM tile, on the tile's native grid.

    Returns a lazy, dask-backed ``(time, y, x)`` Dataset holding ``variable``,
    in the tile's UTM CRS. This grid is the example grid: every other input is
    warped onto it. A Dataset rather than a DataArray because that is what
    ``to_zarr`` takes, and caching this to a store is the normal thing to do
    with it; the package's ``as_flux`` accepts either.

    A tile's lon/lat envelope is a curved quadrilateral, so its bounding box
    necessarily overlaps neighbouring tiles -- including ones in adjacent UTM
    zones. ``lazy_dino`` answers such a request with a ``DataTree`` keyed by
    EPSG (it mosaics same-CRS tiles but never reprojects), so this function
    picks the node matching ``tile.epsg`` and then trims to the tile's exact
    native bounds. The result is the published tile, pixel for pixel.

    ``chunk`` sets the spatial chunk in pixels; the ``time`` axis is always kept
    whole, because the accumulation kernel builds one prefix sum over the full
    series and a split time axis would force a shuffle for every day-of-season
    slot. Leave it ``None`` to keep whatever chunking ``lazy_dino`` returned.
    """
    minx, miny, maxx, maxy = tile.lonlat_bounds()
    kwargs = {"reader": reader} if reader is not None else {}
    source = WAPOR(
        [variable],
        x_range=(minx, maxx),
        y_range=(miny, maxy),
        time_range=time_range,
        **kwargs,
    )
    result = source.dataset

    if isinstance(result, xr.DataTree):
        node = f"EPSG_{tile.epsg}"
        if node not in result.children:
            raise ValueError(
                f"tile {tile.code} expects {node}, but WaPOR returned "
                f"{sorted(result.children)} for {variable!r}. The tile's grid is "
                "not among the published data for this period."
            )
        da = result[node].ds[variable]
    else:
        da = result[variable]

    da = _trim_to_tile(da, tile)
    height, width = tile.shape
    if da.sizes["y"] != height or da.sizes["x"] != width:
        raise ValueError(
            f"tile {tile.code}: expected {height}x{width} pixels, got "
            f"{da.sizes['y']}x{da.sizes['x']}. The catalog grid and the "
            "published rasters disagree."
        )

    ds = da.to_dataset(name=variable)
    if chunk is not None:
        ds = ds.chunk({"time": -1, "y": chunk, "x": chunk})
    return ds


def load_phenology(
    tile: Tile,
    years: list[int] | range,
    *,
    seasons: tuple[int, ...] = (1, 2),
    include_qa: bool = True,
    pad_deg: float = PHENOLOGY_PAD_DEG,
    example: xr.DataArray | xr.Dataset | None = None,
    chunk: int | None = None,
) -> xr.Dataset:
    """Copernicus season start/end for one tile, warped onto the tile's grid.

    Returns a Dataset of ``SOSD``, ``EOSD`` (and ``QA`` when ``include_qa``)
    with dims ``(season, year, y, x)`` on ``example``'s grid -- the WaPOR grid
    from :func:`load_flux`. ``SOSD``/``EOSD`` are day-of-year (float32, NaN
    where no season was detected); ``QA`` is uint8, lower is better.

    ``example`` must be supplied to place the result on the flux grid. Without
    it the phenology is returned on its native EPSG:4326 grid, which is useful
    for inspection but is not what the pipeline consumes.

    ``chunk`` sets the spatial chunk in pixels, applied after the warp; the
    ``season`` and ``year`` axes are always kept whole, since a pixel's season
    lookup reads all of both. Leave it ``None`` to keep the warp's own chunking.
    """
    years = list(years)
    variables = [
        f"{LSP_COLLECTION}.lsp300_{measure}_s{season}"
        for season in seasons
        for measure in (("sosd", "eosd", "qa") if include_qa else ("sosd", "eosd"))
    ]
    minx, miny, maxx, maxy = tile.lonlat_bounds(pad_deg=pad_deg)
    source = CDSE(
        variables,
        x_range=(minx, maxx),
        y_range=(miny, maxy),
        time_range=_phenology_time_range(variables[0], years),
    )
    raw = source.dataset
    if isinstance(raw, xr.DataTree):
        raise ValueError(
            "the phenology request spans several CRSs, which should be "
            "impossible for a global EPSG:4326 product; got "
            f"{sorted(raw.children)}."
        )

    ds = _stack_seasons(raw, seasons=seasons, include_qa=include_qa)
    ds = _label_years(ds, years)
    if example is not None:
        ds = align_phenology(ds, example)

    if chunk is not None:
        ds = ds.chunk({"y": chunk, "x": chunk, "season": -1, "year": -1})
    return ds


def _phenology_time_range(variable: str, years: list[int]) -> tuple[str, str]:
    """The requested years, clipped to what the LSP catalog actually publishes.

    Each LSP year is stamped 1 January, so asking up to ``1 January`` of the
    last wanted year covers it exactly -- asking to 31 December would overshoot
    the catalog's upper bound and be rejected outright.

    Copernicus phenology lags the flux by a year or more, so a caller asking
    for the current year is normal rather than an error; the range is clipped
    and the shortfall is handled by
    :func:`~seasonal_anomaly_meter.season.forward_fill_phenology`.
    """
    time_dim = next(d for d in CDSE.dimensions if d.role == "time")
    available = CDSE.catalog.resolved_bounds(variable, time_dim)

    start = np.datetime64(f"{min(years)}-01-01")
    end = np.datetime64(f"{max(years)}-01-01")
    if available is not None:
        first, last = (np.datetime64(b, "D") for b in available)
        if end > last:
            logger.info(
                "phenology is published up to %s; clipping the request for %d "
                "and forward-filling later years.",
                last,
                max(years),
            )
        start, end = max(start, first), min(end, last)
        if end < start:
            raise ValueError(
                f"no Copernicus phenology for {min(years)}-{max(years)}; the "
                f"product covers {first} to {last}."
            )
    return str(start), str(end)


def _trim_to_tile(da: xr.DataArray, tile: Tile) -> xr.DataArray:
    """Clip to the tile's native bounds, honouring each axis's direction."""
    x0, x1 = tile.x_range
    y0, y1 = tile.y_range
    if float(da.x[0]) > float(da.x[-1]):
        x0, x1 = x1, x0
    if float(da.y[0]) > float(da.y[-1]):
        y0, y1 = y1, y0
    return da.sel(x=slice(x0, x1), y=slice(y0, y1))


def _stack_seasons(
    raw: xr.Dataset, *, seasons: tuple[int, ...], include_qa: bool
) -> xr.Dataset:
    """Fold ``lsp300_<measure>_s<n>`` data_vars into a ``season`` dimension."""
    measures = {"sosd": "SOSD", "eosd": "EOSD"}
    if include_qa:
        measures["qa"] = "QA"

    data_vars = {}
    for measure, name in measures.items():
        per_season = [
            raw[f"{LSP_COLLECTION}.lsp300_{measure}_s{season}"] for season in seasons
        ]
        stacked = xr.concat(per_season, dim="season")
        stacked = stacked.assign_coords(season=list(seasons))
        # Everything stays float here, QA included: lazy_dino decodes nodata to
        # NaN, and casting NaN straight to uint8 is undefined behaviour. QA is
        # returned to an integer dtype in _warp_to_example, once its NaN has
        # been replaced by the documented 255 fill.
        data_vars[name] = stacked.astype(np.float32)
    return xr.Dataset(data_vars)


def _label_years(ds: xr.Dataset, years: list[int]) -> xr.Dataset:
    """Replace the LSP ``time`` axis with an integer ``year`` coordinate."""
    if "time" not in ds.dims:
        raise ValueError(f"phenology has no time dimension; dims are {dict(ds.sizes)}.")
    stamped = ds["time"].dt.year.values
    ds = ds.rename({"time": "year"}).assign_coords(year=stamped)
    missing = sorted(set(years) - set(stamped.tolist()))
    if missing:
        logger.info(
            "phenology is missing year(s) %s; forward-fill them with "
            "season.forward_fill_phenology before use.",
            missing,
        )
    return ds.sortby("year")
