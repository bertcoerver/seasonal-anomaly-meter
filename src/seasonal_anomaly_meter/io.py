"""Zarr encoding for the two stores.

The baseline store is the big one -- ``(season, pos, y, x)`` is ``2 x 49``
rasters per tile -- so it is packed to ``int16`` with a scale factor. The
anomaly store is per query date and small by comparison.

**Nothing signed is ever stored unsigned.** The original pipeline wrote its
accumulations as ``uint16``, and because an accumulated anomaly can legitimately
be negative, deficits wrapped around to ~65529 and had to be masked in the
dashboard's front end. Every anomaly variable here is signed, and
:func:`check_packing_range` refuses to write values a scale factor cannot hold
rather than letting them saturate quietly.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

__all__ = [
    "PACKED_FILL",
    "baseline_encoding",
    "anomaly_encoding",
    "check_packing_range",
    "open_zarr",
    "write_zarr",
]

#: ``int16`` fill value for packed variables. ``-32768`` is the type's minimum,
#: so it can never collide with a real scaled measurement.
PACKED_FILL = -32768

#: Largest magnitude an ``int16`` can represent before saturating.
_INT16_MAX = 32767

_COMPRESSOR_KWARGS = {"cname": "zstd", "clevel": 5, "shuffle": "bitshuffle"}


def _compressors() -> list[dict]:
    """Blosc/zstd with bitshuffle -- the codec the reference store used.

    Given in zarr's own JSON metadata form rather than as a
    ``zarr.codecs.BloscCodec``. The two land as the same codec on disk, but the
    dict means nothing in this package imports zarr: every encoding here stays a
    plain dict, and zarr is needed only by whoever finally calls ``to_zarr``.

    Returns a fresh list each call, so a caller editing one variable's encoding
    cannot reach into another's.
    """
    return [{"name": "blosc", "configuration": dict(_COMPRESSOR_KWARGS)}]


def baseline_encoding(
    ds: xr.Dataset,
    *,
    scale_factor: float | None = 0.1,
    chunks: tuple[int, int] = (256, 256),
) -> dict:
    """Zarr encoding for :func:`~seasonal_anomaly_meter.baseline.build_baseline`.

    ``acc_mean`` and ``acc_std`` are packed to ``int16`` at ``scale_factor``
    resolution, which halves the store against ``float32``. With the default
    ``0.1`` the representable range is +/-3276.7 -- comfortable for a season of
    accumulated NPP (gC/m2) or ET (mm), but check it against your variable and
    raise the scale factor if needed. Pass ``scale_factor=None`` to store
    ``float32`` instead and sidestep the question entirely.

    The ``pos`` axis is kept whole in each chunk: the anomaly stage reads a
    pixel's entire season curve, so splitting it would turn one read into
    several.
    """
    n_pos = ds.sizes["pos"]
    n_season = ds.sizes["season"]
    chunk_shape = (n_season, n_pos, *chunks)

    encoding: dict[str, dict] = {}
    for name in ("acc_mean", "acc_std"):
        if name not in ds:
            continue
        encoding[name] = {"chunks": chunk_shape, "compressors": _compressors()}
        if scale_factor is not None:
            encoding[name].update(
                dtype="int16", scale_factor=scale_factor, _FillValue=PACKED_FILL
            )
        else:
            encoding[name]["dtype"] = "float32"
    if "acc_count" in ds:
        encoding["acc_count"] = {
            "chunks": chunk_shape,
            "dtype": "uint8",
            "compressors": _compressors(),
        }
    return encoding


def anomaly_encoding(
    ds: xr.Dataset,
    *,
    chunks: tuple[int, int] = (256, 256),
) -> dict:
    """Zarr encoding for :func:`~seasonal_anomaly_meter.anomaly.compute_anomaly`.

    One date per chunk along ``time`` so an operational rerun appends without
    rewriting earlier dates. The anomaly fields stay ``float32``: they are
    signed, they are the product people read, and at one date per file the
    space saved by packing is not worth the extra failure mode.
    """
    chunk_shape = (1, *chunks)
    encoding = {}
    for name in ds.data_vars:
        spec: dict = {"chunks": chunk_shape, "compressors": _compressors()}
        # DOS and season carry no _FillValue on purpose. Zero is a *meaningful*
        # value in both ("this pixel is not in season") and is the mask the rest
        # of the product keys off. Declaring it as the fill would make xarray
        # decode it back to NaN and promote the array to float, quietly
        # destroying both the mask and the integer storage.
        if name == "DOS":
            spec["dtype"] = "uint16"
        elif name == "season":
            spec["dtype"] = "uint8"
        else:
            spec["dtype"] = "float32"
        encoding[name] = spec
    return encoding


def check_packing_range(ds: xr.Dataset, scale_factor: float, variables=("acc_mean", "acc_std")):
    """Raise if packing ``ds`` at ``scale_factor`` would saturate ``int16``.

    Computes the data, so call it on a materialised (or cheaply computable)
    Dataset. Saturation is exactly the failure that produced the original
    store's wrapped-around deficits, and it is silent -- worth one pass to
    rule out.
    """
    limit = _INT16_MAX * scale_factor
    for name in variables:
        if name not in ds:
            continue
        peak = float(abs(ds[name]).max())
        if not np.isfinite(peak):
            continue
        if peak > limit:
            raise ValueError(
                f"{name} peaks at {peak:.1f}, beyond the +/-{limit:.1f} that "
                f"int16 at scale_factor={scale_factor} can hold. Pass a larger "
                "scale_factor (coarser resolution) or scale_factor=None to "
                "store float32."
            )


def open_zarr(store, **kwargs) -> xr.Dataset:
    """Open a store written by :func:`write_zarr`, with its CRS intact.

    Plain ``xr.open_zarr`` leaves ``spatial_ref`` as an ordinary data variable,
    so ``.rio.crs`` comes back ``None`` and the dataset looks unreferenced even
    though the store is fine -- GDAL and QGIS read it correctly either way.
    Promoting it back to a coordinate needs ``decode_coords="all"``, which is
    easy to forget and confusing when missed, so it is the default here.
    """
    kwargs.setdefault("decode_coords", "all")
    kwargs.setdefault("consolidated", False)
    return xr.open_zarr(store, **kwargs)


def write_zarr(ds: xr.Dataset, store, encoding: dict, *, mode: str = "w") -> None:
    """Write ``ds`` with GeoZarr CRS attributes so GDAL and QGIS can read it.

    xarray omits the CF ``grid_mapping`` attribute that points a variable at
    its ``spatial_ref`` coordinate, which leaves the store unreadable as
    geodata; ``xr_utils.set_geozarr_attrs`` puts it back.

    It does so by writing into each variable's ``.encoding``, which an explicit
    ``encoding=`` argument to ``to_zarr`` replaces wholesale -- so the two have
    to be merged rather than passed independently. Getting this wrong produces
    a store that looks fine until something tries to read its CRS.

    ``xr_utils`` is an optional dependency, so this function is too: the
    encodings from :func:`baseline_encoding` and :func:`anomaly_encoding` are
    plain dicts that work with a bare ``ds.to_zarr(store, encoding=...)``. You
    lose only the georeferencing that makes the store open in GDAL and QGIS.
    """
    try:
        from xr_utils import set_geozarr_attrs
    except ImportError:
        raise ImportError(
            "write_zarr needs xr_utils for the GeoZarr attributes, which is an "
            "optional dependency: pip install 'seasonal-anomaly-meter[geo]'. To "
            "write without it, pass this module's encoding dict straight to "
            "ds.to_zarr(store, encoding=..., consolidated=False) -- the store "
            "will be valid but will not carry its CRS."
        ) from None

    ds = set_geozarr_attrs(ds)
    merged = {name: dict(spec) for name, spec in encoding.items()}
    for name, variable in ds.data_vars.items():
        grid_mapping = variable.encoding.get("grid_mapping")
        if grid_mapping and name in merged:
            merged[name]["grid_mapping"] = grid_mapping
    ds = _align_to_encoding(ds, merged)
    ds.to_zarr(store, mode=mode, encoding=merged, consolidated=False)


def _align_to_encoding(ds: xr.Dataset, encoding: dict) -> xr.Dataset:
    """Rechunk dask-backed variables onto the zarr chunk grid they will be written to.

    The encodings here fix the zarr chunk shape, while a lazy Dataset's dask
    chunks come from whatever the *input* stores were written with -- and those
    two only line up by luck. When they do not, ``to_zarr`` refuses the write
    outright ("would overlap multiple Dask chunks"), because two dask chunks
    landing in one zarr chunk means two parallel tasks writing the same file.

    The refusal arrives at the end of the run, after every earlier stage has
    been paid for, so it is worth pre-empting. Each misaligned dimension is
    rechunked to the nearest whole multiple of its zarr chunk, which keeps the
    task size the graph already chose rather than forcing it down to one task
    per zarr chunk. Dimensions that already align are left untouched, so a
    Dataset written on a matching grid -- the normal case -- is not rechunked
    at all.
    """
    ds = ds.copy()
    for name, spec in encoding.items():
        zarr_chunks = spec.get("chunks")
        variable = ds.get(name)
        if not zarr_chunks or variable is None or variable.chunks is None:
            continue
        rechunk = {}
        for dim, size, dask_chunks in zip(variable.dims, zarr_chunks, variable.chunks):
            # Only interior boundaries matter: a short final chunk is a partial
            # zarr chunk, which is fine, and every other boundary has to fall on
            # a multiple of the zarr chunk size.
            edges = np.cumsum(dask_chunks[:-1])
            if not np.any(edges % size):
                continue
            rechunk[dim] = size * max(1, round(max(dask_chunks) / size))
        if rechunk:
            ds[name] = variable.chunk(rechunk)
    return ds
