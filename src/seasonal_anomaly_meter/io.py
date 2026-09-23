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

This module builds encodings and nothing else -- it knows this package's
variable names and dimension order, which is the part no general-purpose
library can supply. The writing and reading is generic and lives in
``xr_utils.geozarr``: pass one of these dicts to ``write_geozarr`` to get a
store GDAL and QGIS open georeferenced, and read it back with ``open_geozarr``
so its CRS survives as a coordinate. Both are optional -- the dicts here are
plain ``to_zarr`` encodings and work with a bare
``ds.to_zarr(store, encoding=..., consolidated=False)``; you lose only the
georeferencing.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

__all__ = [
    "PACKED_FILL",
    "baseline_encoding",
    "anomaly_encoding",
    "value_encoding",
    "check_packing_range",
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


#: The anomaly variables carried in the flux's own units, and so the only ones
#: a flux scale factor means anything for. ``anomaly_rel`` is a percentage and
#: ``anomaly_z`` a standard-deviation count: both have their own natural
#: resolution, and packing them at the flux's would be a category error -- at
#: ``scale_factor=1`` a z-score would be rounded to whole sigma.
_PACKABLE_ANOMALIES = ("acc", "acc_baseline", "anomaly_abs")


def anomaly_encoding(
    ds: xr.Dataset,
    *,
    scale_factor: float | None = None,
    chunks: tuple[int, int] = (256, 256),
) -> dict:
    """Zarr encoding for :func:`~seasonal_anomaly_meter.anomaly.compute_anomaly`.

    One date per chunk along ``time`` so an operational rerun appends without
    rewriting earlier dates.

    By default every anomaly field stays ``float32``: they are signed, they are
    the product people read, and at one date per file the space saved by packing
    is not worth the extra failure mode. Pass ``scale_factor`` -- the same one
    the baseline was packed at -- to halve the store anyway, which is worth
    doing once the run covers years of dates rather than a handful. Only
    :data:`_PACKABLE_ANOMALIES` are packed, since only they are in the flux's
    units; check the range first with :func:`check_packing_range`.
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
        elif scale_factor is not None and name in _PACKABLE_ANOMALIES:
            spec.update(dtype="int16", scale_factor=scale_factor, _FillValue=PACKED_FILL)
        else:
            spec["dtype"] = "float32"
        encoding[name] = spec
    return encoding


#: Encoding keys that describe a zarr store's layout rather than the values.
_LAYOUT_KEYS = ("chunks", "compressors")


def value_encoding(ds: xr.Dataset, **kwargs) -> dict:
    """The storage dtype, ``scale_factor`` and ``_FillValue`` of each variable.

    :func:`baseline_encoding` or :func:`anomaly_encoding` -- picked by whether
    ``ds`` is a baseline or an anomaly result -- without their zarr chunk and
    compressor settings. What is left is plain CF packing, which ``to_zarr``,
    ``to_netcdf`` and most other writers accept as is; choose chunking and
    compression for your own format. ``kwargs`` (``scale_factor``) go to the
    underlying function, so the defaults are the same.
    """
    build = baseline_encoding if "acc_mean" in ds else anomaly_encoding
    return {
        name: {k: v for k, v in spec.items() if k not in _LAYOUT_KEYS}
        for name, spec in build(ds, **kwargs).items()
    }


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
