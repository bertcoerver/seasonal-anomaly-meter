"""The package's input contract, and the checks that enforce it.

Everything downstream works on two caller-supplied arrays:

``flux``
    ``(time, y, x)``, a **per-day rate** -- mm/day, gC/m2/day -- on whatever
    grid you want the answer on. This is the example grid; it is never moved.

``phenology``
    ``(season, year, y, x)`` with ``SOSD`` and ``EOSD`` as day-of-year numbers
    and an optional ``QA``, already on the flux's grid.

Where those come from is the caller's business: WaPOR and Copernicus are what
the methodology was built for, but nothing here knows that. See
``examples/wapor_sources.py`` for one way to produce them.

The checks below exist because this boundary used to be guaranteed by the
loader and now is not. Two of the three things that can go wrong here are
silent -- a mismatched grid pairs each pixel with a stranger's season, and a
flux that is already a per-period total inflates every accumulation by the
period length -- so both are checked or warned about rather than left to
surface as implausible numbers weeks later.
"""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

__all__ = [
    "PHENOLOGY_VARS",
    "align_phenology",
    "as_flux",
    "check_phenology",
    "check_same_grid",
]

#: The phenology variables this package reads. ``QA`` is optional: supply it and
#: overlapping seasons are resolved by quality, omit it and they are resolved by
#: duration alone.
PHENOLOGY_VARS = ("SOSD", "EOSD")

#: Widest day-of-year a Copernicus season bound can plausibly take. Seasons
#: routinely run outside their nominal year -- a SOSD of -61 means late October
#: of the year before -- but a value beyond this range means the field holds
#: something other than a day-of-year, most likely a date or a scaled integer.
_DOY_LIMITS = (-250.0, 620.0)


def as_flux(flux: xr.DataArray | xr.Dataset, variable: str | None = None) -> xr.DataArray:
    """Normalise the flux input to a checked ``(time, y, x)`` DataArray.

    Accepts a DataArray, or a Dataset holding exactly one data variable (or one
    named by ``variable``), so a caller can hand over whatever their loader
    returned.

    **The values must be a rate per day.** Accumulation multiplies each period
    by its true length in days, which is the only way to handle a dekad D3 that
    runs 8 to 11 days. Passing per-period totals instead runs cleanly and
    overstates every accumulation about tenfold, so a ``units`` attribute that
    does not look like a daily rate is logged as a warning here.
    """
    if isinstance(flux, xr.Dataset):
        names = list(flux.data_vars)
        if variable is not None:
            if variable not in flux:
                raise KeyError(
                    f"{variable!r} is not in the flux dataset; it holds {names}."
                )
            flux = flux[variable]
        elif len(names) == 1:
            flux = flux[names[0]]
        else:
            raise ValueError(
                f"the flux dataset holds {len(names)} variables {names}; pass "
                "variable= to say which one is the flux."
            )

    missing = [d for d in ("time", "y", "x") if d not in flux.dims]
    if missing:
        raise ValueError(
            f"the flux is missing the {missing} dimension(s); it has "
            f"{list(flux.dims)}. Rename your dimensions to time/y/x -- this "
            "package indexes them by name."
        )
    if not np.issubdtype(flux["time"].dtype, np.datetime64):
        raise TypeError(
            f"the flux time coordinate is {flux['time'].dtype}, not datetime64. "
            "Period indexing goes through real calendar dates, so integer or "
            "string time labels cannot be placed on the season axis."
        )

    units = str(flux.attrs.get("units", ""))
    if units and not any(tag in units for tag in ("/day", "/d", "day-1", "d-1")):
        logger.warning(
            "flux units are %r, which does not look like a per-day rate. "
            "Accumulation weights each period by its length in days, so "
            "per-period totals would be overstated by roughly that factor.",
            units,
        )
    return flux


def check_phenology(phenology: xr.Dataset, flux: xr.DataArray | None = None) -> xr.Dataset:
    """Verify the phenology input, optionally against the flux grid.

    Checks the variable names, the ``season``/``year`` dimensions, that
    ``SOSD``/``EOSD`` hold day-of-year numbers rather than dates, and -- when
    ``flux`` is given -- that the two sit on the same pixels.

    Returns ``phenology`` unchanged; it raises or it passes.
    """
    missing = [v for v in PHENOLOGY_VARS if v not in phenology.data_vars]
    if missing:
        raise ValueError(
            f"the phenology is missing {missing}; it holds "
            f"{list(phenology.data_vars)}. Expected {list(PHENOLOGY_VARS)} "
            "(day-of-year) and optionally QA."
        )
    for dim in ("season", "year", "y", "x"):
        if dim not in phenology.dims:
            raise ValueError(
                f"the phenology is missing the {dim!r} dimension; it has "
                f"{list(phenology.dims)}. Even a single season needs the axis, "
                "so expand it with .expand_dims(season=[1])."
            )

    for name in PHENOLOGY_VARS:
        var = phenology[name]
        if np.issubdtype(var.dtype, np.datetime64):
            raise TypeError(
                f"{name} is a datetime64 array, but SOSD/EOSD must be "
                "day-of-year numbers -- possibly negative, or past 366, when a "
                "season crosses a year boundary. Convert dates back to a "
                "day-of-year offset from 1 January of that season's year."
            )
        # Reading only the corners of the array keeps this cheap on a lazy
        # input: a whole-array reduction would pull every chunk through just to
        # sanity check the units.
        sample = var.isel(
            {d: slice(0, 4) for d in var.dims if d in ("y", "x")}
        ).values
        finite = sample[np.isfinite(sample)]
        low, high = _DOY_LIMITS
        if finite.size and (finite.min() < low or finite.max() > high):
            raise ValueError(
                f"{name} spans {finite.min():.0f} to {finite.max():.0f}, outside "
                f"the {low:.0f}..{high:.0f} a day-of-year can plausibly take. "
                "Check that this field is not a date, a period index or a "
                "scaled integer."
            )

    if flux is not None:
        check_same_grid(flux, phenology, "flux", "phenology")
    return phenology


def check_same_grid(a, b, a_name: str = "flux", b_name: str = "phenology") -> None:
    """Raise unless ``a`` and ``b`` sit on exactly the same ``y``/``x`` pixels.

    The original implementation asserted only that the sizes matched and then
    overwrote the coordinates, which silently accepted a genuinely different
    grid as long as it had the same shape -- pairing every pixel with some other
    pixel's season. Compare the coordinates themselves.
    """
    for axis in ("y", "x"):
        if axis not in a.coords or axis not in b.coords:
            raise ValueError(
                f"{a_name} and {b_name} cannot be compared: one of them has no "
                f"{axis!r} coordinate."
            )
        left, right = a[axis].values, b[axis].values
        if left.shape != right.shape or not np.allclose(left, right):
            raise ValueError(
                f"{a_name} and {b_name} disagree on the {axis} axis "
                f"({left.shape} vs {right.shape}). They must share a grid: warp "
                f"the {b_name} onto the {a_name} with "
                "seasonal_anomaly_meter.align_phenology(phenology, flux)."
            )


def align_phenology(
    phenology: xr.Dataset,
    flux: xr.DataArray | xr.Dataset,
    *,
    resampling: str = "nearest",
) -> xr.Dataset:
    """Warp ``phenology`` onto ``flux``'s grid, leaving the flux untouched.

    The flux is the example grid: it keeps the pixel values its publisher
    produced, and only the phenology moves. That is the reverse of the original
    methodology, which interpolated every period of flux onto the phenology
    grid -- far more data movement, and it resampled the measurement rather than
    the annotation.

    **Nearest neighbour is not a shortcut here, it is the only correct choice.**
    SOSD/EOSD are day-of-year codes and QA is a class flag; averaging them would
    produce season boundaries and quality classes that exist in no source pixel,
    and would smear the NaN marking "no season here" across its neighbours.

    Needs ``xr_utils`` and a CRS on both inputs. If your phenology is already on
    the flux grid -- which it will be if you resampled it yourself -- skip this.
    """
    try:
        from xr_utils import reproject_like
    except ImportError:
        raise ImportError(
            "align_phenology needs xr_utils, which is an optional dependency: "
            "pip install 'seasonal-anomaly-meter[geo]'. You only need it to move "
            "the phenology onto the flux grid -- if the two already share "
            "y/x coordinates, pass them straight to seasonal_baseline."
        ) from None

    flux = flux if isinstance(flux, xr.DataArray) else as_flux(flux)
    reference = flux.isel({d: 0 for d in flux.dims if d not in ("y", "x")})

    warped = {}
    for name, da in phenology.data_vars.items():
        out = reproject_like(da, reference, resampling=resampling, dst_nodata=np.nan)
        if name == "QA":
            # QA's own nodata is 255; restore it, and the integer dtype, now
            # that the warp -- which needs a NaN-capable float -- is done.
            out = out.fillna(255).astype(np.uint8)
        warped[name] = out
    return xr.Dataset(warped, attrs=phenology.attrs)
