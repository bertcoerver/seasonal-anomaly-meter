"""End-to-end run for one WaPOR UTM tile, using NPP.

Fetches the two inputs from WaPOR and Copernicus (``wapor_sources``, the only
code in this repository that uses ``lazy_dino``), builds the multi-year baseline
for a tile, then computes anomalies for a few query dates against it, writing
both stores as zarr in the tile's own UTM CRS.

The package itself takes the flux and the phenology as arrays and does no I/O,
so swapping in a different source means replacing ``load_flux``/
``load_phenology`` below and nothing else.

A whole tile is ~3000 x 2400 pixels over 8 years of dekads, so the default here
is a small window -- enough to see real numbers in a minute or two. Pass
``--full`` for the whole tile, and expect it to take a while.

Run with::

    python examples/npp_single_tile.py                 # small window
    python examples/npp_single_tile.py --tile 31U      # a different tile
    python examples/npp_single_tile.py --full          # the entire tile
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import date
from pathlib import Path

import numpy as np
from wapor_sources import load_flux, load_phenology
from wapor_tiles import get_tile

from seasonal_anomaly_meter import (
    anomaly_encoding,
    baseline_encoding,
    check_packing_range,
    seasonal_anomalies,
    seasonal_baseline,
    write_zarr,
)

VARIABLE = "L1-UTM-NPP-D"
#: 36Q covers the Gezira scheme in Sudan -- irrigated, strongly seasonal, and
#: the area the original drought-depth work was tested on.
DEFAULT_TILE = "36Q"
DEFAULT_DATES = ["2024-03-01", "2024-06-01", "2024-09-01"]
SCALE_FACTOR = 0.1

#: WaPOR starts in 2018, so that is the earliest a baseline year can be. The
#: Copernicus phenology reaches back to 2014, but a baseline needs both.
WAPOR_START_YEAR = 2018


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile", default=DEFAULT_TILE)
    parser.add_argument("--variable", default=VARIABLE)
    parser.add_argument("--dates", nargs="+", default=DEFAULT_DATES)
    parser.add_argument(
        "--full", action="store_true", help="process the whole tile, not a window"
    )
    parser.add_argument(
        "--window", type=int, default=256, help="window size in pixels (ignored with --full)"
    )
    parser.add_argument("--centre-y", type=int, default=None, help="window centre row")
    parser.add_argument("--centre-x", type=int, default=None, help="window centre column")
    parser.add_argument("--out", type=Path, default=Path("./output"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("rasterio").setLevel(logging.ERROR)

    tile = get_tile(args.variable, args.tile)
    print(f"tile {tile.code}  EPSG:{tile.epsg}  {tile.shape[0]}x{tile.shape[1]} px")

    # One window, shared by both stages -- they must sit on the same grid, and
    # the package checks that they do. The flux is cropped *before* the
    # phenology is warped onto it, so a windowed run warps only the window.
    window = None
    if not args.full:
        half = args.window // 2
        cy, cx = args.centre_y or tile.shape[0] // 2, args.centre_x or tile.shape[1] // 2
        window = {"y": slice(cy - half, cy + half), "x": slice(cx - half, cx + half)}

    args.out.mkdir(parents=True, exist_ok=True)
    suffix = "full" if args.full else f"w{args.window}"
    # The current year is excluded from the baseline: a season still in
    # progress would contribute a truncated accumulation and drag every later
    # slot down.
    baseline_years = range(WAPOR_START_YEAR, date.today().year)

    # ---- Stage 1: the baseline (slow, rebuilt rarely) --------------------
    print("\nbuilding baseline ...")
    started = time.perf_counter()
    flux, phenology = _inputs(
        args.variable,
        tile,
        window,
        years=baseline_years,
        time_range=(f"{min(baseline_years)}-01-01", f"{max(baseline_years)}-12-31"),
    )
    baseline = seasonal_baseline(flux, phenology, variable=args.variable).compute()
    print(f"  computed in {time.perf_counter() - started:.0f}s")
    _report_baseline(baseline)

    check_packing_range(baseline, SCALE_FACTOR)
    baseline_store = args.out / f"{args.variable}_{tile.code}_{suffix}_baseline.zarr"
    write_zarr(baseline, baseline_store, baseline_encoding(baseline, scale_factor=SCALE_FACTOR))
    print(f"  wrote {baseline_store} ({_du(baseline_store):.1f} MB)")

    # ---- Stage 2: the anomalies (fast, rerun operationally) --------------
    print("\ncomputing anomalies ...")
    started = time.perf_counter()
    query_years = sorted({int(str(d)[:4]) for d in args.dates})
    flux, phenology = _inputs(
        args.variable,
        tile,
        window,
        # Phenology from the baseline anchor year, so season_indices places
        # every season on the same period axis the stored baseline used.
        years=range(WAPOR_START_YEAR, max(query_years) + 1),
        # A Copernicus season can run ~500 days, so reach back far enough to
        # contain the start of every season active on the query dates.
        time_range=(f"{min(query_years) - 2}-01-01", max(args.dates)),
    )
    anomalies = seasonal_anomalies(flux, phenology, baseline, args.dates).compute()
    print(f"  computed in {time.perf_counter() - started:.0f}s")
    _report_anomalies(anomalies)

    anomaly_store = args.out / f"{args.variable}_{tile.code}_{suffix}_anomaly.zarr"
    write_zarr(anomalies, anomaly_store, anomaly_encoding(anomalies))
    print(f"  wrote {anomaly_store} ({_du(anomaly_store):.1f} MB)")


def _inputs(variable, tile, window, *, years, time_range):
    """The two arrays the package wants: a flux rate, and phenology on its grid."""
    flux = load_flux(variable, tile, time_range)
    if window is not None:
        flux = flux.isel(window)
    phenology = load_phenology(tile, years, example=flux)
    return flux, phenology


def _report_baseline(baseline) -> None:
    mean = baseline["acc_mean"]
    seasonal = np.isfinite(mean.isel(pos=0)).sum().item()
    total = mean.isel(pos=0).size
    print(
        f"  pixels with a season: {seasonal}/{total} "
        f"({100 * seasonal / max(total, 1):.1f}%) -- the rest is bare ground/water"
    )
    if seasonal:
        peak = float(np.nanmax(mean.values))
        print(f"  peak accumulated NPP over a season: {peak:.0f} {baseline.attrs.get('units','')}")
        print(f"  slots with >= min_years of data:   {int((baseline['acc_count'] > 0).sum())}")


def _report_anomalies(anomalies) -> None:
    for i, date_value in enumerate(anomalies["time"].values):
        in_season = int((anomalies["DOS"].isel(time=i) > 0).sum())
        values = anomalies["anomaly_abs"].isel(time=i).values
        finite = values[np.isfinite(values)]
        label = str(date_value)[:10]
        if finite.size:
            negative = 100 * (finite < 0).mean()
            print(
                f"  {label}: {in_season:6d} px in season, "
                f"anomaly median {np.median(finite):+.0f}, {negative:.0f}% below baseline"
            )
        else:
            print(f"  {label}: {in_season:6d} px in season, no finite anomalies")


def _du(path: Path) -> float:
    """Store size in MB."""
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


if __name__ == "__main__":
    main()
