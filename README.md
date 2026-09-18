# seasonal-anomaly-meter

Seasonal, day-of-season-aligned anomalies for gridded fluxes.

For every pixel this integrates a flux from the start of that pixel's own growing
season — from a phenology product such as Copernicus LSP — up to a query date,
and compares it with what the same pixel usually accumulates by the same **day of
season**. The baseline day can fall on a quite different calendar date from year
to year, which is the whole reason for pairing phenology with the flux.

**The package does no I/O.** You hand it two xarray objects and it hands back the
anomalies, so it works with WaPOR and Copernicus or with anything else shaped
like them. Fetching those two inputs from WaPOR and Copernicus via
[`lazy_dino`][lazy_dino] lives in [`examples/`](examples/), outside the package.

[lazy_dino]: https://github.com/bertcoerver/lazy_dino
[xr_utils]: https://github.com/bertcoerver/xr-utils

## The two inputs

| | dims | contents |
|---|---|---|
| `flux` | `(time, y, x)` | a **per-day rate** — mm/day, gC/m²/day |
| `phenology` | `(season, year, y, x)` | `SOSD`, `EOSD` as day-of-year, optional `QA` |

Both must sit on the same `y`/`x` pixels, which is checked. Two things about
this contract are worth stating plainly, because getting either wrong runs
cleanly and gives wrong numbers:

- **The flux is a rate, not a per-period total.** Accumulation weights each
  period by its true length in days — a dekad's third period runs 8 to 11 — so
  handing over totals inflates everything roughly tenfold. A `units` attribute
  that does not look like a daily rate is warned about.
- **`SOSD`/`EOSD` are day-of-year numbers, not dates,** and legitimately fall
  outside their nominal year: `-61` means late October of the year before, `416`
  means February of the next. That is how a cross-year season is encoded, and it
  is handled rather than clipped.

**The flux grid is the example grid** — it is never resampled. `align_phenology`
moves the phenology onto it with nearest-neighbour resampling, since start- and
end-of-season are day-of-year codes and interpolating them would invent season
boundaries that exist in no source pixel. That reverses the original
methodology, which warped every period of flux onto the phenology grid: far more
data movement, and it resampled the measurement rather than the annotation.

Work on an area that stays in one CRS. For WaPOR's UTM mosaicsets that is one
tile, which is why the example is per-tile.

## Two stages, deliberately separated

| Stage | Cadence | Reads | Writes |
|---|---|---|---|
| `seasonal_baseline` | rarely | ~8 years of flux + phenology | `*_baseline.zarr` |
| `seasonal_anomalies` | every new period | current season + the stored baseline | `*_anomaly.zarr` |

Stage 2 never touches the historical archive, which is what makes an operational
rerun cheap. The baseline stores statistics of the **accumulation** at each
day-of-season slot, not of the per-period rate:

```
acc_mean (season, pos, y, x)   mean across baseline years
acc_std  (season, pos, y, x)   spread across years  -> z-scores
acc_count(season, pos, y, x)   contributing years   -> reliability mask
```

Storing accumulations rather than rates is what makes the standard deviation
meaningful: the spread of a seasonal total cannot be recovered from the spreads
of its parts without their covariance, and within a growing season that
covariance is large. It also means the operational stage does two lookups
instead of re-integrating a season.

There is deliberately **no stored mean-rate array**. Interpolating between two
consecutive cumulative slots is arithmetically identical to adding
`rate * days_into_slot`, so such an array would be redundant — and at ~1.3 GB
per tile, expensively so. `tests/test_accumulate.py` pins that equivalence.

## Usage

```python
from xr_utils import open_geozarr, write_geozarr

from seasonal_anomaly_meter import (
    seasonal_baseline, seasonal_anomalies, align_phenology,
    baseline_encoding, anomaly_encoding,
)

# flux:      (time, y, x) per-day rate, loaded however you like
# phenology: (season, year, y, x) with SOSD/EOSD day-of-year
phenology = align_phenology(phenology, flux)      # skip if grids already match

# Stage 1 -- slow, run rarely.
baseline = seasonal_baseline(flux, phenology).compute()
write_geozarr(baseline, "36P_baseline.zarr", baseline_encoding(baseline))

# Stage 2 -- fast, run whenever new data lands.
baseline = open_geozarr("36P_baseline.zarr")
anomalies = seasonal_anomalies(flux, phenology, baseline, ["2024-09-01"]).compute()
write_geozarr(anomalies, "36P_anomaly.zarr", anomaly_encoding(anomalies))
```

This package builds the *encodings* — it knows the variable names and the
dimension order, which is the part no general-purpose library can supply — and
leaves the writing and reading to [`xr_utils.geozarr`][xr_utils], where
`write_geozarr` merges the CF `grid_mapping` pointer into the encoding (so the
store opens georeferenced in QGIS) and rechunks onto the zarr chunk grid, and
`open_geozarr` brings the CRS back as a coordinate. Neither is required: the
encodings are plain dicts, so `ds.to_zarr(store, encoding=..., consolidated=False)`
works on the bare install and costs you only the georeferencing.

The temporal resolution is read off the time axis; pass `resolution=` to override.
`year_min` — the anchor both stages index their periods against — is recorded in
the baseline's attributes and reused by `seasonal_anomalies`, so the two stages
cannot drift onto different period axes.

Anomaly outputs, on the flux's own grid:

| Variable | Meaning |
|---|---|
| `DOS` | days since the season started; `0` is the in-season mask |
| `season` | phenology season label in effect; `0` where none |
| `acc` | flux accumulated since the season started |
| `acc_baseline` | what the baseline says by this day of season |
| `anomaly_abs` | `acc - acc_baseline`, **signed** |
| `anomaly_rel` | percentage of baseline |
| `anomaly_z` | standardised by the baseline's spread across years |

Use `open_zarr` rather than `xr.open_zarr` to read these back: plain xarray
leaves `spatial_ref` as an ordinary variable and reports no CRS, though the store
itself is fine and GDAL/QGIS read it correctly.

### Example

```bash
pip install -e ".[examples]"
python examples/npp_single_tile.py --tile 36P --centre-y 693 --centre-x 1000 --window 96
```

runs both stages over the Gezira irrigation scheme in Sudan with WaPOR NPP and
Copernicus phenology. Drop the window arguments and pass `--full` for the whole
tile; pick a tile with `wapor_tiles.list_tiles("L1-UTM-NPP-D")` (527 of them).

`examples/sources.py` and `examples/wapor_tiles.py` are the only code in
this repository that uses `lazy_dino`. Swapping in a different data source means
replacing those two files and nothing else.

## Scope and limits

- **Dekadal and monthly** cadences. Daily and annual series cannot carry a
  day-of-season methodology and are rejected.
- With WaPOR, **baseline years start in 2018**, when it begins. Copernicus
  phenology reaches back to 2014 but a baseline needs both.
- **Phenology lags the flux** by a year or more, so the current season's start
  and end are assumed to repeat the most recent year on record. Assumed years
  are flagged with a `forward_filled` coordinate; pass `forward_fill=False` to
  refuse rather than assume.
- Mosaicking tiles and exporting global EPSG:4326 COGs is **not** done here yet.

## Installation

The package itself needs nothing but PyPI, and only what it computes with —
xarray, dask, numpy:

```bash
pip install -e .
```

Storing the result is an extra, and so are the unpublished siblings:

| Extra | Adds | Needed for |
|---|---|---|
| `zarr` | `zarr>=3` | storing a result at all — `write_geozarr` or your own `to_zarr` |
| `geo` | `zarr` + [`xr_utils`][xr_utils] | `align_phenology`, `write_geozarr`, `open_geozarr` |
| `examples` | `geo` + [`lazy_dino`][lazy_dino] | `examples/` |

`baseline_encoding` and `anomaly_encoding` need none of these: they return plain
dicts, so you can build the encoding with the bare install and hand it to a
`to_zarr` elsewhere.

```bash
pip install -e ".[dev]"
pytest
```

Reading the Copernicus phenology in the example needs CDSE credentials; store
them once with `lazy_dino`'s `login()`.
