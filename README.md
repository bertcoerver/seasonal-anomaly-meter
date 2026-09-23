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

## USAGE

**To use the package, read [USAGE.md](USAGE.md).** It describes the inputs,
the two stages (`seasonal_baseline`, `required_flux_start` +
`seasonal_anomalies`), the output datasets, recommended storage encodings, and
a live dashboard built from the output.

## Design

**Two stages, deliberately separated.** `seasonal_baseline` reads years of flux
and runs rarely; `seasonal_anomalies` reads only the current season and the
stored baseline, and runs whenever a new dekad lands. Stage 2 never touches the
historical archive, which is what makes an operational rerun cheap. Each date is
computed independently, so keeping the anomalies current means appending the
new dates, not recomputing the store.

**The flux grid is the example grid** — it is never resampled. `align_phenology`
moves the phenology onto it with nearest-neighbour resampling, since start- and
end-of-season are day-of-year codes and interpolating them would invent season
boundaries that exist in no source pixel. That reverses the original
methodology, which warped every period of flux onto the phenology grid: far more
data movement, and it resampled the measurement rather than the annotation.
Work on an area that stays in one CRS; for WaPOR's UTM mosaicsets that is one
tile.

**The baseline stores accumulations, not rates.** It holds statistics of the
flux accumulated since season start at each day-of-season slot. That is what
makes the standard deviation meaningful: the spread of a seasonal total cannot
be recovered from the spreads of its parts without their covariance, and within
a growing season that covariance is large. It also means Stage 2 does two
lookups instead of re-integrating a season.

There is deliberately **no stored mean-rate array**. Interpolating between two
consecutive cumulative slots is arithmetically identical to adding
`rate * days_into_slot`, so such an array would be redundant — and at ~1.3 GB
per tile, expensively so. `tests/test_accumulate.py` pins that equivalence.

## Scope and limits

- Built for **dekadal** data (WaPOR L1).
- With WaPOR, **baseline years start in 2018**, when it begins. Copernicus
  phenology reaches back to 2014 but a baseline needs both.
- **Phenology lags the flux** by a year or more, so the current season's start
  and end are assumed to repeat the most recent year on record. Assumed years
  are flagged with a `forward_filled` coordinate; pass `forward_fill=False` to
  refuse rather than assume.
- Mosaicking tiles and exporting global EPSG:4326 COGs is **not** done.

## Installation

The package itself needs nothing but PyPI, and only what it computes with —
xarray, dask, numpy:

```bash
pip install -e .
```

Storing the result is an extra, and so are the unpublished siblings:

| Extra | Adds | Needed for |
|---|---|---|
| `zarr` | `zarr>=3` | storing a result as Zarr — `write_geozarr` or your own `to_zarr` |
| `geo` | `zarr` + [`xr_utils`][xr_utils] | `align_phenology`, `write_geozarr`, `open_geozarr` |
| `examples` | `geo` + [`lazy_dino`][lazy_dino] | `examples/` |

The encoding helpers (`value_encoding`, `baseline_encoding`, `anomaly_encoding`)
need none of these: they return plain dicts.

```bash
pip install -e ".[dev]"
pytest
```

Reading the Copernicus phenology in the examples needs CDSE credentials; store
them once with `lazy_dino`'s `login()`.
