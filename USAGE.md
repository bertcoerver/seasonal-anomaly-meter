# Using seasonal-anomaly-meter

## What it calculates

For each pixel, the package adds up a flux from the start of that pixel's own growing season to a given dekad. It then compares that total with what the same pixel usually accumulates by the **same day of the season** in a reference period (the baseline, e.g. 2018–2025). Because the comparison is by day of season rather than calendar date, a season that starts late isn't flagged as a deficit just for being late.

It uses two products:

- **Flux:** WaPOR L1 dekadal data, either **AETI** (actual evapotranspiration, mm/day) or **NPP** (net primary production, gC/m2/day).
- **Phenology:** **Copernicus Land Surface Phenology** (CLMS LSP, 300 m, yearly). It gives the start and end of up to two growing seasons per year for each pixel.

The results are the accumulated flux, its baseline value, and three anomalies: absolute, relative (%) and standardised (z-score).

The package takes `xarray` objects and returns `xarray` objects. It does no reading or writing of its own. In the examples, `load_flux`, `load_phenology`, `load_baseline` and `save` are placeholders for your own I/O.

```python
from seasonal_anomaly_meter import (
    required_flux_start, seasonal_anomalies, seasonal_baseline, value_encoding,
)
```

There are two stages:

| Stage | Runs | Reads |
|---|---|---|
| 1. Baseline (`seasonal_baseline`) | rarely (e.g. once a year) | flux and phenology over the baseline years |
| 2. Anomalies (`seasonal_anomalies`) | every new dekad | the current season's flux, recent phenology and the stored baseline |

---

## Example data

The dashboard at <https://storage.googleapis.com/fao-wapor-accounter/wapor-anomalies/main.html> shows this package's output for NPP and AETI over WaPOR tile 36Q (Gezira scheme, Sudan). It uses four variables of the Stage 2 output, described [below](#stage-2-required_flux_start--seasonal_anomalies):

| Dashboard element | Variable(s) |
|---|---|
| Map | `anomaly_abs`, one layer per dekad |
| Graph, clicked pixel | `acc` (current season) and `acc_baseline` (long-term average), plotted against `DOS` |
| Masking of out-of-season pixels | `DOS == 0` |

The other outputs (`anomaly_rel`, `anomaly_z`, `season`) aren't used by the dashboard, so they don't need to be stored for now. The baseline itself only has to be stored so Stage 2 can read it; it isn't shown directly.

---

## Inputs

The flux and the phenology must be on **the same grid**: identical `y`/`x` coordinates, in the same CRS. This is checked, and a mismatch raises. Work on an area that fits in one CRS (for WaPOR, one UTM tile). Both inputs can be lazy (dask-backed).

### Flux

A `DataArray` of dekadal WaPOR data:

```
<xarray.DataArray 'NPP' (time, y, x)>  float32
Coordinates:
  * time  datetime64[ns]   one step per dekad, sorted
  * y     float64          projected coordinates
  * x     float64
Attributes:
    units:  gC/m2/day
```

- **The values must be a rate per day** (mm/day, gC/m2/day), as WaPOR publishes them, not a total per dekad. Each dekad is multiplied by its true length in days (8–11), so dekad totals would come out about 10× too high.
- **Timestamps:** any date inside a dekad identifies it, whether that's its first day or its middle.
- **NaN** means no data for that pixel and dekad.

### Phenology

A `Dataset` of Copernicus season starts and ends:

```
<xarray.Dataset>  dims (season: 2, year: N, y, x)
Coordinates:
  * season  int      season labels [1, 2]
  * year    int      e.g. [2018, ..., 2025]
  * y, x             same as the flux
Data variables:
    SOSD    float32  start of season, day-of-year within `year`
    EOSD    float32  end of season, day-of-year within `year`
    QA      uint8    quality flag; lower is better
```

- `SOSD`/`EOSD` are 1-based day-of-year numbers relative to 1 January of `year`, not dates. They can be negative or above 365 when a season crosses a year boundary (−61 is late October of the previous year). NaN means no season.
- When a pixel's two seasons overlap, the one with the better `QA` is used; if `QA` is equal, the longer one.

---

## Stage 1: `seasonal_baseline`

```python
flux = load_flux("2018-01-01", "2025-12-31")
phenology = load_phenology(years=range(2018, 2026))

baseline = seasonal_baseline(flux, phenology)      # lazy; computed when saved
save(baseline, encoding=value_encoding(baseline))
```

The baseline years are simply the years covered by the flux you pass in. Give the phenology at least those years.

For each pixel, season and **day-of-season slot**, the result is the mean, standard deviation and count of the flux accumulated since season start, across the baseline years. The slot is the `pos` dimension: 1 is the dekad the season starts in, and there are up to 49 dekads.

```
<xarray.Dataset>  dims (season: 2, pos: 49, y, x)
Coordinates:
  * season  int      Copernicus season label (1, 2)
  * pos     int16    1..49, dekads since season start
  * y, x             as the flux
Data variables:
    acc_mean   float32  mean accumulated flux          (units: gC/m2)
    acc_std    float32  std. dev. across years
    acc_count  uint8    years contributing to the slot
Attributes (informational; Stage 2 doesn't need them):
    baseline_years       "2018-2025"
    temporal_resolution  "dekadal"
    min_years            3
    units, rate_units, variable
```

**The `season` dimension.** Copernicus reports up to two growing seasons per pixel per year: season 1 and season 2, as in regions with two rainy seasons or double cropping. The two build up differently (a short second season doesn't follow the curve of a long main one), so each gets its own baseline curve. Stage 2 compares a pixel against the curve of whichever season it is currently in. Where a pixel only ever has one season, the season-2 values are NaN.

**`min_years`** (default 3). The baseline years are fixed, but a given slot at a given pixel isn't reached in every one of them:
- a pixel may have had no detected season in some years;
- a late slot (e.g. dekad 40 of a season) is only reached in years with a long season.

A slot reached in fewer than `min_years` years is set to NaN rather than stored as a "normal" based on one or two seasons. A standard deviation from one year is undefined anyway. Set `min_years=1` to keep every slot, and use `acc_count` to judge reliability instead.

Other options: `chunks=256` sets the spatial chunk size. The `time` axis is always kept in one chunk.

---

## Stage 2: `required_flux_start` + `seasonal_anomalies`

```python
dates = ["2026-09-01"]                        # the new dekad(s) to compute
baseline = load_baseline()
phenology = load_phenology(years=[2025, 2026])

start = required_flux_start(phenology, dates)   # -> numpy.datetime64
flux = load_flux(start, max(dates))

anomalies = seasonal_anomalies(flux, phenology, baseline, dates)
save(anomalies, encoding=value_encoding(anomalies))       # or append along time
```

**What phenology Stage 2 needs:**
- **Years:** the query year and the year before it. A season labelled with the previous year can still be running on the query date. Older years, including the baseline years, aren't used.
- **Unpublished years:** Copernicus phenology is published a year or more after the fact. Years not available yet are filled by repeating the latest year supplied (`forward_fill=True`, the default), and flagged with a `forward_filled` coordinate.

**How much flux:** only the current season's flux is needed, not the archive. `required_flux_start` works out how much. It reads the phenology (not the flux) and returns the start of the earliest season still running on any of `dates`. Fetch flux from that date to the last query date. For one new dekad that is typically a few months.
- It is capped at 49 dekads back, the longest season the baseline stores.
- Call it with the same `dates` you then pass to `seasonal_anomalies`.

The result has one time step per query date:

```
<xarray.Dataset>  dims (time: len(dates), y, x)
Coordinates:
  * time  datetime64[ns]   the query dates
  * y, x                   as the flux
Data variables:
    DOS           uint16   days since start of season; 0 = not in season
    season        uint8    season label in effect (1, 2); 0 = none
    acc           float32  flux accumulated since season start    (gC/m2)
    acc_baseline  float32  baseline at the same day of season     (gC/m2)
    anomaly_abs   float32  acc - acc_baseline; signed             (gC/m2)
    anomaly_rel   float32  100 * anomaly_abs / acc_baseline       (%)
    anomaly_z     float32  anomaly_abs / baseline std. dev.       (sigma)
Attributes:
    variable, baseline_years
    units   the input flux's units; each variable carries its own `units`
```

- Outside a season, the float fields are NaN and `DOS`/`season` are 0.
- `anomaly_rel` and `anomaly_z` are also NaN where the baseline accumulation is below 1 unit, where a ratio isn't meaningful.

**Running it operationally:** each date is computed independently. Adding a date later gives exactly the same result as computing it in an earlier batch. So compute only the new dekad(s) and **append** them along `time`; never recompute the whole store. Rebuild the anomalies from scratch only when the baseline changes.

---

## Encoding

`value_encoding(ds)` returns the recommended storage type for each variable as a plain dict of CF keys (`dtype`, `scale_factor`, `_FillValue`). `to_zarr` and `to_netcdf` accept it directly; other writers can read the settings from it.

The example scripts use **`scale_factor=1.0` for both outputs**, i.e. `value_encoding(baseline, scale_factor=1.0)` and `value_encoding(anomalies, scale_factor=1.0)`. That gives:

| Output | Variable | `dtype` | `scale_factor` | `_FillValue` | Stored resolution / range |
|---|---|---|---|---|---|
| baseline | `acc_mean` | int16 | 1.0 | −32768 | 1 gC/m2 (NPP) or 1 mm (AETI); ±32767 |
| baseline | `acc_std` | int16 | 1.0 | −32768 | as above |
| baseline | `acc_count` | uint8 | — | — | 0–255 years |
| anomalies | `acc` | int16 | 1.0 | −32768 | 1 gC/m2 or 1 mm; ±32767 |
| anomalies | `acc_baseline` | int16 | 1.0 | −32768 | as above |
| anomalies | `anomaly_abs` | int16 | 1.0 | −32768 | as above; signed |
| anomalies | `anomaly_rel` | float32 | — | NaN | full precision (%) |
| anomalies | `anomaly_z` | float32 | — | NaN | full precision (sigma) |
| anomalies | `DOS` | uint16 | — | none | 0–65535 days |
| anomalies | `season` | uint8 | — | none | 0, 1, 2 |

- **What `scale_factor` does:** it applies only to the variables in the flux's own units (`acc_mean`, `acc_std`, `acc`, `acc_baseline`, `anomaly_abs`). They are stored as int16, which rounds each value to the nearest multiple of `scale_factor` and limits the range to ±32767 × `scale_factor`. `anomaly_rel` and `anomaly_z` are never packed. With `scale_factor=None`, the flux-unit variables are stored as float32 instead.
- **Without a `scale_factor` argument,** the defaults differ from the scripts: int16 at 0.1 for the baseline, and float32 for the anomalies.
- **Check the range before packing.** `check_packing_range(baseline, scale_factor)` raises if the data doesn't fit the int16 range; it computes the data. At 1.0 the range is ±32767, far above a season's accumulated NPP or AETI.
- **Never store anomalies as an unsigned type.** Deficits are negative.
- **`DOS` and `season` have no fill value on purpose.** 0 is a real value (the not-in-season mask). A fill value would turn it into NaN when the data is read back.

**Recommended chunking**, if your format supports it:

- **Baseline:** keep `season` and `pos` whole in each chunk and tile only `y`/`x` (e.g. 256×256). Stage 2 reads a pixel's whole season curve at once.
- **Anomalies:** one time step per chunk (`time: 1`), tiled in `y`/`x`. Appending a dekad then never rewrites existing chunks.
