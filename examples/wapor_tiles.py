"""The unit of work: one tile of WaPOR's global UTM grid.

WaPOR republishes its L1/L2 products as "mosaicsets" tiled on a global UTM grid
(``L1-UTM-NPP-D``, ``L2-UTM-AETI-D``, ...), one MGRS grid-zone tile per file.
Each tile lives in its own UTM zone, so a tile is the largest area that can be
processed without either reprojecting the flux or splitting it across CRSs --
which is exactly why it is the unit of work here.

Tiles are enumerated through ``lazy_dino``'s GISMGR discovery backend rather
than being derived arithmetically, so the grid we iterate is the one WaPOR
actually publishes (527 tiles for ``L1-GRID``), with each tile's true affine and
shape. :func:`list_tiles` needs network access; :class:`Tile` itself does not.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from lazy_dino.databases.wapor.wapor import WAPOR_COG_BASE
from lazy_dino.discovery.gismgr import GismgrDiscovery, is_utm_mosaicset

__all__ = ["Tile", "list_tiles", "get_tile"]


@dataclass(frozen=True)
class Tile:
    """One WaPOR UTM grid tile: the example grid for everything downstream."""

    code: str
    """MGRS grid-zone code, e.g. ``"31U"``."""

    epsg: int
    """UTM EPSG of this tile's native grid, e.g. ``32631``."""

    x_range: tuple[float, float]
    """``(xmin, xmax)`` in native UTM metres."""

    y_range: tuple[float, float]
    """``(ymin, ymax)`` in native UTM metres."""

    pixel_size: dict[str, float]
    """``{"x": 300.0, "y": 300.0}``, positive, in native UTM metres."""

    @property
    def shape(self) -> tuple[int, int]:
        """``(height, width)`` in pixels."""
        height = round((self.y_range[1] - self.y_range[0]) / self.pixel_size["y"])
        width = round((self.x_range[1] - self.x_range[0]) / self.pixel_size["x"])
        return height, width

    def lonlat_bounds(self, pad_deg: float = 0.0) -> tuple[float, float, float, float]:
        """``(minx, miny, maxx, maxy)`` in EPSG:4326, optionally padded.

        Used to ask ``lazy_dino`` for data catalogued in lon/lat (the Copernicus
        phenology) over this tile's footprint. The transform is densified, so a
        UTM box maps to the true curved lon/lat envelope rather than to the
        four reprojected corners, which would clip the edges.

        ``pad_deg`` widens the box on all sides. It exists because
        ``spatial_buffer`` -- ``lazy_dino``'s pixel-based equivalent -- raises
        for the LSP phenology variables, whose catalog entries declare no pixel
        size. A small pad guarantees the source covers the tile after warping.
        """
        from rasterio.warp import transform_bounds

        minx, miny, maxx, maxy = transform_bounds(
            f"EPSG:{self.epsg}",
            "EPSG:4326",
            self.x_range[0],
            self.y_range[0],
            self.x_range[1],
            self.y_range[1],
            densify_pts=21,
        )
        if pad_deg:
            minx, miny = minx - pad_deg, miny - pad_deg
            maxx, maxy = maxx + pad_deg, maxy + pad_deg
        # A tile straddling the antimeridian would need splitting, not clamping;
        # clamp only to the valid lon/lat domain so downstream bounds validate.
        return (
            max(minx, -180.0),
            max(miny, -90.0),
            min(maxx, 180.0),
            min(maxy, 90.0),
        )


@lru_cache(maxsize=8)
def list_tiles(variable: str) -> dict[str, Tile]:
    """Every tile of the UTM grid that ``variable`` is published on.

    ``variable`` must be a UTM mosaicset code such as ``"L1-UTM-NPP-D"``; the
    grid it is tiled on (``"L1-GRID"``) is resolved from the catalog, so L1 and
    L2 products each get their own grid without the caller knowing which.

    Requires network access. Cached per variable, since a tiled run asks for
    this once per tile otherwise.
    """
    if not is_utm_mosaicset(variable):
        raise ValueError(
            f"{variable!r} is not a WaPOR UTM mosaicset. Expected a code such as "
            "'L1-UTM-NPP-D' -- the plain mapsets ('L1-NPP-D') are a single global "
            "lon/lat grid, not tiled, and the L3 products are per project area."
        )
    discovery = GismgrDiscovery(cog_base=WAPOR_COG_BASE)
    grid_code = discovery.mosaicset_grid_code(variable)
    records = discovery.grid_tiles(grid_code)

    tiles: dict[str, Tile] = {}
    for code, record in records.items():
        x_range, y_range, pixel_size = GismgrDiscovery.tile_extent(record)
        tiles[code] = Tile(
            code=code,
            epsg=int(record["epsg"]),
            x_range=x_range,
            y_range=y_range,
            pixel_size=pixel_size,
        )
    return tiles


def get_tile(variable: str, code: str) -> Tile:
    """One tile by its MGRS grid-zone code, e.g. ``get_tile("L1-UTM-NPP-D", "36Q")``."""
    tiles = list_tiles(variable)
    try:
        return tiles[code]
    except KeyError:
        raise KeyError(
            f"{code!r} is not a tile of {variable!r}'s grid "
            f"({len(tiles)} tiles available)."
        ) from None
