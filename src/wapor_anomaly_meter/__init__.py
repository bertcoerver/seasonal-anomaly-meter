"""wapor-anomaly-meter package."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("wapor-anomaly-meter")
except PackageNotFoundError:
    __version__ = "unknown"
