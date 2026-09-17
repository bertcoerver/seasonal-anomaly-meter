"""Basic import and version tests."""

import seasonal_anomaly_meter


def test_version():
    """Test that version is defined."""
    assert hasattr(seasonal_anomaly_meter, "__version__")
    assert isinstance(seasonal_anomaly_meter.__version__, str)
