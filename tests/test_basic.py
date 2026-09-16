"""Basic import and version tests."""

import wapor_anomaly_meter


def test_version():
    """Test that version is defined."""
    assert hasattr(wapor_anomaly_meter, "__version__")
    assert isinstance(wapor_anomaly_meter.__version__, str)
