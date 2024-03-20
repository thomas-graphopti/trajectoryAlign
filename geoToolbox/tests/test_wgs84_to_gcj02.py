import pytest
from geoToolbox.wgs84_to_gcj02 import wgs84_to_gcj02, gcj02_to_wgs84

# Test data: [(input_wgs84_lat, input_wgs84_lng, expected_gcj02_lat, expected_gcj02_lng)]
test_data = [
    (40.0, 116.0, 40.001213650174, 116.006141221789),
]


@pytest.mark.parametrize(
    "wgs84_lat, wgs84_lng, expected_gcj02_lat, expected_gcj02_lng", test_data)
def test_wgs84_to_gcj02(wgs84_lat, wgs84_lng, expected_gcj02_lat,
                        expected_gcj02_lng):
    # Perform the conversion
    result = wgs84_to_gcj02(wgs84_lat, wgs84_lng)

    # Assert that the results are close to the expected values within a small tolerance
    assert result[0] == pytest.approx(expected_gcj02_lat, abs=1e-7)
    assert result[1] == pytest.approx(expected_gcj02_lng, abs=1e-7)


@pytest.mark.parametrize(
    "expected_wgs84_lat, expected_wgs84_lng, gcj02_lat, gcj02_lng", test_data)
def test_gcj02_to_wgs84(gcj02_lat, gcj02_lng, expected_wgs84_lat,
                        expected_wgs84_lng):
    # Perform the conversion
    result = gcj02_to_wgs84(gcj02_lat, gcj02_lng)
    print(result)
    print(expected_wgs84_lat)
    print(expected_wgs84_lng)
    # Assert that the results are close to the expected values within a small tolerance
    assert result[0] == pytest.approx(expected_wgs84_lat, abs=1e-4)
    assert result[1] == pytest.approx(expected_wgs84_lng, abs=1e-4)
