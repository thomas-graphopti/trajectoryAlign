import pytest
from geoToolbox.wgs84_to_cartesian import wgs84_to_cartesian, cartesian_to_wgs84

test_data = [
    (40.0, 116.0, 0.0, 40.10, 116.1, 0, 8526.9116, 11108.3367, -15.38994),
]


@pytest.mark.parametrize(
    "origin_lat, origin_lng, origin_alt, target_lat, target_lng, target_alt, expected_x, expected_y, expected_z",
    test_data)
def test_wgs84_to_cartesian(origin_lat, origin_lng, origin_alt, target_lat,
                            target_lng, target_alt, expected_x, expected_y,
                            expected_z):
    east, north = wgs84_to_cartesian([origin_lat, origin_lng],
                                     [target_lat, target_lng])
    assert east == pytest.approx(expected_x, abs=1e-3)
    assert north == pytest.approx(expected_y, abs=1e-3)


@pytest.mark.parametrize(
    "origin_lat, origin_lng, origin_alt, expected_lat, expected_lng, expected_alt, east, north, z",
    test_data)
def test_cartesian_to_wgs84(origin_lat, origin_lng, origin_alt, expected_lat,
                            expected_lng, expected_alt, east, north, z):
    lat, lon = cartesian_to_wgs84([origin_lat, origin_lng], [east, north])
    assert lat == pytest.approx(expected_lat, abs=1e-6)
    assert lon == pytest.approx(expected_lng, abs=1e-6)
