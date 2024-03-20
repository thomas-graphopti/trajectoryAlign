import numpy as np

from pathlib import Path

from modelAlign.app import to_geoJson
from modelAlign.app import alignment


def test_to_geoJson():
    # Test case 1
    R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    t = np.array([0, 0, 0])
    origin = np.array([0, 0])
    expected_result = {
        "type": "LocaltoWGS84",
        "CoordinateSystem": "WGS84",
        "quaternion": [0.0, 0.0, 0.0, 1.0],
        "translation": [0.0, 0.0, 0.0],
        "origin": [0.0, 0.0]
    }
    assert to_geoJson(R, t, origin) == expected_result

    # Test case 2
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    t = np.array([1, 2, 3])
    origin = np.array([4, 5])
    expected_result = {
        "type": "LocaltoWGS84",
        "CoordinateSystem": "WGS84",
        "quaternion": [0.0, 0.0, 0.7071067811865475, 0.7071067811865475],
        "translation": [1.0, 2.0, 3.0],
        "origin": [4.0, 5.0]
    }
    assert to_geoJson(R, t, origin) == expected_result


def test_alignment():
    base_path = Path(__file__).parent
    pose_folder = base_path / 'rtk_test_data_2/cameras'
    rtk_data_folder = base_path / 'rtk_test_data_2/rtk'
    json = alignment(str(rtk_data_folder), str(pose_folder))
    assert json is not None, "No JSON returned."
    print(json)
