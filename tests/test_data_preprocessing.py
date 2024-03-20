import json
import numpy as np
import pytest
from pathlib import Path
from modelAlign.data_preprocessing import read_pose, pose_to_local, read_rtk_data
from modelAlign.data_preprocessing import rtk_data_to_local, read_all_rtk_data
from modelAlign.data_preprocessing import find_rtk_data_origin, load_rtk_data
from modelAlign.data_preprocessing import load_poses

from unittest.mock import patch


@pytest.fixture
def pose_data():
    return {
        "cx": 969.39349365234375,
        "neighbors": [],
        "width": 1920,
        "timestamp": 484009.04417091602,
        "t_20": 0.32118603587150574,
        "t_22": 0.6296045184135437,
        "t_11": 0.0055511882528662682,
        "t_23": -0.11549199372529984,
        "t_00": 0.32845363020896912,
        "t_12": 0.4593597948551178,
        "fy": 1512.2042236328125,
        "cy": 719.08648681640625,
        "fx": 1512.2042236328125,
        "t_10": -0.88823282718658447,
        "height": 1440,
        "t_02": 0.62656736373901367,
        "t_13": 0.2792048454284668,
        "t_03": -0.026150837540626526,
        "t_21": -0.70741617679595947,
        "globaltimestamp": 1709732809.038537,
        "t_01": 0.70677530765533447
    }


def test_read_pose(pose_data):
    # Assuming your project structure is as follows and the json file is stored accordingly
    base_path = Path(__file__).parent
    json_path = base_path / 'test_datas/cameras/1709732809.038537025451660.json'

    # Read the pose using your function
    result = read_pose(str(json_path))

    # Check if globaltimestamp is correctly converted
    assert result['timeStamp'] == pose_data[
        'globaltimestamp'], "Globaltimestamp does not match."

    # Expected homogeneous transformation matrix
    R = np.array([[pose_data['t_00'], pose_data['t_01'], pose_data['t_02']],
                  [pose_data['t_10'], pose_data['t_11'], pose_data['t_12']],
                  [pose_data['t_20'], pose_data['t_21'], pose_data['t_22']]])
    t = np.array([[pose_data['t_03']], [pose_data['t_13']],
                  [pose_data['t_23']]])
    expected_homogeneous_matrix = np.vstack((np.hstack((R, t)), [0, 0, 0, 1]))

    # Check if the homogeneous matrix matches the expected result
    np.testing.assert_array_almost_equal(
        result['matrix'],
        expected_homogeneous_matrix,
        decimal=6,
        err_msg="Homogeneous matrix does not match.")


def test_pose_to_local(pose_data):
    # Use the read_pose function to get the pose matrix from the fixture data
    base_path = Path(__file__).parent
    json_path = base_path / 'test_datas/cameras/1709732809.038537025451660.json'
    pose = read_pose(str(json_path))
    local_pose = pose_to_local(pose)
    expected_local_x = pose_data['t_03']  # Translation component in x
    expected_local_y = pose_data['t_13']  # Translation component in y
    expected_local_z = pose_data['t_23']  # Translation component in z
    assert local_pose['x'] == expected_local_x, "X coordinate does not match."
    assert local_pose['y'] == expected_local_y, "Y coordinate does not match."
    assert local_pose['z'] == expected_local_z, "Z coordinate does not match."
    assert local_pose['timeStamp'] == pose_data[
        'globaltimestamp'], "timeStamp not match."


@pytest.fixture
def expected_rtk_data():
    return {
        'timeStamp': 1709732809.035717,
        'createTime': 1709732809.0355492,
        'fixStatus': 0,
        'latitude': 31.227105105,
        'verticalAccuracy': 0.012,
        'height': 5.776,
        'diffStatus': '固定解',
        'horizontalAccuracy': 0.014,
        'longitude': 121.545474897
    }


def test_read_rtk_data(expected_rtk_data):
    base_path = Path(__file__).parent
    rtk_path = base_path / 'test_datas/rtk/1709732809.035717010498047.json'
    # Read the RTK data using the function under test
    result = read_rtk_data(str(rtk_path))
    # Assert that the processed data matches the expected data
    assert result == expected_rtk_data, "Processed RTK data does not match"


def test_rtk_data_to_local():
    rtk_data = {
        "timeStamp": 1709732809.035717,
        "latitude": 31.227105105,
        "longitude": 121.545474897,
        "height": 5.776,
        "fixStatus": 0,  # Adjust as necessary for different scenarios
        "diffStatus": "固定解",
        "horizontalAccuracy": 0.014,
        "verticalAccuracy": 0.012
    }
    origin = [31.227, 121.545]  # Mock origin
    expected_local_x = 0  # Example values, replace with correct calculations
    expected_local_y = 0  # Example values, replace with correct calculations
    expected_local_z = rtk_data["height"]
    expected_is_bad_data = False  # Based on the fixStatus in rtk_data
    expected_variance = 0.01  # Default value, as fixStatus is 0
    with patch('modelAlign.data_preprocessing.wgs84_to_cartesian',
               return_value=[expected_local_x, expected_local_y]):
        result = rtk_data_to_local(rtk_data, origin)

    assert result["x"] == expected_local_x and result["y"] == expected_local_y and \
           result["z"] == expected_local_z, "z does not match the expected result."
    assert result["is_bad_data"] == expected_is_bad_data, "is_bad_data wrong."
    assert result["variance"] == expected_variance, "Variance wrong."
    assert result['horizontalAccuracy'] == rtk_data[
        'horizontalAccuracy'], "horizontalAccuracy wrong."
    assert result['verticalAccuracy'] == rtk_data[
        'verticalAccuracy'], "verticalAccuracy wrong."  # NOTICE check the assert here
    rtk_data["diffStatus"] = "浮点解"
    result = rtk_data_to_local(rtk_data, origin)
    assert result["variance"] == 1.0, "Variance not match expecttion."
    assert result["is_bad_data"] == True, "is_bad_data flag match the expect."
    rtk_data["diffStatus"] = "单点解"
    result = rtk_data_to_local(rtk_data, origin)
    assert result["variance"] == 10.0, "Variance not match expecttion."
    assert result["is_bad_data"] == True, "is_bad_data flag match the expect."


def test_read_all_rtk_data():
    """Test the read_all_rtk_data function with the temporary RTK files."""
    base_path = Path(__file__).parent
    rtk_data_folder = base_path / 'test_datas/rtk'

    sorted_data = read_all_rtk_data(str(rtk_data_folder))

    # Verify the list is sorted by timeStamp
    timestamps = [data['timeStamp'] for data in sorted_data]
    assert timestamps == sorted(
        timestamps), "RTK data is not sorted by timeStamp."


def test_find_rtk_data_origin():
    base_path = Path(__file__).parent
    rtk_data_folder = base_path / 'test_datas/rtk'
    sorted_data = read_all_rtk_data(str(rtk_data_folder))
    origin = find_rtk_data_origin(sorted_data)
    assert origin == [31.227105105,
                      121.545474897], "Origin not found correctly."


def test_load_rtk_data():
    base_path = Path(__file__).parent
    rtk_data_folder = base_path / 'test_datas/rtk'
    all_rtk_data, _ = load_rtk_data(str(rtk_data_folder))
    # Assert not bad data
    assert all([not data['is_bad_data']
                for data in all_rtk_data]), "Bad data found."
    assert len(all_rtk_data) == 216, "Data length not match."
    # Assert all contain x, y, z, timeStamp, variance
    assert all([
        all(key in data for key in ['x', 'y', 'z', 'timeStamp', 'variance'])
        for data in all_rtk_data
    ]), "Data format not match."
    # Variance check, all data contains variance, within 1.0
    assert all([0 <= data['variance'] <= 1.0
                for data in all_rtk_data]), "Variance not match."
    # timeStamp check, increasing sorted by timeStamp
    timestamps = [data['timeStamp'] for data in all_rtk_data]
    assert timestamps == sorted(
        timestamps), "RTK data is not sorted by timeStamp."


def test_load_poses():
    base_path = Path(__file__).parent
    pose_folder = base_path / 'test_datas/cameras'
    all_poses = load_poses(str(pose_folder))
    # Assert all contain x, y, z, timeStamp
    assert all([
        all(key in data for key in ['x', 'y', 'z', 'timeStamp'])
        for data in all_poses
    ]), "Data format not match."
    # timeStamp check, increasing sorted by timeStamp
    timestamps = [data['timeStamp'] for data in all_poses]
    assert timestamps == sorted(
        timestamps), "Poses data is not sorted by timeStamp."
    # assert length
    assert len(all_poses) == 258, "Data length not match."
