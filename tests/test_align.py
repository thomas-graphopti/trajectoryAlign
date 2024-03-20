import numpy as np
from pathlib import Path
from modelAlign.data_preprocessing import load_poses, load_rtk_data
from modelAlign.align import data_association, aligner_SVD_2D, aligner_SVD_3D
from modelAlign.align import coarse_aligner_3D
from modelAlign.align import fine_aligner_3D
from modelAlign.align import coarse_to_fine_align

from unittest.mock import patch


def test_data_association():
    base_path = Path(__file__).parent
    pose_folder = base_path / 'test_datas/cameras'
    poses = load_poses(str(pose_folder))
    rtk_data_folder = base_path / 'test_datas/rtk'
    rtk_data, _ = load_rtk_data(str(rtk_data_folder))
    assert len(poses) > 0, "No pose data loaded."
    assert len(rtk_data) > 0, "No RTK data loaded."
    # Test the data association
    shifted_rtk, shifted_poses, variances = data_association(
        poses, rtk_data, 0)
    # assert length
    assert len(shifted_rtk) == len(shifted_poses), "Length not match."
    assert len(shifted_rtk) == len(variances), "Length not match."
    assert len(shifted_rtk) > 20, "No data associated."
    # size check
    assert shifted_rtk[0].shape == (3, ), "Size not match."
    assert shifted_poses[0].shape == (3, ), "Size not match."
    assert variances[0].shape == (3, ), "Size not match."


def test_aligner_SVD():
    base_path = Path(__file__).parent
    pose_folder = base_path / 'rtk_test_data_2/cameras'
    poses = load_poses(str(pose_folder))
    rtk_data_folder = base_path / 'rtk_test_data_2/rtk'
    rtk_data, _ = load_rtk_data(str(rtk_data_folder))
    shifted_poses, shifted_rtk, variances = data_association(
        poses, rtk_data, 0)
    R, t, error = aligner_SVD_2D(shifted_poses, shifted_rtk)
    assert R.shape == (2, 2), "Size not match."
    assert np.isclose(np.linalg.det(R), 1.0), "R is not a rotation matrix."
    assert t.shape == (2, ), "Size not match."
    assert error > 0, "Error not correct."
    assert error < 0.5, "Error not correct."


def test_aligner_SVD_3D():
    base_path = Path(__file__).parent
    pose_folder = base_path / 'rtk_test_data_2/cameras'
    poses = load_poses(str(pose_folder))
    rtk_data_folder = base_path / 'rtk_test_data_2/rtk'
    rtk_data, _ = load_rtk_data(str(rtk_data_folder))
    shifted_poses, shifted_rtk, variances = data_association(
        poses, rtk_data, 0)
    R, t, error = aligner_SVD_3D(shifted_poses, shifted_rtk)
    assert R.shape == (3, 3), "Size not match."
    # Assert R is a rotation matrix
    assert np.isclose(np.linalg.det(R), 1.0), "R is not a rotation matrix."
    assert t.shape == (3, ), "Size not match."
    assert error > 0, "Error not correct."
    assert error < 0.5, "Error not correct."


def test_coarse_aligner_3D():
    base_path = Path(__file__).parent
    pose_folder = base_path / 'rtk_test_data_2/cameras'
    poses = load_poses(str(pose_folder))
    rtk_data_folder = base_path / 'rtk_test_data_2/rtk'
    rtk_data, _ = load_rtk_data(str(rtk_data_folder))

    time_shift_interval = [-1, 1]
    interval_step = 0.1
    R, t, error, time_shift = coarse_aligner_3D(poses, rtk_data,
                                                time_shift_interval,
                                                interval_step)

    assert R.shape == (3, 3), "Size not match."
    assert np.isclose(np.linalg.det(R), 1.0), "R is not a rotation matrix."
    assert t.shape == (3, ), "Size not match."
    assert error >= 0, "Error not correct."
    assert time_shift >= time_shift_interval[
        0] and time_shift <= time_shift_interval[
            1], "Time shift not within the interval."


def test_fine_aligner_3D():
    base_path = Path(__file__).parent
    pose_folder = base_path / 'rtk_test_data_2/cameras'
    poses = load_poses(str(pose_folder))
    rtk_data_folder = base_path / 'rtk_test_data_2/rtk'
    rtk_data, _ = load_rtk_data(str(rtk_data_folder))
    best_time_shift = 0
    R, t, error, time_shift = fine_aligner_3D(poses,
                                              rtk_data,
                                              best_time_shift,
                                              step=0.01,
                                              max_iter=20)

    assert R.shape == (3, 3), "Size not match."
    assert np.isclose(np.linalg.det(R), 1.0), "R is not a rotation matrix."
    assert t.shape == (3, ), "Size not match."
    assert error >= 0, "Error not correct."
    assert time_shift >= best_time_shift - 0.01 * 10, "Time shift not correct."
    assert time_shift <= best_time_shift + 0.01 * 10, "Time shift not correct."
    coarse_R, coarse_t, coarse_error, _ = coarse_aligner_3D(
        poses, rtk_data, [-1, 1], 0.1)
    assert error <= coarse_error, "Fine error not less than coarse error."


def test_coarse_to_fine_align():
    base_path = Path(__file__).parent
    pose_folder = base_path / 'rtk_test_data_2/cameras'
    poses = load_poses(str(pose_folder))
    rtk_data_folder = base_path / 'rtk_test_data_2/rtk'
    rtk_data, _ = load_rtk_data(str(rtk_data_folder))
    R, t, error = coarse_to_fine_align(poses, rtk_data)
    assert R.shape == (3, 3), "Size not match."
    assert np.isclose(np.linalg.det(R), 1.0), "R is not a rotation matrix."
    assert t.shape == (3, ), "Size not match."
    assert error >= 0, "Error not correct."
    coarse_R, coarse_t, coarse_error, _ = coarse_aligner_3D(
        poses, rtk_data, [-1, 1], 0.1)
    fine_R, fine_t, fine_error, _ = fine_aligner_3D(poses, rtk_data, 0, 0.01,
                                                    20)
    assert error <= coarse_error, "Final error not less than coarse error."
    assert error <= fine_error, "Final error not less than fine error."
