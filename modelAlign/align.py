#TODOlists:
#1. RTK and pose alignment
#2. Rotate the camera to get a top-down view
#3. Rotate the camera to project the 3D model to align the North
#4. Project the 3D model to image
#5. Calculate the west-south point and the east-north point
#6. Generate the GeoJson file
import json
import numpy as np
import os
import sys
import math
import json

#NOTICE: pdb only for testing,delete it when you finish the code
import pdb

from types import SimpleNamespace
from geoToolbox import wgs84_to_cartesian, cartesian_to_wgs84


def data_association(pose_data, rtk_data, time_shift):
    """
    Performs data association between pose data and RTK data.

    Args:
        pose_data (list): List of dictionaries containing pose data.
        rtk_data (list): List of dictionaries containing RTK data.
        time_shift (float): Time shift value.

    Returns:
        tuple: A tuple containing three lists:
            - pose_shifted: List of shifted pose data.
            - rtk_data_shifted: List of shifted RTK data.
            - variances: List of variances associated with each data point.
    """
    pose_shifted = []
    rtk_data_shifted = []
    variances = []
    current_ar_index = 0
    rtk_index = 0  # index for rtk_data
    for rtk_datum in rtk_data:
        rtk_index += 1
        time_stamp = rtk_datum['timeStamp'] - time_shift
        for j in range(current_ar_index, len(pose_data) - 1):
            if pose_data[j]['timeStamp'] <= time_stamp and pose_data[
                    j + 1]['timeStamp'] > time_stamp:
                current_ar_index = j
                prev_timestamp = pose_data[j]['timeStamp']
                curr_timestamp = pose_data[j + 1]['timeStamp']
                prev_pose = np.array(
                    [pose_data[j]['x'], pose_data[j]['y'], pose_data[j]['z']])
                curr_pose = np.array([
                    pose_data[j + 1]['x'], pose_data[j + 1]['y'],
                    pose_data[j + 1]['z']
                ])
                percent = (time_stamp - prev_timestamp) / (curr_timestamp -
                                                           prev_timestamp)
                mid_pose = percent * (curr_pose - prev_pose) + prev_pose

                rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, -1, 0]])
                mid_pose = np.dot(rotation_matrix, mid_pose)
                variances.append(
                    np.array([
                        rtk_datum['variance'], rtk_datum['verticalAccuracy'],
                        rtk_datum['horizontalAccuracy']
                    ]))
                rtk_data_shifted.append(
                    np.array([rtk_datum['x'], rtk_datum['y'], rtk_datum['z']]))
                pose_shifted.append(mid_pose)
                break
            if pose_data[j + 1]['timeStamp'] > time_stamp:
                break  # should go to new rtk_datum
        if time_stamp > pose_data[-1]['timeStamp']:
            break
    return pose_shifted, rtk_data_shifted, variances


def aligner_SVD_2D(poses, rtk_datas):
    """
    Aligns 2D poses with corresponding RTK data using Singular Value Decomposition (SVD).

    Args:
        poses (list): List of 2D poses, where each pose is a list or array-like object with at least 2 elements.
        rtk_datas (list): List of RTK data points, where each data point is a list or array-like object with at least 2 elements.

    Returns:
        tuple: A tuple containing the rotation matrix (R), translation vector (t), and the alignment error.

    Raises:
        None

    """
    N = len(poses)
    #TODO: only get the pose first two elements,
    # and the first two elements of rkt
    temp_poses = np.array([[pose[0], pose[1]] for pose in poses])
    poses = temp_poses
    rtk_datas = np.array([[rtk_data[0], rtk_data[1]]
                          for rtk_data in rtk_datas])
    if N != len(rtk_datas) or N < 2:
        print("Wrong input data!")
        return None, None, None
    poses = np.array(poses)
    rtk_datas = np.array(rtk_datas)
    # Calculate mean
    poses_mean = np.mean(poses, axis=0)
    rtk_data_mean = np.mean(rtk_datas, axis=0)
    # Calculate Sigma
    Sigma = np.zeros((2, 2))
    for i in range(N):
        rtk_diff = (rtk_datas[i] - rtk_data_mean).reshape(2, 1)
        ar_diff = (poses[i] - poses_mean).reshape(1, 2)
        Sigma += (rtk_diff @ ar_diff) / N
    # Perform SVD
    U, S, Vt = np.linalg.svd(Sigma, full_matrices=True)
    W = np.identity(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        W[1, 1] = -1
    # Calculate rotation (R) and translation (t)
    R = U @ W @ Vt
    scale = 1.0
    t = rtk_data_mean - scale * (R @ poses_mean)
    # Calculate error
    error = 0
    for i in range(N):
        error += np.linalg.norm(rtk_datas[i] - (scale * R @ poses[i] + t)) / N

    return R, t, error


def aligner_SVD_3D(poses, rtk_datas):
    """
    Aligns 3D poses with corresponding RTK data using Singular Value Decomposition (SVD).

    Args:
        poses (list): List of 3D poses.
        rtk_datas (list): List of corresponding RTK data.

    Returns:
        tuple: A tuple containing the rotation matrix (R), translation vector (t), and error value.
    """
    N = len(poses)
    if N != len(rtk_datas) or N < 2:
        print("Wrong input data!")
        return None, None, None

    poses = np.array(poses)
    rtk_datas = np.array(rtk_datas)

    # Calculate mean
    poses_mean = np.mean(poses, axis=0)
    rtk_data_mean = np.mean(rtk_datas, axis=0)
    # Calculate Sigma
    Sigma = np.zeros((3, 3))
    for i in range(N):
        ar_diff = (poses[i] - poses_mean).reshape(1, 3)
        rtk_diff = (rtk_datas[i] - rtk_data_mean).reshape(3, 1)
        Sigma += (rtk_diff @ ar_diff) / N

    # Perform SVD
    U, _, Vt = np.linalg.svd(Sigma, full_matrices=True)
    W = np.identity(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        W[2, 2] = -1  # NOTICE: W:error here

    # Calculate rotation (R) and translation (t)
    R = U @ W @ Vt
    scale = 1.0
    t = rtk_data_mean - scale * (R @ poses_mean)

    # Calculate error
    error = 0
    for i in range(N):
        error += np.linalg.norm(rtk_datas[i] - (scale * R @ poses[i] + t)) / N
    return R, t, error


def coarse_aligner_3D(pose_data,
                      rtk_data,
                      time_shift_interval=[-1, 1],
                      coarse_step=0.1):
    '''
    Performs coarse alignment in 3D by finding the best rotation matrix (R), translation vector (t),
    alignment error, and time shift for a given pose data and RTK data.

    Args:
        pose_data (list): List of pose data points.
        rtk_data (list): List of RTK data points.
        time_shift_interval (int): Maximum time shift interval to consider.
        interval_step (int): Step size for iterating over the time shift interval.

    Returns:
        tuple: A tuple containing the best rotation matrix (R), translation vector (t),
        alignment error, and time shift.

    '''
    best_error = sys.float_info.max
    best_R = None
    best_t = None
    best_time_shift = 0
    left_edge, right_edge = time_shift_interval
    for i in np.arange(left_edge, right_edge + coarse_step, coarse_step):
        shifted_poses, shifted_rtk, variances = data_association(
            pose_data, rtk_data, i)
        R, t, error = aligner_SVD_3D(shifted_poses, shifted_rtk)
        if error < best_error:
            best_error = error
            best_R = R
            best_t = t
            best_time_shift = i
    return best_R, best_t, best_error, best_time_shift


def fine_aligner_3D(pose_data,
                    rtk_data,
                    best_time_shift,
                    step=0.01,
                    max_iter=20):
    '''
    Perform fine alignment of 3D pose data and RTK data.

    Args:
        pose_data (list): List of 3D pose data.
        rtk_data (list): List of RTK data.
        best_time_shift (int): The best time shift value.
        step (int): The step size for iterating over time shifts.
        max_iter (int, optional): Maximum number of iterations. Defaults to 10.

    Returns:
        tuple: A tuple containing the best rotation matrix (best_R), 
               the best translation vector (best_t), the best error (best_error),
               and the best time shift value (best_time_shift).
    '''
    best_error = sys.float_info.max
    best_R = None
    best_t = None
    best_time_shift = 0
    for i in np.arange(best_time_shift - max_iter / 2 * step,
                       best_time_shift + max_iter / 2 * step, step):
        shifted_poses, shifted_rtk, variances = data_association(
            pose_data, rtk_data, i)
        R, t, error = aligner_SVD_3D(shifted_poses, shifted_rtk)
        if error < best_error:
            best_error = error
            best_R = R
            best_t = t
            best_time_shift = i

    return best_R, best_t, best_error, best_time_shift


def coarse_to_fine_align(pose_data,
                         rtk_data,
                         time_shift_interval=[-1, 1],
                         coarse_step=0.1,
                         fine_step=0.01):
    '''
    Aligns the pose data with the RTK data using a two-step alignment process.

    Args:
        pose_data (list): List of pose data.
        rtk_data (list): List of RTK data.
        time_shift_interval (list, optional): Time shift interval for coarse alignment. Defaults to [-1, 1].
        coarse_step (float, optional): Coarse alignment step size. Defaults to 0.1.
        fine_step (float, optional): Fine alignment step size. Defaults to 0.01.

    Returns:
        tuple: A tuple containing the rotation matrix (R), translation vector (t), and alignment error.
    '''
    _, _, _, best_coarse_time_shift = coarse_aligner_3D(
        pose_data, rtk_data, time_shift_interval, coarse_step=coarse_step)
    max_iter = math.ceil(coarse_step / fine_step) * 2
    R, t, error, _ = fine_aligner_3D(pose_data, rtk_data,
                                     best_coarse_time_shift, fine_step,
                                     max_iter)
    return R, t, error
