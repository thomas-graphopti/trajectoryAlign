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

#NOTICE: pdb only for testing,delete it when you finish the code
import pdb

from types import SimpleNamespace
from geoToolbox import wgs84_to_cartesian, cartesian_to_wgs84


def read_pose(pose_path):
    '''
    Read the pose from the pose file, convert globaltimestamp to double, 
    and construct a homogeneous transformation matrix. Package the timestamp and 
    homogeneous matrix into a single variable.
    
    Parameters:
    - pose_path: str, the file path to the pose JSON file.
    
    Returns:
    - A dictionary containing:
        - The pose data as a dictionary under the key 'pose_data'.
        - The globaltimestamp as a double under the key 'timestamp'.
        - The homogeneous transformation matrix as a numpy array under the key
        'homogeneous_matrix'.
    '''
    with open(pose_path, 'r') as file:
        pose_data = json.load(file)

    globaltimestamp = float(pose_data['globaltimestamp'])

    R = np.array([[pose_data['t_00'], pose_data['t_01'], pose_data['t_02']],
                  [pose_data['t_10'], pose_data['t_11'], pose_data['t_12']],
                  [pose_data['t_20'], pose_data['t_21'], pose_data['t_22']]])

    t = np.array([[pose_data['t_03']], [pose_data['t_13']],
                  [pose_data['t_23']]])

    homogeneous_matrix = np.vstack((np.hstack((R, t)), [0, 0, 0, 1]))

    # Packaging the pose data, timestamp, and homogeneous matrix into a dictionary
    processed_pose = {
        'timeStamp': globaltimestamp,  # Converted globaltimestamp to double
        'matrix': homogeneous_matrix  # Constructed homogeneous matrix
    }
    return processed_pose


def pose_to_local(pose):
    '''
    transform the pose to local pose
    The local pose coordinate system is a y-up coordinate system, 
    with the origin at the center of the model.
    '''
    local_x = pose['matrix'][0, 3]
    local_y = pose['matrix'][1, 3]
    local_z = pose['matrix'][2, 3]
    time_stamp = pose['timeStamp']
    local_pose = {
        'timeStamp': time_stamp,
        'x': local_x,
        'y': local_y,
        'z': local_z
    }
    return local_pose


def read_all_pose(folder_path):
    '''
    Reads all pose data JSON files in the specified folder, 
    aggregates them into a list,
    and sorts the list by their timestamp.
    
    Parameters:
    - folder_path: str, the path to the folder containing pose JSON files.
    
    Returns:
    - A sorted list of dictionaries, each containing the pose data, timestamp, 
    and homogeneous matrix.
    '''
    pose_data_list = []

    # List all files in the directory specified by folder_path
    for filename in os.listdir(folder_path):
        # Construct the full file path
        file_path = os.path.join(folder_path, filename)
        # Check if the file is a JSON file
        if os.path.isfile(file_path) and filename.endswith('.json'):
            # Read the pose data from the file
            pose_data = read_pose(file_path)
            # Add the data to the list
            pose_data_list.append(pose_data)

    # Sort the list by timeStamp
    pose_data_sorted = sorted(pose_data_list, key=lambda x: x['timeStamp'])

    return pose_data_sorted


def transfer_all_pose_to_local(poses):
    '''
    Transfers all poses to the local coordinate system.

    Parameters:
    - poses: list of dictionaries, each dictionary contains the pose data, 
    timestamp, 
    and homogeneous matrix.

    Returns:
    - A list of dictionaries, each containing the local pose coordinates 
    and timestamp.
    '''
    local_poses = []

    # Iterate through each pose in the list
    for pose in poses:
        # Convert the pose to the local coordinate system
        local_pose = pose_to_local(pose)
        # Append the local pose to the list
        local_poses.append(local_pose)

    # sort the list by timeStamp
    local_poses_sorted = sorted(local_poses, key=lambda x: x['timeStamp'])

    return local_poses_sorted


def read_rtk_data(rtk_path):
    '''
    Reads a single RTK data entry from an RTK file.
    
    Parameters:
    - rtk_path: str, the file path to the RTK JSON file.
    
    Returns:
    - A dictionary containing properties of the RTK data point,
    with values in their appropriate Python data types.
    '''
    with open(rtk_path, 'r') as file:
        rtk_data_raw = json.load(file)

    # Assuming you're reading the first entry for simplicity; adjust as needed
    data_point = rtk_data_raw['rtkData'][0]

    processed_point = {
        'timeStamp': float(data_point['timeStamp']),
        'createTime': float(data_point['createTime']),
        'fixStatus': int(data_point['fixStatus']),
        'latitude': float(data_point['latitude']),
        'verticalAccuracy': float(data_point['verticalAccuracy']),
        'height': float(data_point['height']),
        'diffStatus': str(data_point['diffStatus']),
        'horizontalAccuracy': float(data_point['horizontalAccuracy']),
        'longitude': float(data_point['longitude'])
    }

    return processed_point


def read_all_rtk_data(folder_path):
    '''
    Reads all RTK data JSON files in the specified folder, 
    aggregates them into a list,
    and sorts the list by their timeStamp.
    
    Parameters:
    - folder_path: str, the path to the folder containing RTK JSON files.
    
    Returns:
    - A sorted list of dictionaries, each containing properties 
    of an RTK data point.
    '''
    rtk_data_list = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and filename.endswith('.json'):
            rtk_data = read_rtk_data(file_path)
            rtk_data_list.append(rtk_data)

    rtk_data_sorted = sorted(rtk_data_list, key=lambda x: x['timeStamp'])

    return rtk_data_sorted


def rtk_data_to_local(rtk_data, origin):
    '''
    Transfers a single RTK data point to the local coordinate system.
    Parameters:
    - rtk_data: dictionary, contains properties of an RTK data point.
    - origin: list of two floats, the WGS84 coordinates of the origin point.
    Returns:
    - A dictionary containing the local RTK data coordinates and timestamp.
    '''
    WGS84Position = [rtk_data['latitude'], rtk_data['longitude']]
    WGS84Reference = origin
    CartesianPosition = wgs84_to_cartesian(WGS84Reference, WGS84Position)
    local_x = CartesianPosition[0]
    local_y = CartesianPosition[1]
    local_z = rtk_data['height']
    time_stamp = rtk_data['timeStamp']
    variance_map = {
        '单点解': 10.0,  # single
        '码差分': 5.0,  # single_2
        '固定解': 0.01,  # fixed
        '浮点解': 1.0,  # float
    }
    variance = variance_map.get(rtk_data['diffStatus'], 100.0)
    is_bad_data = rtk_data['diffStatus'] not in ('固定解')
    rtk_local_data = {
        'timeStamp': time_stamp,
        'x': local_x,
        'y': local_y,
        'z': local_z,
        'is_bad_data': is_bad_data,
        'variance': variance,
        'horizontalAccuracy': rtk_data['horizontalAccuracy'],
        'verticalAccuracy': rtk_data['verticalAccuracy']
    }
    return rtk_local_data


def transfer_all_rtk_data_to_local(rtk_data, origin):
    '''
    Transfers all RTK data to the local coordinate system.

    Parameters:
    - rtk_data: list of dictionaries, each dictionary contains 
    properties of an RTK data point.
    - origin: list of two floats, the WGS84 coordinates of the origin point.

    Returns:
    - A list of dictionaries, each containing the local RTK data coordinates
    and timestamp.
    '''
    local_rtk_data = []

    # Iterate through each RTK data point in the list
    for data_point in rtk_data:
        # Convert the RTK data to the local coordinate system
        local_data_point = rtk_data_to_local(data_point, origin)
        if local_data_point['is_bad_data']:
            continue
        # Append the local RTK data to the list
        local_rtk_data.append(local_data_point)

    return local_rtk_data


def find_rtk_data_origin(rtk_data):
    '''
    Find the origin point of the RTK data. The lowest accuracy rtk data datum 
    among all elements. if meet the first fixed datum, then return the datum

    Parameters:
    - rtk_data: list of dictionaries, each dictionary contains 
    properties of an RTK data point.

    Returns:
    - A list of two floats, the WGS84 coordinates of the origin point.
    '''
    #NOTICE the origin does not contain the height information, what is the height coordinate  ?
    # coordinate is WGS84 height or the height from the ground?
    lowest_accuracy = float('inf')
    origin = [0, 0]
    for data_datum in rtk_data:
        if data_datum['diffStatus'] == '固定解':
            return [data_datum['latitude'], data_datum['longitude']]
        current_accuracy = max(data_datum['horizontalAccuracy'],
                               data_datum['verticalAccuracy'])
        if current_accuracy < lowest_accuracy:
            origin = [data_datum['latitude'], data_datum['longitude']]
            lowest_accuracy = current_accuracy
    return origin


def load_rtk_data(rtk_data_folder):
    '''
    Loading all rtk data from the rtk data folder, and transfer them
    to the local coordinate system.

    '''
    rtk_data = read_all_rtk_data(rtk_data_folder)
    origin = find_rtk_data_origin(rtk_data)
    local_rtk_data = transfer_all_rtk_data_to_local(rtk_data, origin)
    return local_rtk_data, origin


def load_poses(pose_folder):
    '''
    Loading all poses from the pose folder, and transfer them
    to the local coordinate system.
    '''
    poses = read_all_pose(pose_folder)
    local_poses = transfer_all_pose_to_local(poses)
    return local_poses
