from .align import coarse_to_fine_align
from .data_preprocessing import load_poses, load_rtk_data
from scipy.spatial.transform import Rotation


def alignment(rtk_folder, pose_folder):
    '''
    Aligns the poses from the given pose folder with the RTK data from the RTK folder.
    
    Args:
        rtk_folder (str): The path to the folder containing the RTK data.
        pose_folder (str): The path to the folder containing the pose data.
    
    Returns:
        str: The JSON representation of the aligned data.
    '''
    poses = load_poses(pose_folder)
    rtk_data, origin = load_rtk_data(rtk_folder)
    R, t, error = coarse_to_fine_align(poses, rtk_data)
    json = to_geoJson(R, t, origin)
    return json


def to_geoJson(R, t, origin):
    '''
    Convert rotation matrix, translation vector, and origin point to a GeoJSON object.
    
    Args:
        R (numpy.ndarray): Rotation matrix.
        t (numpy.ndarray): Translation vector.
        origin (numpy.ndarray): Origin point.
        
    Returns:
        dict: GeoJSON object representing the transformation.
    '''
    q = Rotation.from_matrix(R).as_quat()
    geo_json = {
        "type": "LocaltoWGS84",
        "CoordinateSystem": "WGS84",
        "quaternion": [q[0], q[1], q[2], q[3]],
        "translation": [t[0], t[1], t[2]],
        "origin": [origin[0], origin[1]]
    }
    return geo_json
