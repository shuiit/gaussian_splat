import numpy as np
import scipy.io
import pandas as pd


def get_dict_for_points3d(frames):
    """
    Create dictionaries mapping 3D voxel positions and their mean colors.

    This function processes voxel data from multiple frames to generate a dictionary
    that associates unique voxel identifiers with their 3D positions and another dictionary 
    that maps voxel identifiers to their corresponding mean color values.

    Args:
        frames (dict): A dictionary containing frame data, where each key is an image identifier 
                       and each value has attributes for voxel indices and pixel colors.

    Returns:
        tuple: A tuple containing:
            - dict: A dictionary mapping voxel identifiers to their 3D positions (list of floats).
            - dict: A dictionary mapping voxel identifiers to their mean color values (list of floats).
    """

    # generates the dictionary of all 3d-2d mappings from every frame. 
    colors = np.hstack([frames[im_name].color_of_pixel for im_name in frames.keys()])
    all_voxels = np.column_stack((np.vstack(([(frames[im_name].voxels_with_idx) for im_name in frames.keys()])),colors))
    unique_voxels  = np.unique(all_voxels[:,0:4],axis = 0)

    voxel_dict = {vxl[3]:list(vxl[0:3]) for vxl in unique_voxels}
    [voxel_dict[vxl[3]].extend(vxl[4:-1]) for vxl in all_voxels]

    # calculate mean color
    colors_dict = {vxl[3]:[] for vxl in unique_voxels}
    for vxl in all_voxels:
        colors_dict[vxl[3]].extend(vxl[6:])
    colors_dict = {key: [np.mean(np.array(values)).astype(int)]*3 for key, values in colors_dict.items()}
    return voxel_dict,colors_dict


def load_hull(body_wing,path):
    """
    Load the 3D hull points for a specified body part from a .mat file.

    Args:
        body_wing (str): The name of the body part ('body', 'rwing', or 'lwing').
        path (str): The directory path where the .mat file is located.

    Returns:
        numpy.ndarray: An array containing the 3D hull points for the specified body part.
    """
    return scipy.io.loadmat(f'{path}/3d_pts/{body_wing}.mat')['hull']


def define_frames(frames,points_3d):
    """
    Generate a list of image names and a dictionary of 3D points for specified frames.

    Args:
        frames (list): A list of frame identifiers (integers).
        points_3d (dict): A dictionary containing DataFrames of 3D points for body parts 
                          ('body', 'rwing', 'lwing'), each with a 'frame' column.

    Returns:
        tuple: A tuple containing:
            - list: Image names for each frame and camera combination (e.g., ['P1CAM1.jpg', ...]).
            - dict: A dictionary mapping frame identifiers to concatenated DataFrames of 
                    3D points for the specified body parts.
    """

    image_name = []
    points_in_idx = {}
    for frame in frames:
        image_name += [f'P{frame}CAM{cam + 1}' for cam in range(4)]
        points_in_idx[f'P{frame}'] = pd.concat([points_3d[body_wing][points_3d[body_wing]['frame'] == frame] for body_wing in ['body','rwing','lwing']])
    return image_name,points_in_idx