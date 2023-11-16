import os
import numpy as np
import hou

import icp

def icp_houdini(max_iterations: int=20, tolerance: float=0.001):
    first_input_geo = hou.pwd().geometry()
    second_input_geo = hou.pwd().inputs()[1].geometry()
    fisrt_input_points = first_input_geo.points()
    second_input_points = second_input_geo.points()
    src_points = np.array([point.position() for point in fisrt_input_points])
    dst_points = np.array([point.position() for point in second_input_points])
    result_pos = icp.icp_pos(src_points, dst_points)
    for idx, point in enumerate(fisrt_input_points):
        pos = result_pos[idx]
        point.setPosition(result_pos[idx])

def save_point_clouds():
    first_input_geo = hou.pwd().geometry()
    second_input_geo = hou.pwd().inputs()[1].geometry()
    fisrt_input_points = first_input_geo.points()
    second_input_points = second_input_geo.points()
    src_points = np.array([point.position() for point in fisrt_input_points])
    dst_points = np.array([point.position() for point in second_input_points])
    data_dir_path = "{}/data".format(os.path.dirname(os.path.realpath(__file__)).rsplit("/", 1)[0])
    with open("{}/src_points.npy".format(data_dir_path), 'wb') as f:
        np.save(f, src_points)
    with open("{}/dst_points.npy".format(data_dir_path), 'wb') as g:
        np.save(g, dst_points)
