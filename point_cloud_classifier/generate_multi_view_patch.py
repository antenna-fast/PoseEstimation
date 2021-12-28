"""
Author: ANTenna on 2021/12/23 11:34 上午
aliuyaohua@gmail.com

Description:

"""

"""
1. load models 
2. set view 
3. crop out point patch
"""

import os
import numpy as np
import open3d as o3d


if __name__ == '__main__':
    root_path = '/Users/aibee/Downloads/Dataset/3DDataset/6DPose/UWA'

    model_name = 'parasaurolophus_high.ply'
    model_path = os.path.join(root_path, 'model', model_name)
    pcd = o3d.io.read_point_cloud(model_path)

    print("Define parameters used for hidden_point_removal")
    diameter = 100000
    camera = [1000, 2510, 100]  # camera location
    radius = diameter * 100

    print("Get all points that are visible from given view point")
    # camera location, radius
    _, pt_map = pcd.hidden_point_removal(camera, radius)

    print("Visualize result")
    pcd = pcd.select_by_index(pt_map)
    o3d.visualization.draw_geometries([pcd])
