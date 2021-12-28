"""
Author: ANTenna on 2021/12/27 11:34 上午
aliuyaohua@gmail.com

Description:
load models, calculate diameter and save to file
"""

import os
import sys
import glob
import numpy as np
import open3d as o3d

sys.path.insert(0, os.getcwd() + '/..')
from utils.o3d_impl import get_aabb


def get_diameter(model_pcd):
    # get bbox
    # obb = model_pcd.get_oriented_bounding_box()
    aabb = get_aabb(model_pcd)
    diameter = np.linalg.norm([aabb[0], aabb[1]])
    return diameter


if __name__ == '__main__':
    root_path = '/Users/aibee/Downloads/Dataset/3DDataset/6DPose/UWA'
    model_path = os.path.join(root_path, 'model')
    model_list = glob.glob(model_path + '/*.ply')

    diam_save_path = os.path.join(root_path, 'model', 'model_diameter.npy')  # {model_name: diam}

    model_diameter = {}
    for model_file in model_list:
        model_name = os.path.basename(model_file).split('.')[0]
        # load model
        pcd = o3d.io.read_point_cloud(model_file)
        # pcd_np = np.array(pcd.points)

        # get diameter
        diam = get_diameter(pcd)
        model_diameter[model_name] = diam

        print('model: {}  diam: {}'.format(model_name, diam))
    np.save(diam_save_path, model_diameter)
