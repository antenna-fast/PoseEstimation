"""
LineMOD dataset pre-processing

"""

import copy
import os
import numpy as np
import open3d as o3d


def show_pointcloud(pcd_path, is_mesh=0, is_down_sample=0):
    print("Load a ply point cloud, print it, and render it")
    if is_mesh:
        pcd = o3d.io.read_triangle_mesh(pcd_path)
    else:
        pcd = o3d.io.read_point_cloud(pcd_path)

        if is_down_sample:
            pcd = pcd.voxel_down_sample(voxel_size=0.01)

        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # pcd.paint_uniform_color([0, 0.40, 0.1])
    # pcd_np = np.asarray(pcd.points)  # 将点转换为numpy数组

    o3d.visualization.draw_geometries([pcd],
                                      window_name='ANTenna3D',
                                      # zoom=0.3412,
                                      # front=[0.4257, -0.2125, -0.8795],
                                      # lookat=[2.6172, 2.0475, 1.532],
                                      # up=[-0.0694, -0.9768, 0.2024]
                                      )


def show_pcds(pcd_paths, gt=np.array([0])):
    pcd_list = []
    for p in pcd_paths:
        pcd = o3d.io.read_point_cloud(p)
        pcd_list.append(pcd)

    if not gt.any() == 0:
        pcd_list[0] = pcd_list[0].transform(gt)

    o3d.visualization.draw_geometries(pcd_list,
                                      window_name='ANTenna3D',
                                      # zoom=0.3412,
                                      # front=[0.4257, -0.2125, -0.8795],
                                      # lookat=[2.6172, 2.0475, 1.532],
                                      # up=[-0.0694, -0.9768, 0.2024]
                                      )

    print()


if __name__ == '__main__':
    # load scene
    root_path = '/Users/aibee/Downloads/Dataset/3DDataset/6DPose/UWA'
    folder = 'scene/unzip'
    scene = 'rs1'
    # scene = 'Scene0_1-8'  # 下采样了的
    scene_path = os.path.join(root_path, folder, scene + '.ply')

    # load gt
    model_list = ['chef', 'chicken', 'parasaurolophus', 'T-rex']  # there are only 4 model's GT

    # model_name = 'chicken'
    model_name = model_list[0]
    print('viewing model: {}'.format(model_name))
    gt_file = os.path.join(root_path, 'gt', 'GroundTruth_3Dscenes', model_name + '-' + scene + '.xf')
    # gt_f = open(gt_file)
    # gt = gt_f.readlines()
    gt = np.loadtxt(gt_file)

    # load model
    map_gt_to_model = {'chef': 'cheff', 'chicken': 'chicken_high',
                       'parasaurolophus': 'parasaurolophus_high',
                       'T-rex': 'T-rex_high'}
    model_file = os.path.join(root_path, 'model/unzip', map_gt_to_model.get(model_name) + '.ply')
    model = o3d.io.read_point_cloud(model_file)

    # show_pcds([model_file, scene_path], gt=0)
    show_pcds([model_file, scene_path], gt=gt)

    # show_pointcloud(pcd_path=scene_path)
    print()
