import open3d as o3d
import numpy as np
import time
from ml_lib import *

import os

# TODO：去除离群的点+

source_path = 'D:/SIA/Dataset/RealSense D435i/original/'
target_path = 'D:/SIA/Dataset/RealSense D435i/filted/'


file_list = os.listdir(source_path)
print('file_list:', file_list)
# file_list = ['bear.ply', 'mboy.ply']

for name in file_list:

    pcd = o3d.io.read_point_cloud(source_path + name)
    print(pcd)

    # DBSCAN clustering
    # 希望通过DBSCAN找到数量最多的簇，然后通过这些对应的索引进行截取
    # labels = np.array(pcd.cluster_dbscan(eps=0.5, min_points=5, print_progress=True))
    # print('labels:', labels)  # -1：噪声  从0开始是聚类后
    # max_label = labels.max()  # 最大值：并不一定是想要的，而是要最多的值
    # print(f"point cloud has {max_label + 1} clusters")

    # print(dir(pcd))

    # 去除背景中的点
    # 假设前景物体占有最多的点
    pts = np.asarray(pcd.points)
    print('pts np')

    # scene_down_np, search_radius=0.005, min_sample=5, threshold_n=50, class_num=5
    seg_res = dbscan_segment(pts, 0.005, 15, 5, 5)
    # print('seg_res:', seg_res)
    print('seg_res')

    # 找到label中元素最多的
    seg_cluster_num = len(seg_res)
    init_num = 0
    for idx in range(seg_cluster_num):
        cluster_num = len(seg_res[idx])
        if cluster_num > init_num:
            init_num = cluster_num  # 更新最大的
            init_idx = idx

    pcd = pcd.select_by_index(seg_res[init_idx])

    o3d.io.write_point_cloud(target_path + name, pcd)
    print('dense based outlier removed pcd saved as {0}'.format(target_path + name))

    # Flip it, otherwise the pointcloud will be upside down
    # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    o3d.visualization.draw_geometries([pcd,
                                       # aabb,
                                       # obb
                                       ],
                                      # zoom=0.7,
                                      # front=[0.5439, -0.2333, -0.8060],
                                      # lookat=[2.4615, 2.1331, 1.338],
                                      # up=[-0.1781, -0.9708, 0.1608]
                                      )
