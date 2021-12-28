import numpy as np
import open3d as o3d
import matplotlib.pylab as plt
# from o3d_pose_lib import *

'''
DBSCAN  保存最大簇
'''

model_name = 'C:/Users/yaohua-win/PycharmProjects/pythonProject/pose_sys_pnp/' \
             'dataset/real_camera/scene1_2/'


for num in range(0, 3):
    # file_path = '../data/fragment.ply'
    file_path = 'scene1_2/' + str(num) + '.ply'

    # 下采样
    # model_path, trans_init, voxel_size
    # source, source_down
    trans_init = np.eye(4)
    voxel_size = 0.01
    pcd, pcd_down = load_model(file_path, trans_init, voxel_size)
    print(pcd)

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(pcd.cluster_dbscan(eps=10.015, min_points=50, print_progress=True))

    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))  # 颜色映射  序列
    # colors[labels < 0] = 0

    # print(len(labels))  # 每个点的标签

    bin_count = np.bincount(labels)  # 里面不能有负数
    print(bin_count)

    bin_max = max(bin_count)  # 最多的点有多少个

    # 最大数量的：
    max_label = np.where(bin_count == bin_max)[0][0]
    print('max_label:', max_label)
    label_mask = np.where(labels == max_label)[0]
    print('label_mask:', label_mask)

    # pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    p2 = pcd.select_by_index(label_mask)  # 选出了最大的

    o3d.io.write_point_cloud(model_name + str(num) + '.ply', p2)  # outlier的反而是我们想要的

    o3d.visualization.draw_geometries([
        # pcd,
        p2,
                                       ],
                                      zoom=0.455,
                                      front=[-0.4999, -0.1659, -0.8499],
                                      lookat=[2.1813, 2.0619, 2.0999],
                                      up=[0.1204, -0.9852, 0.1215]
    )
