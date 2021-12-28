import open3d as o3d
import numpy as np
import time
import matplotlib.pyplot as plt

# 读取去除远景之后的点云
# 去除大平面（地面背景），结果保存到fragment
# TODO：配准，得到三维模型

# 坐标系
coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

# source_path = 'C:/Users/yaohua-win/PycharmProjects/pythonProject/' \
#               'pose_sys_pnp/dataset/real_camera/bottle/'

# source_path = 'C:/Users/yaohua-win/PycharmProjects/pythonProject/' \
#               'pose_sys_pnp/dataset/real_camera/filted/bottle/'

# source_path = 'C:/Users/yaohua-win/PycharmProjects/pythonProject/pose_sys_pnp/dataset/real_camera/fountain/'
# source_path = 'C:/Users/yaohua-win/PycharmProjects/pythonProject/pose_sys_pnp/dataset/real_camera/fountain2/'
source_path = 'C:/Users/yaohua-win/PycharmProjects/pythonProject/pose_sys/dataset/real_camera/scene1/'

# target_path = 'C:/Users/yaohua-win/PycharmProjects/pythonProject/' \
#               'pose_sys_pnp/dataset/real_camera/bottle2/'

# target_path = 'C:/Users/yaohua-win/PycharmProjects/pythonProject/pose_sys_pnp/dataset/real_camera/fountain3/'
target_path = 'C:/Users/yaohua-win/PycharmProjects/pythonProject/pose_sys/dataset/real_camera/scene1_2/'

save_mode = 1
vis_mode = 1

start_num = 0  # 记得从1开始！否则并不直接提示没有文件的报错
max_num = 12

for num in range(start_num, max_num+1):

    file_path = source_path + str(num) + '.ply'
    pcd = o3d.io.read_point_cloud(file_path)
    # print(dir(pcd))

    # 使用RANSAC分割平面
    plane_model, inliers = pcd.segment_plane(distance_threshold=19.8,
    # plane_model, inliers = pcd.segment_plane(distance_threshold=9.8,
                                             ransac_n=200,
                                             num_iterations=1200)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    # print('dir:', dir(pcd))

    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])

    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    # outlier_cloud.paint_uniform_color([0, 1, 0])

    o3d.io.write_point_cloud(target_path + str(num) + '.ply', outlier_cloud)  # outlier的反而是我们想要的
    print('ground removed pcd saved as {0}'.format(target_path + str(num) + '.ply'))

    # Flip it, otherwise the pointcloud will be upside down
    inlier_cloud.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    outlier_cloud.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # # coord.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud], )
