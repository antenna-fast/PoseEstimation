from ..utils.o3d_pose_lib import *
import os

pcd_root = 'D:/SIA/Dataset/CVLab/3D Models/2010-03-03/filted/'
pcd_name_list = os.listdir(pcd_root)

for model_name in pcd_name_list:
    pcd_path = pcd_root + model_name
    pcd, pcd_down, pcd_down_fpfh = read_pcd(pcd_path, 0.4)
    print(pcd)

    o3d.visualization.draw_geometries([
        pcd,
        # axis_pcd,
    ],
        # window_name='ANTenna3D',
        # zoom=1,
        # front=[0, -0.1, -1],  # 相机位置
        # lookat=[0, 0, 0],  # 对准的点
        # up=[0, 1, 0.5],  # 用于确定相机右x轴
        # point_show_normal=True
    )