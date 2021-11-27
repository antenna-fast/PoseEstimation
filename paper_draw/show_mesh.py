from o3d_impl import *
import os


# 用来加载模型库，并且可视化
# 颜色比较纠结

model_list = os.listdir('D:/SIA/Dataset/Stanford/3D models')

for model_name in model_list:
    model_root = 'D:/SIA/Dataset/Stanford/3D models/' + model_name
    # print(model_root)
    model_path = model_root + '/' + os.listdir(model_root)[0]
    print(model_path)

    # mesh_color = [0.3, 0.9, 0.0]
    # mesh_color = [0.3, 0.9, 0.0]
    mesh_color = [1, 0.5, 0.2]
    mesh = read_mesh(model_path, mesh_color)

    o3d.visualization.draw_geometries([
        mesh,
        # axis_pcd,
        ],
        window_name='ANTenna3D',
        zoom=1,
        front=[0, -0.1, -1],  # 相机位置
        lookat=[0, 0, 0],  # 对准的点
        up=[0, 1, 0.5],  # 用于确定相机右x轴
        # point_show_normal=True
    )