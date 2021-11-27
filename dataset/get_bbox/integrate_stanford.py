from o3d_impl import *
from o3d_pose_lib import *

import os

# 加载斯坦福模型，施加不同的变换做成场景
# 不同的场景加载不同的模型

# 这样固定视角后，颜色就会比较一致了

# 编码
class_encode = ['armadillo', 'buddha', 'bunny', 'chinese_dragon', 'dragon', 'statuette']
color_map = {0: [1, 0, 0], 1: [0.2, 1, 0], 2: [0.1, 0.8, 0.5], 3: [1, 1, 0], 4: [0.2, 0.8, 1], 5: [1, 0, 1]}

# 假定一个场景有三个模型
scene_model_list = [1, 2, 3]

trans_list = [[0, 0, 0],
              [0.5, 0, 0],
              [1, 0, 0]]

voxel_size = 0.005
root_path = 'D:/SIA/Dataset/Stanford/3D models/'
model_list = os.listdir(root_path)

for i in scene_model_list:

    model_path = root_path + model_list[i] + '/' + os.listdir(root_path + model_list[i])[0]

    # model path, voxel size
    pcd, pcd_down, pcd_down_fpfh = read_pcd(model_path, voxel_size)  # 返回 pcd, pcd_down, pcd_down_fpfh

    model_color = [0.5, 0.5, 0.7]
    show_pcd(pcd, model_color)
