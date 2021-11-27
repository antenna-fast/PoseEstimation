import open3d as o3d
from numpy import *
import os


target_path = 'D:/SIA/Dataset/Stanford/3D models/'
bbox_path = 'D:/SIA/Dataset/Stanford/model bbox/'
file_list = os.listdir(target_path)
print('file_list:', file_list)


for name in file_list:
    model_path = target_path + name + '/' + os.listdir(target_path + name )[0]
    pcd = o3d.io.read_point_cloud(model_path)
    print(pcd)

    # bbox
    aabb = pcd.get_axis_aligned_bounding_box()  # 绿
    aabb.color = (0, 1, 0)
    aabb_np = array(aabb.get_box_points())
    # print()
    # print(dir(aabb))
    # print(aabb)

    obb = pcd.get_oriented_bounding_box()  # 红
    obb.color = (1, 0, 0)
    obb_np = array(obb.get_box_points())

    # print('obb_np:', obb_np)

    # 变换
    # trans = array([13.5, 0.5, 1])
    # # ror_mat =
    # model_trans_init = eye(4)  # 初始化模型位姿，更好地可视化
    # model_trans_init[:3, 3:] = trans.reshape(3, -1)

    # pcd.transform(model_trans_init)
    #
    # obb.translate(trans)
    # obb.rotate(eye(3))

    # save_name = bbox_path + 'obb_' + os.path.splitext(name)[0] + '.txt'
    save_name = bbox_path + 'aabb_' + os.path.splitext(name)[0] + '.txt'
    savetxt(save_name, aabb_np)

    o3d.visualization.draw_geometries([pcd,
                                       aabb,
                                       obb
                                       ],
                                      # zoom=0.7,
                                      # front=[0.5439, -0.2333, -0.8060],
                                      # lookat=[2.4615, 2.1331, 1.338],
                                      # up=[-0.1781, -0.9708, 0.1608]
                                      )
