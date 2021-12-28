from o3d_pose_lib import *
from path_lib import *

from scipy.spatial.transform import Rotation as R

scene_buff = []

model_root = 'D:/SIA/Dataset/Stanford/3D models/'

model_list = get_file_list(model_root)
print(model_list)

# 每个模型的变换

for i, model_name in enumerate(model_list):
    model_path = model_root + model_name + '/' + os.listdir(model_root + model_name)[0]
    # print(model_path)

    # 加载模型
    vox = 0.006
    pcd, pcd_down, pcd_down_fpfh = read_pcd(model_path, vox)

    # 不同的模型不同的变换
    trans = eye(4)

    r = R.from_rotvec(pi / 180 * array([i*10, i*30, 0]))  # 角度->弧度
    r_mat = r.as_matrix()
    # print(r_mat)

    # 旋转
    trans[:3, :3] = r_mat

    if i < 3:
        trans[:3, 3:] = array([i * 0.23, 0, 0]).reshape(3, -1)
        if i == 2:
            trans[:3, 3:] = array([i * 0.23, -0.1, 0]).reshape(3, -1)
    else:
        trans[:3, 3:] = array([(i-3) * 0.23, -0.25, 0]).reshape(3, -1)

        if i == 4:
            trans[:3, 3:] = array([(i - 3) * 0.23, -0.49, 0]).reshape(3, -1)

    print(trans[:3, 3:])  # 场景中模型的GT（没有旋转）  加点旋转看看是不是匹配错误？？

    pcd.transform(trans)

    # 放到同一个场景
    if i == 0:
        pcd_np = array(pcd.points)
    else:
        pcd_np = r_[pcd_np, array(pcd.points)]

    print(len(pcd_np))

    # 保存GT
    gt_path = 'D:/SIA/Dataset/Stanford/ANTennnaScene/gt_pose/' + model_name + '.txt'
    savetxt(gt_path, trans)

# 最后将叠加后的集成起来
pcd_scene = o3d.geometry.PointCloud()
pcd_scene.points = o3d.utility.Vector3dVector(pcd_np)
# pcd_scene.colors
pcd_scene.paint_uniform_color(array([30, 144, 255]) / 255)

pcd_scene.estimate_normals()

# 保存
save_path = 'D:/SIA/Dataset/Stanford/ANTennnaScene/6modelScene.ply'
o3d.io.write_point_cloud(save_path, pcd_scene)

show_pcd(pcd_scene)
