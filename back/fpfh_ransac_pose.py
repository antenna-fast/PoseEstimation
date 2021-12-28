# baseline ransac pose estimation in open3d


from utils.o3d_pose_lib import *


def draw_registration_result(source, target, transformation, is_rgb):
    # 可视化
    source_temp = deepcopy(source)
    target_temp = deepcopy(target)

    if not is_rgb:
        source_temp.paint_uniform_color([1, 0.706, 0])  #
        target_temp.paint_uniform_color([0, 0.651, 0.929])

    source_temp.transform(transformation)  # 将source对齐到target  模型对齐到场景
    # source_temp.translate([-1, 0, 0, 0])
    # target_temp.flip()

    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      # zoom=0.4559,
                                      # front=[0.6452, -0.3036, -0.7011],
                                      # lookat=[1.9892, 2.0208, 1.8945],
                                      # up=[-0.2779, -0.9482, 0.1556]
                                      )


def prepare_dataset(model_path, scene_path, voxel_size):
    # 准备数据
    # 读取场景和目标，下采样，计算FPFH
    print(":: Load two point clouds and disturb initial pose.")
    source = o3d.io.read_point_cloud(model_path)  # 模型
    target = o3d.io.read_point_cloud(scene_path)  # 场景
    # estimate normals
    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    draw_registration_result(source, target, np.identity(4), 1)

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


if __name__ == '__main__':
    # path
    root_path = '/Users/aibee/Downloads/Dataset/3DDataset/6DPose/'
    scene_path = root_path + 'CVLab/2010-03-03/Scene1/scene1.ply'  # 小人
    model_path = root_path + 'CVLab/2010-03-03/Scene1/model1.ply'

    # scene_path = 'D:/SIA/Dataset/FoodPackage/realsense/1.ply'  #
    # model_path = 'D:/SIA/Dataset/FoodPackage/out_remove/1.ply'

    # pose estimation parameters
    voxel_size = 0.001  # means 5cm for the dataset
    # voxel_size = 0.4  # means 5cm for the dataset  对于Kinect的场景 要大一些
    # voxel_size = 0.005  # means 5cm for the dataset  对于Kinect的场景 要大一些
    source, target, source_down, target_down, source_fpfh, target_fpfh = \
        prepare_dataset(model_path, scene_path, voxel_size)

    # RANSAC coarse
    # result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    result_ransac = execute_fast_global_registration(source_down, target_down, source_fpfh,
                                                     target_fpfh, voxel_size)
    result_ransac_trans = result_ransac.transformation
    # vis ransac result
    draw_registration_result(source_down, target_down, result_ransac_trans, 1)

    # ICP refine
    result_icp = refine_registration(source, target, result_ransac_trans, voxel_size)
    print(result_icp.transformation)

    # vis ICP refined result
    draw_registration_result(source, target, result_icp.transformation, 1)
