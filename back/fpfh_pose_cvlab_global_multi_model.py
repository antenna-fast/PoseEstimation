import copy
import numpy as np
from numpy import *
import cv2
import open3d as o3d


# TODO: 多类别：
# 是的，基于ransac内点概率作为分类依据


# def read_pcd(model_path, voxel_size):
#     # 读取单个点云
#     # 下采样，计算FPFH
#     pcd = o3d.io.read_point_cloud(model_path)
#     pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
#     pcd_down, pcd_fpfh = preprocess_point_cloud(pcd, voxel_size)
#     return pcd, pcd_down, pcd_fpfh


def draw_registration_result(source, target, transformation):
    # 可视化
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])  # 对于rgb点云不进行绘制颜色
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)  # 变换source

    # source_temp.translate([-1.0, 0, 0, 0])
    # target_temp.flip()

    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      # zoom=0.4559,
                                      # front=[0.6452, -0.3036, -0.7011],
                                      # lookat=[1.9892, 2.0208, 1.8945],
                                      # up=[-0.2779, -0.9482, 0.1556]
                                      )


# 预处理 下采样，计算FPFH
def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    return pcd_down, pcd_fpfh


# 加载数据
# 读取场景和目标，下采样，计算FPFH

def prepare_dataset(model_path, scene_path, voxel_size):
    print(":: Load two point clouds and disturb initial pose.")
    source = o3d.io.read_point_cloud(model_path)
    target = o3d.io.read_point_cloud(scene_path)

    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    return source, target, source_down, target_down, source_fpfh, target_fpfh


# 全局粗配准
# 基于RANSAC得到初始位姿  这里面集成了匹配
def execute_global_registration(source_down, target_down,
                                source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    # print(":: RANSAC registration on downsampled point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % voxel_size)
    # print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
             distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(1000, 0.999)
    )

    return result


# 快速全局配准（M估计降噪）
def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" % distance_threshold)
    result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result


# ICP 精细配准
def refine_registration(source, target, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    # print("   clouds to refine the alignment. This time we use a strict")
    # print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result


# CVLab3D dataset 1&2

if __name__ == '__main__':
    root_path = 'D:/SIA/Dataset/SHOT/'
    scene_path = root_path + 'dataset1_scenes/3D models/Stanford/Random/Scene0.ply'
    # scene_path = root_path + '/dataset3/3D models/CVLab/2009-10-27/Scene1.ply'
    model_path = root_path + 'dataset1-2_models/3D models/Stanford/buddha/happy_vrip_res3.ply'
    # model_path = root_path + '/dataset3/3D models/CVLab/2009-10-27/model1.ply'
    # scene_path = root_path + 'dataset4/3D models/CVLab/2010-06-12/Scene1/Scene1.ply'
    # model_path = root_path + 'dataset4/3D models/CVLab/2010-06-12/Scene1/model1.ply'

    # voxel_size = 0.255  # means 5cm for the dataset
    voxel_size = 0.005  # means 5cm for the dataset
    source, target, source_down, target_down, source_fpfh, target_fpfh = \
        prepare_dataset(model_path, scene_path, voxel_size)

    # test 施加了变换
    # trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
    #                          [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    # source.transform(trans_init)
    draw_registration_result(source, target, np.identity(4))

    # coarse registration
    # result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    result_ransac = execute_fast_global_registration(source_down, target_down,
                                                     source_fpfh, target_fpfh,
                                                     voxel_size)
    result_ransac_trans = result_ransac.transformation
    # print('result_ransac:\n', result_ransac)
    print('result_ransac:\n', result_ransac_trans)
    draw_registration_result(source_down, target_down, result_ransac_trans)

    # ICP pose refine
    result_icp = refine_registration(source, target, voxel_size)

    print('result_icp:\n', result_icp)
    draw_registration_result(source, target, result_icp.transformation)

