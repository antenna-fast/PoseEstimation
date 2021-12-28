import os
import sys
import numpy as np
import open3d as o3d
from copy import deepcopy

sys.path.insert(0, os.getcwd())
from utils.o3d_impl import *


def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    # 基于RANSAC得到初始位姿  这里面集成了匹配  直接使用特征了
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


# 快速全局配准（M估计）
def execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" \
          % distance_threshold)
    result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result


def refine_registration(source, target, result_ransac, voxel_size):
    # ICP pose refine
    distance_threshold = voxel_size * 0.4
    # print(":: Point-to-plane ICP registration is applied on original point")
    # print("   clouds to refine the alignment. This time we use a strict")
    # print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result


def draw_registration_result(source, target, transformation=np.eye(4), win_name='ANTenna3D'):
    source_temp = deepcopy(source)
    target_temp = deepcopy(target)
    source_temp.transform(transformation)  # 变换source
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      window_name=win_name)


def draw_registration_with_bbox(source, target, bbox_gt, bbox_pred,
                                transformation, is_rgb=0, win_name='ANTenna3D'):
    source_temp = deepcopy(source)
    target_temp = deepcopy(target)

    if not is_rgb:
        source_temp.paint_uniform_color([1, 0.706, 0])  # 模型 对于rgb点云不进行绘制颜色
        target_temp.paint_uniform_color([0, 0.651, 0.929])  # 场景

    source_temp = source_temp.transform(transformation)  # 变换source

    o3d.visualization.draw_geometries([source_temp, target_temp, bbox_gt, bbox_pred],
                                      window_name=win_name,
                                      # zoom=1,
                                      # front=[0, -0.8, 0.1],  # 相机位置
                                      # lookat=[0, 0, 0],  # 对准的点
                                      # up=[0, 1, 0.1],  # 用于确定相机右x轴
                                      # point_show_normal=True
                                      )


# 从np恢复出bbox  生成LineSet吧！！！
def np2bbox(pts, width, color):
    # bbox = keypoints_np_to_spheres(pts, width, color)
    line_idx = np.array([[0, 3], [0, 2], [0, 1],  # 顶点连线
                      [1, 6], [1, 7],
                      [2, 5], [2, 7],
                      [3, 5], [3, 6],
                      [4, 5], [4, 6], [4, 7]])
    colors = np.tile(color, (len(line_idx), 1))  # 复制数组
    bbox = draw_line(pts, line_idx, colors)
    return bbox


# 直接输出bbox
# 输入：bbox模型坐标系下的坐标、GT（相对于给定顶点的初始位姿）
def get_bbox(bbox_vtx, trans_pose, color):
    ransac_rot_mat, ransac_trans = trans_pose[:3, :3], trans_pose[:3, 3:].reshape(-1)
    pred_bbox_trans = np.dot(ransac_rot_mat, bbox_vtx.T).T + ransac_trans
    bbox_pred = np2bbox(pred_bbox_trans, 0.1, color)
    return bbox_pred


# 场景分割效果
def draw_segmentation_result(target, is_rgb=0):
    target_temp = deepcopy(target)  # 深拷贝  否则里面
    # if not is_rgb:
    #     target_temp.paint_uniform_color([0, 0.651, 0.929])  # 场景

    # source_temp.translate([-1.0, 0, 0, 0])
    # target_temp.flip()
    o3d.visualization.draw_geometries([target_temp],
                                      # zoom=1,
                                      # front=[0, -1, 0.1],  # 相机位置
                                      # lookat=[0, 0, 0],  # 对准的点
                                      # up=[0, 1, 0.1],  # 用于确定相机右x轴
                                      )


# 为了可视化 生成连接线
# 输入模型和场景，以及各自匹配的索引  即可得到连接线
def get_line_set(model_np, scene_np, match_model_idx, match_scene_idx, color):
    model_vtx_num = len(model_np)
    # print('model_vtx_num:', model_vtx_num)  # md 纠结好久，原来是场景家的不对！索引不是顶点的
    line_vtx = np.r_[model_np, scene_np]  # 顶点坐标stack起来
    line_idx = np.c_[match_model_idx, match_scene_idx + model_vtx_num]  # idx的场景那一半得加上对应的长度

    colors = np.tile(color, (len(line_idx), 1))  # 复制数组
    line_set = draw_line(line_vtx, line_idx, colors)  # 对应点的匹配线  # points lines color

    return line_set


def preprocess_point_cloud(pcd, voxel_size):
    # 下采样，计算FPFH
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    # 下采样完了需要重新估计法向量
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_down_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_down_fpfh


# get point cloud feature for flann matching
def get_fpfh(pcd_in, voxel_size, max_nn=100):
    radius_feature = voxel_size * 5
    # print("Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_in, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=max_nn))
    return np.float32(pcd_fpfh.data.T)


def prepare_dataset(model_path, voxel_size, max_nn=100, trans=np.eye(4)):
    # load model
    source = o3d.io.read_point_cloud(model_path)
    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size, max_nn=30))
    # transform
    source.transform(trans)
    # down sample
    source_down = source.voxel_down_sample(voxel_size)
    # estimate normals
    radius_normal = voxel_size * 2
    source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    # get fpfh features
    radius_feature = voxel_size * 5
    source_down_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=max_nn))

    return source, source_down, source_down_fpfh


# 单纯加载模型并下采样
def load_model(model_path, trans_init):
    # print(":: Load one point cloud.")
    source = o3d.io.read_point_cloud(model_path)
    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    source.transform(trans_init)
    return source


# 读取单个点云  turn to prepare_dataset
# 下采样，计算FPFH
# def read_pcd(model_path, voxel_size):
#     pcd = o3d.io.read_point_cloud(model_path)
#     pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
#     pcd_down, pcd_down_fpfh = preprocess_point_cloud(pcd, voxel_size)
#     return pcd, pcd_down, pcd_down_fpfh
