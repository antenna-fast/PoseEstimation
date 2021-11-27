# TODO:给定图像的关键点坐标，根据深度相机得到三维空间坐标
# 目的是为了模型注册

# 输入： 图片
# 输出： 3d点
# 规定图像尺度为像素
# 实际坐标的尺度为mm

import pyrealsense2 as rs
from numpy import *
import numpy as np

import open3d as o3d
import cv2

import os
import time
from enum import IntEnum

from o3d_pose_lib import *

np.set_printoptions(suppress=True)


# 这个是将整个深度图转为点云
def get_cloud_xyz(depth, scale, u0, v0, fx, fy):
    global xmap, ymap
    zmap = depth.flatten()

    print('xmap:', xmap.shape)
    print('ymap:', ymap.shape)
    print('zmap:', zmap.shape)

    # Z = zmap * scale
    Z = zmap
    X = (xmap - v0) * Z / fx
    Y = (ymap - u0) * Z / fy

    X = X[:, newaxis].astype(np.float32)
    Y = Y[:, newaxis].astype(np.float32)
    Z = Z[:, newaxis].astype(np.float32)

    cloud = np.concatenate((X, Y, Z), axis=1)

    return cloud


# 添加颜色  问题：有一点对齐误差
def get_cloud_xyzrgb(depth, color, scale, u0, v0, fx, fy):
    global xmap, ymap

    # zmap = depth.flatten()
    zmap = depth.reshape(-1)
    # Z = zmap * scale  # 乘 就变成了m为单位
    Z = zmap
    Y = (xmap - v0) * Z / fy  # 因为检索时xy实际上是y行x列
    X = (ymap - u0) * Z / fx

    X = X[:, newaxis].astype(np.float32)  # 可以优化
    Y = Y[:, newaxis].astype(np.float32)
    Z = Z[:, newaxis].astype(np.float32)
    cloud = np.concatenate((X, Y, Z), axis=1)  # 拼接坐标

    # colors
    rgbs = color.reshape(-1, 3)

    return cloud, rgbs / 255


# 将单个xyZ转为XYZ  图像坐标-》相机坐标
def get_xyz(pix, Z, u0, v0, fx, fy):  # (x,y)
    Y = (pix[0] - v0) / fy * Z
    X = (pix[1] - u0) / fx * Z
    pt_w = np.array([X, Y, Z])
    return pt_w


if __name__ == "__main__":

    # 参数设置

    # 画幅
    # width = 1280
    # height = 720

    width = 512
    height = 424

    # 最简单的测试：给定一个xyz，得到XYZ
    # 像素坐标映射
    xmap, ymap = mgrid[0:height, 0:width]  # 前面是行范围 后面是列范围  对应到图像坐标，则xmap是y范围
    xmap, ymap = xmap.flatten(), ymap.flatten()

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_scale = 0.001
    print('depth_scale:', depth_scale)

    # We will not display the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 3  # 阈值 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # align_to = rs.stream.color  # 将depth对齐到color
    # align = rs.align(align_to)

    # 内参  ----------------------
    # intrinsic = get_intrinsic_matrix(color_frame)
    # print(intrinsic)
    # u0, v0 = intrinsic.ppx, intrinsic.ppy
    # fx, fy = intrinsic.fx, intrinsic.fy
    u0, v0 = width / 2, height / 2
    fx, fy = 0.001, 0.001

    dist_coeffs = zeros((4, 1))  # Assuming no lens distortion

    # 保存内参
    # intrinsic_mat = np.array([[fx, 0, u0],
    #                           [0, fy, v0],
    #                           [0, 0, 1]])

    # OPEN3D begain  ---------------------
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='ANTenna3D')

    # 设置窗口背景颜色
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0., 0.0])  # up to 1
    # print(dir(opt))

    pcd = o3d.geometry.PointCloud()
    flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]

    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=[0, 0, 0], size=0.1)  # 坐标系
    coord.transform(flip_transform)
    # OPEN3D end

    # Streaming loop
    frame_count = 0

    # while True:
    for frame_idx in range(1):

        s_time = time.time()

        frame_path = 'D:/SIA/Dataset/Kinect2/depth/1.bmp'

        # Align the depth frame to color frame
        # aligned_frames = align.process(frames)

        # 对齐 rgbd
        # 加载
        depth_img = cv2.imread(frame_path, 0)
        print(depth_img)
        print(shape(depth_img))

        # 整个点云
        pts = get_cloud_xyz(depth_img, depth_scale, u0, v0, fx, fy)
        # pts, color = get_cloud_xyzrgb(depth_img, rgb_img, depth_scale, u0, v0, fx, fy)
        # print(pts.shape)

        # # 使用open3d 查看效果
        pcd.points = o3d.utility.Vector3dVector(pts)  # 效率极低！ 30FPS -》 2.7FPS。。。
        # pcd.colors = o3d.utility.Vector3dVector(color)

        # show_pcd(pcd)

        # 写文件
        o3d.io.write_point_cloud(str(frame_count) + '.ply', pcd)

        pcd.transform(flip_transform)

        if frame_count == 0:
            vis.add_geometry(pcd)
            vis.add_geometry(coord)

        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

        frame_count += 1

        delta_time = time.time() - s_time
        print('FPS:', 1/delta_time)
