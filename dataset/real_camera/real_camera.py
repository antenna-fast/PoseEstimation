# TODO:给定图像的关键点坐标，根据深度相机得到三维空间坐标
# 目的是为了模型注册

import pyrealsense2 as rs
from numpy import *
import numpy as np

import open3d as o3d
import cv2

import os
import time
from enum import IntEnum

np.set_printoptions(suppress=True)


class Preset(IntEnum):
    Custom = 0
    Default = 1
    Hand = 2
    HighAccuracy = 3
    HighDensity = 4
    MediumDensity = 5


def get_intrinsic_matrix(frame):
    intrinsics = frame.profile.as_video_stream_profile().intrinsics
    # out = o3d.camera.PinholeCameraIntrinsic(640, 480, intrinsics.fx,
    #                                         intrinsics.fy, intrinsics.ppx,
    #                                         intrinsics.ppy)
    # return out
    return intrinsics


# 这个是将整个深度图转为点云
def get_cloud_xyz(depth, scale, u0, v0, fx, fy):
    global xmap, ymap
    zmap = depth.flatten()

    # Z = zmap * scale
    Z = zmap
    X = (xmap - v0) * Z / fx
    Y = (ymap - u0) * Z / fy

    X = X[:, newaxis].astype(np.float32)
    Y = Y[:, newaxis].astype(np.float32)
    Z = Z[:, newaxis].astype(np.float32)

    cloud = np.concatenate((X, Y, Z), axis=1)

    return cloud


# XYZ+RGB  问题：有一点对齐误差
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


# 将单个xyZ转为XYZ  图像坐标->相机坐标
def get_xyz(pix, Z, u0, v0, fx, fy):  # (x,y)
    Y = (pix[0] - v0) / fy * Z
    X = (pix[1] - u0) / fx * Z
    pt_w = np.array([X, Y, Z])
    return pt_w


def flann_init():
    # FLANN matcher
    # While using ORB, you can pass the following.
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,  # 12
                        key_size=12,  # 20
                        multi_probe_level=1)  # 2
    search_params = dict(checks=50)  # or pass empty dictionary  这是第二个字典，指定了索引里的树应该被递归遍历的次数\
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    return flann


# model_name = 'bottle/'
# model_name = 'clamp/'
# model_name = 'fountain/'
model_name = 'scene1/'

if __name__ == "__main__":

    # 参数设置
    key_points_num = 100

    # 画幅
    # width = 1280
    # height = 720

    width = 640
    height = 480

    # 最简单的测试：给定一个xy，得到XYZ
    # 像素坐标映射
    # xmap, ymap = mgrid[0:480, 0:640]  # 前面是行范围 后面是列范围  对应到图像坐标，则xmap是y范围
    xmap, ymap = mgrid[0:height, 0:width]  # 前面是行范围 后面是列范围  对应到图像坐标，则xmap是y范围
    xmap, ymap = xmap.flatten(), ymap.flatten()

    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()

    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
    # config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, 30)

    # Start streaming
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()

    # Using preset HighAccuracy for recording
    depth_sensor.set_option(rs.option.visual_preset, Preset.Default)
    # Custom = 0
    # Default = 1
    # Hand = 2
    # HighAccuracy = 3
    # HighDensity = 4
    # MediumDensity = 5
    #
    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_scale = depth_sensor.get_depth_scale()
    print('depth_scale:', depth_scale)

    # We will not display the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 3  # 阈值 meter
    clipping_distance = clipping_distance_in_meters / depth_scale
    # print(depth_scale)

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color  # 将depth对齐到color
    align = rs.align(align_to)

    # Get frameset of color and depth
    frames = pipeline.wait_for_frames()

    # Align the depth frame to color frame
    aligned_frames = align.process(frames)

    # Get aligned frames
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    # 内参  ----------------------
    intrinsic = get_intrinsic_matrix(color_frame)
    # print(intrinsic)
    print(intrinsic)
    width, height = intrinsic.width / 2, intrinsic.height / 2
    u0, v0 = intrinsic.ppx, intrinsic.ppy
    fx, fy = intrinsic.fx, intrinsic.fy

    dist_coeffs = zeros((4, 1))  # Assuming no lens distortion

    # 保存内参
    intrinsic_mat = np.array([[fx, 0, u0],
                              [0, fy, v0],
                              [0, 0, 1]])

    # if not os.path.exists('../../cam_intri.txt'):
    #     savetxt('cam_intri.txt', intrinsic_mat)

    # OPEN3D begain  ---------------------
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='ANTenna3D')

    # 设置窗口背景颜色
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0., 0.0])  # up to 1

    pcd = o3d.geometry.PointCloud()
    flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]

    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=[0, 0, 0], size=0.1)  # 坐标系
    coord.transform(flip_transform)

    # pose estimation
    flann = flann_init()

    # Streaming loop
    frame_count = 0

    pts_temp = []  # 因为有的地方并没有depth 所以我们只添加有depth的des

    try:
        while True:

            s_time = time.time()

            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_img = array(aligned_depth_frame.get_data())
            rgb_img = array(color_frame.get_data())
            # print(rgb_img.shape)

            # 整个点云
            # pts = get_cloud_xyz(depth_img, depth_scale, u0, v0, fx, fy)
            pts, color = get_cloud_xyzrgb(depth_img, rgb_img, depth_scale, u0, v0, fx, fy)
            # print(pts.shape)

            # # 使用open3d 查看效果
            pcd.points = o3d.utility.Vector3dVector(pts)  # 效率极低！！！ 30FPS -》 2.7FPS。。。
            pcd.colors = o3d.utility.Vector3dVector(color)

            # 写文件
            o3d.io.write_point_cloud(model_name + str(frame_count) + '.ply', pcd)

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

    finally:
        pipeline.stop()
