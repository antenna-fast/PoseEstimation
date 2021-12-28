# a open3d function wrap

import os
import sys
import numpy as np
from numpy import linalg
from scipy.spatial import Delaunay
import open3d as o3d

sys.path.insert(0, os.getcwd() + '/utils')
from utils.geometry_lib import pt_to_plane, get_plan, get_coord  # 点投影到面


# IO modules
def read_mesh(mesh_path, mesh_color=[0.0, 0.6, 0.1]):
    # 数据加载
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(mesh_color)
    return mesh


# Mesh
def get_non_manifold_vertex_mesh(verts, triangles):
    # 从点和索引构建mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    # mesh.compute_triangle_normals()
    return mesh


# 自定义数据处理
# 从顶点创建mesh
# 要改名成create_mesh
def get_mesh(now_pt, vici_pts):
    # 得到邻域构成的面片 得到面片法向量  顶点法向量
    coord = get_coord(now_pt, vici_pts)  # 列向量表示三个轴
    normal = coord[:, 2]  # 第三列
    # print('coord:\n', coord)

    # 还有一步 利用法向量和中心点,得到平面方程[ABCD]
    p = get_plan(normal, now_pt)

    # * 找到拓扑结构 START
    all_pts = np.vstack((now_pt, vici_pts))
    # 将周围的点投影到平面
    # plan_pts = []
    # for pt in all_pts:  # 投影函数可以升级  向量化
    plan_pts = pt_to_plane(all_pts, p, normal)  # px p pn
    # plan_pts = array(plan_pts)  # 投影到平面的点

    # 将投影后的点旋转至z轴,得到投影后的二维点
    coord_inv = linalg.inv(coord)  # 反变换
    # rota_pts = dot(coord_inv, all_pts.T).T  # 将平面旋转到与z平行
    # 首先要将平面上的点平移到原点 然后再旋转  其实不平移也是可以的，只要xy平面上的结构不变
    rota_pts = np.dot(coord_inv, plan_pts.T).T  # 将平面旋转到与z平行

    # rota_pts[:, 2] = 0  # 已经投影到xoy(最大平面),在此消除z向轻微抖动
    pts_2d = rota_pts[:, 0:2]

    # Delauney三角化
    tri = Delaunay(pts_2d)
    tri_idx = tri.simplices  # 三角形索引
    # print(tri_idx)

    # 统计三角形的数量
    # tri_num = len(tri_idx)
    # print(tri_num)

    # 可视化二维的投影
    # plt.triplot(pts_2d[:, 0], pts_2d[:, 1], tri.simplices.copy())
    # plt.plot(pts_2d[:, 0], pts_2d[:, 1], 'o')
    # plt.show()
    # * 找到拓扑结构 END

    # 根据顶点和三角形索引创建mesh
    mesh = get_non_manifold_vertex_mesh(all_pts, tri_idx)

    # 求mesh normal
    mesh.compute_triangle_normals()
    mesh_normals = np.array(mesh.triangle_normals)
    # print(mesh_normals)

    # 法向量同相化
    for i in range(len(mesh_normals)):
        if np.dot(mesh_normals[i], normal) < 0:
            mesh_normals[i] = -mesh_normals[i]
    mesh.triangle_normals = o3d.utility.Vector3dVector(mesh_normals)

    return mesh, mesh_normals, normal


# Format Transformation
def keypoints_to_spheres(keypoints):
    # 数据变换
    # 将pcd格式点转换成球
    # This function is only used to make the keypoints look better on the rendering
    spheres = o3d.geometry.TriangleMesh()
    for keypoint in keypoints.points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.001)
        # sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sphere.translate(keypoint)
        spheres += sphere
    spheres.paint_uniform_color([1.0, 0.75, 0.0])
    return spheres


def keypoints_np_to_spheres(keypoints, size=0.1, color=[0, 1, 0]):
    # np格式的
    spheres = o3d.geometry.TriangleMesh()
    for keypoint in keypoints:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=size)
        # sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sphere.translate(keypoint)
        spheres += sphere
    spheres.paint_uniform_color(color)
    return spheres


def mesh2pcd(mesh_in):
    mesh_vertices = np.array(mesh_in.vertices)  # nx3
    # print('pcd1_num:', len(mesh_vertices))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(mesh_vertices)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=8, max_nn=10))

    return pcd


def mesh2np(mesh_in):
    mesh_vertices = np.array(mesh_in.vertices)  # nx3
    return mesh_vertices


def np2pcd(pcd_np, color=[0, 1, 0]):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_np)
    pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (len(pcd.points), 1)))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=10))
    return pcd


# 可视化
def draw_line(points, lines, colors):
    # parameters:
    # points 顶点坐标  np格式
    # lines 索引  nx2索引
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


# axis-aligned bounding box
def get_aabb(input_pcd):
    input_pcd_np = np.array(input_pcd.points)
    min_point = [input_pcd_np[:, 0].min(),
                 input_pcd_np[:, 1].min(),
                 input_pcd_np[:, 2].min()
                 ]
    max_point = [input_pcd_np[:, 0].max(),
                 input_pcd_np[:, 1].max(),
                 input_pcd_np[:, 2].max()
                 ]
    return [min_point, max_point]


# 在下采样的上面划线
def draw_line_down(source_down, target_down, inlier_line_set, outlier_linr_set,
                   axis_mesh=None, win_name='ANTenna3D'):
    if axis_mesh is None:
        o3d.visualization.draw_geometries([
            source_down,  # 重合的原因：source是直接抠出来的
            target_down,
            inlier_line_set,
            outlier_linr_set
        ],
            window_name=win_name,
            # zoom=1,
            # front=[0, -1, 0.1],  # 相机位置
            # lookat=[0, 0, 0],  # 对准的点
            # up=[0, 1, 0.1],  # 用于确定相机右x轴
            # point_show_normal=True
        )

    else:
        o3d.visualization.draw_geometries([
            source_down,  # 重合的原因：source是直接抠出来的
            target_down,
            inlier_line_set,
            outlier_linr_set,
            axis_mesh
        ],
            window_name='ANTenna3D',
            # zoom=1,
            # front=[0, -1, 0.1],  # 相机位置
            # lookat=[0, 0, 0],  # 对准的点
            # up=[0, 1, 0.1],  # 用于确定相机右x轴
            # point_show_normal=True
        )


def show_pcd(pcd_in, win_name='ANTenna3D'):
    o3d.visualization.draw_geometries([pcd_in], window_name=win_name)


def show_batch_pcd(pcd_in, bbox_gt_list=[], bbox_pred_list=[], win_name='ANTenna3D'):
    if type(pcd_in) != type([]):
        pcd_in = [pcd_in]
    # 最后列表直接相加就可以了
    for i in [bbox_gt_list, bbox_pred_list]:
        if len(i) != 0:
            pcd_in += i
    o3d.visualization.draw_geometries(pcd_in, window_name=win_name)
