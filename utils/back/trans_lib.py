import numpy as np
from numpy.linalg import *
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt


def get_coord(now_pt, vici_pts):
    # 根据近邻点 求出平面坐标系  PCA局部坐标系拟合
    # average_data = np.mean(vici_pts, axis=0)  # 求 NX3 向量的均值
    # decentration_matrix = vici_pts - average_data  # 邻域点连接到顶点的向量
    decentration_matrix = vici_pts - now_pt  # 邻域点连接到顶点的向量
    H = np.dot(decentration_matrix.T, decentration_matrix)  # 求解协方差矩阵 H

    U, s, Vh = svd(H)  # U.shape, s.shape, Vh.shape

    # 排序索引 由小到大
    # x_axis, y_axis, z_axis = Vh  # 由大到小排的 对应xyz轴
    return Vh.T  # 以列向量表示三个坐标轴


if __name__ == '__main__':
    # 定义平面方程
    p = np.array([1, 1, 1, -10])  # z如果相对于点的长宽太小 看上是过原点
    p_n = p[:3]
    # 注意 平面要按照方程生成
    pts_buff = []
    for i in range(-10, 10):
        for j in range(-10, 10):
            z = -1 * (p[3] + p[0] * i + p[1] * j) / p[2]  # 问题:如果c=0 就不行了
            pts_buff.append([i, j, z])
            # pts_buff.append([i, j, abs(z)])

    pts_buff = np.array(pts_buff)
    pts = pts_buff

    # ax = plt.figure(1).gca(projection='3d')
    #
    # ax.plot(pts_buff.T[0], pts_buff.T[1], pts_buff.T[2], 'g.')
    # ax.set_xlabel("X Axis")
    # ax.set_ylabel("Y Axis")
    # ax.set_zlabel("Z Axis")
    # plt.title('point cloud')
    # plt.show()

    # print(pts_2d)  # 对这个坐标三角化  三角化后得到三角形的顶点索引,根据这个索引从点云中检索,得到三维mesh的索引,构建mesh,求法向量

    # now_pt, vici_pts
    coord = get_coord(pts_buff[0], pts_buff)  # 平面的旋转坐标变换

    coord_inv = inv(coord)  # 反变换
    roto_pts = np.dot(coord_inv, pts_buff.T).T  # 将平面旋转到与z平行
    pts_buff = roto_pts

    pts_buff[:, 2] = 0  # 已经投影到xoy(最大平面),在此消除z向轻微抖动

    pts_2d = pts_buff[:, 0:2]

    # 三角化 找到拓扑结构
    tri = Delaunay(pts_2d)
    # print(dir(tri))
    # print(tri.points)
    # print(len(tri.points))
    tri_idx = tri.simplices
    # print(tri_idx)  # 三角形索引

    plt.triplot(pts_2d[:, 0], pts_2d[:, 1], tri.simplices.copy())
    plt.plot(pts_2d[:, 0], pts_2d[:, 1], 'o')
    plt.show()

    # 此处的输入点应当是原始点云中的
    # mesh = get_non_manifold_vertex_mesh(pts, tri_idx)
    # mesh.compute_triangle_normals()

    # print(dir(mesh.triangle_normals))
    # print(array(mesh.triangle_normals))

    ax = plt.figure(1).gca(projection='3d')

    # a = pts_buff
    ax.plot(pts_buff.T[0], pts_buff.T[1], pts_buff.T[2], 'g.')
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    plt.title('point cloud')
    plt.show()
