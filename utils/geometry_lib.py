"""
Author: ANTenna on 2021/12/23 2:44 下午
aliuyaohua@gmail.com

Description:
geometry library for computer vision
"""

import numpy as np
from numpy import linalg


# POINT

# 投影点[XYZ] 投影线[ABC]
def pt_to_line(p, l):
    # 求参数t
    l_n = np.array([l[0], l[1]])
    t = -(l[2] + l[0] * p[0] + l[1] * p[1]) / (l[0] * l_n[0] + l[1] * l_n[1])
    # print('t:', t)
    x = p + t * l_n
    # print(x)
    return x


# 输入:点(nx3) 平面 平面法向量
# 输出:投影点
def pt_to_plane(pts_array, plan, p_n):
    # 参数t
    res = []
    for pts in pts_array:  # 这里可以矩阵化  加速计算
        t = -1 * (plan[0] * pts[0] + plan[1] * pts[1] + plan[2] * pts[2] + plan[3]) / (plan[0] * p_n[0] + plan[1] * p_n[1] + plan[2] * p_n[2])
        x = pts + t * p_n
        res.append(x)
    return np.array(res)


# PLANE
# 从法向量和一点得到平面
def get_plan(normal, pt):
    D = -1 * np.dot(normal.T, pt)
    p = np.array([normal[0], normal[1], normal[2], D])
    return p


# COORDINATE
def get_coord(now_pt, vici_pts):
    # 根据近邻点 求出平面坐标系  PCA局部坐标系拟合
    # average_data = np.mean(vici_pts, axis=0)  # 求 NX3 向量的均值
    # decentration_matrix = vici_pts - average_data  # 邻域点连接到顶点的向量
    decentration_matrix = vici_pts - now_pt  # 邻域点连接到顶点的向量
    H = np.dot(decentration_matrix.T, decentration_matrix)  # 求解协方差矩阵 H
    U, s, Vh = linalg.svd(H)  # U.shape, s.shape, Vh.shape

    # 排序索引 由小到大
    # x_axis, y_axis, z_axis = Vh  # 由大到小排的 对应xyz轴
    return Vh.T  # 以列向量表示三个坐标轴


if __name__ == '__main__':
    # PLANE TEST
    n = np.array([1, 1, 1])
    pt = np.array([0, 0, 0])
    p = get_plan(n, pt)
    print(p)

