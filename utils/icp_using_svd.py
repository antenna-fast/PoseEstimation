import numpy as np
import numpy.linalg as linalg
from scipy.spatial.transform import Rotation as R
import open3d as o3d

import matplotlib.pyplot as plt
import time

# SVD实现ICP
# 基本原理：最小二乘
# 缺少最近点匹配过程

# 效率：目前 100个点1ms
# 使用方法：迭代，每次求导并更新


def show_pts(pts_1, pts_2):
    ax = plt.figure(1).gca(projection='3d')
    # ax = plt.figure(1).gca()

    ax.plot(pts_1.T[0], pts_1.T[1], pts_1.T[2], 'g.')
    ax.plot(pts_2.T[0], pts_2.T[1], pts_2.T[2], 'r.')
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")

    plt.title('point cloud')
    plt.show()


# 求出两组点云的加权平均值 也就是加权重心
def get_sum(W, X):
    weighted_sum = np.sum(np.dot(W, X), axis=0)  # 加权
    # weighted_sum = (dot(W, X))
    # w_vec_sum = sum(W.diagonal())
    w_vec_sum = np.sum(W)  # 权重和
    weighted_sum = weighted_sum / w_vec_sum  # 加权平均
    return weighted_sum


# pts_1 模型
# pts_2 场景
# 返回: 模型1在场景2中的位姿
def icp_refine(pts_1, pts_2, W):
    # reference:
    pts_m_1 = get_sum(W, pts_1)
    pts_m_2 = get_sum(W, pts_2)

    # 定义xi yi
    # 点云数据格式：nxd

    # 去平均值 （简化计算）
    Xi = pts_1 - pts_m_1
    Yi = pts_2 - pts_m_2

    # 构造S  Xi.T:dxn  W:nxn  Yi:nxd
    S = np.dot(np.dot(Xi.T, W), Yi)

    # 对S进行SVD分解
    U, s, Vh = linalg.svd(S)
    Vh = Vh.T  # T!!

    # 得到旋转矩阵R
    mid_mat = np.eye(3)
    mid_mat[2][2] = linalg.det(np.dot(Vh, U.T))
    r_mat_res = np.dot(np.dot(Vh, mid_mat), U.T)

    # 得到平移变换
    t_res = pts_m_2 - np.dot(r_mat_res, pts_m_1.T)

    res_mat = np.eye(4)

    res_mat[:3, :3] = r_mat_res
    res_mat[:3, 3:] = t_res.reshape(3, -1)  # 前三行 第四列

    return res_mat


if __name__ == '__main__':
    # 定义变换
    # r = R.from_rotvec(pi / 180 * array([60, 0, 10]))  # 角度->弧度
    r = R.from_rotvec(np.pi / 180 * np.array([0, 10, 0]))  # 角度->弧度
    r_mat = r.as_matrix()
    t_vect = np.array([10, -2, -8], dtype='float')
    # t_vect = array([1, 1, 0], dtype='float')
    print('r_mat_GT:\n', r_mat)

    # 定义被配准的点  模型
    pts_1 = []
    for i in range(10, 20):
        for j in range(10, 30):
            pts_1.append([i, j, 0])

    pts_1 = np.array(pts_1)

    # 定义变换后的点  场景
    pts_2 = np.dot(r_mat, pts_1.T).T + t_vect  # 先旋转再平移

    # 加噪声

    # 每对点的权重
    W = np.eye(len(pts_1))  # 假设每对点的权重都是1

    s_time = time.time()
    res = icp_refine(pts_1, pts_2, W)  # 模型 场景 对应点的权重

    r_mat_res = res[:3, :3]
    t_res = res[:3, 3:].reshape(-1)

    e_time = time.time()

    print('time_cost:', e_time - s_time)
    print('r_mat_res:\n', r_mat_res)  # 迭代时，变化已经很小
    print('t:\n', t_res)

    # 将模型变换到场景上
    pts_1 = np.dot(r_mat_res, pts_1.T).T + t_res  # 变换也是先旋转再平移

    show_pts(pts_1, pts_2)
