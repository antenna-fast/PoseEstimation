# RANSAC LIBRARY FOR COMPUTER VISION
# Powered by ANTenna Vision

import os
import sys
import time
import numpy as np
import numpy.linalg as linalg
from sklearn.neighbors import NearestNeighbors  # KNN
import random as rsample  # 产生随机数

from copy import deepcopy
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd() + '..')
from utils.icp_using_svd import icp_refine
# from icp_using_svd import icp_refine


def line_estimator(p1, p2):
    # line model estimation
    k = (p2[1] - p1[1]) / (p2[0] - p1[0])
    b = p1[1] - k * p1[0]
    return k, b


def plane_estimator():
    return 0


def get_pt2line_dist(pt, line_k, line_b):
    a, b, c = line_k, -1, line_b
    dist = np.abs(a * pt[0] + b * pt[1] + c) / np.sqrt(a ** 2 + b ** 2)
    return dist


# circle estimator
def circle_estimator(pts):
    # 返回模型:至少三个点确定一个圆
    # 参数:圆心 半径
    return 0


# sample_function  用于随机采样,得到不重复的n个索引
def sample_n(idx_list, n):
    res = rsample.sample(idx_list, n)
    return res


# 输入包含噪声的直线数据
# 输出直线参数, 内点, 外点
# 这里提供一个框架,换成其他的模型函数即可拟合其他的数据
def ransac_line(pts, iter_num, inlier_threshold=0.5):
    # ransan 改进：记录变化趋势，减少随机性
    # 1.从所有的数据中进行随机采样，得到一组足以建模的一组数据
    # 2.判断模型质量
    # 重复上述步骤,最后选出置信度最高的

    # pts: 所有数据点
    # iter_num: 迭代次数

    data_len = len(pts)
    idx_list = list(range(data_len))

    model_buff = []  # 保存k, b, dist
    while iter_num > 0:
        iter_num -= 1
        # 1. sample
        random_idx = sample_n(idx_list, 2)
        pt1, pt2 = pts[random_idx[0]], pts[random_idx[1]]
        # 2. line estimator kernel
        k, b = line_estimator(pt2, pt1)
        # 3. model evaluation 将[x, y]代进去，如果点到直线的距离小于某个数值，该模型就得分
        dist = 0  # init dist
        for pt in pts:  # all point!  这里可以直接写成矩阵型形式
            dist_i = get_pt2line_dist(pt, k, b)  # 如果要拟合其他的模型,更换这个即可
            dist += dist_i  # 模型的总体性能
        model_buff.append([k, b, dist])  # 缓存模型以及评分
    model_buff = np.array(model_buff)  # iter_num x 3

    # finally we find min_dist from all model buffer
    idx = np.argmin(model_buff[:, 2])  # 取置信度最高的模型参数
    model_param = model_buff[idx]  # in [k, b]
    # print('model_param:', model_param)

    # 下面输出参数作为可选的(耗时).因为不一定所有任务都需要
    confidence = 0  # 模型置信度
    outlier_buff = []  # a set in {nx2}
    # 内外点阈值
    # 通过最佳模型去除外点
    for pt_idx in idx_list:  # all point!  这里可以直接写成矩阵型形式  Ax=b
        pt = pts[pt_idx]
        dist_i = get_pt2line_dist(pt, model_param[0], model_param[1])  # 如果要拟合其他的模型,更换这个即可

        if dist_i > inlier_threshold:  # 假设局外点更少，所以使用局外点减少次数
            outlier_buff.append(pt_idx)
    inlier_buff = list(set(idx_list) - set(outlier_buff))
    inlier_ratio = len(inlier_buff) / (len(inlier_buff) + len(outlier_buff))
    return model_param, confidence, inlier_buff, outlier_buff, inlier_ratio


def ransac_plane():
    return 0


# 位姿模型，已知点的对应关系。因此采样不是特别随机的
# 问题分析：至少几个点确定一个变换（系统模型）一般4个
# 模型检验：将变换应用到场景和目标，用度量标准度量

# 输入匹配的点坐标
# 返回source在target中的位姿
# 目前直接使用的ICP，尚没有经过迭代和离群点剔除。如何应对模型上就有噪声？
# 迭代一次后，基本能找到离群点
def ransac_pose(source, target, iter_num, inlier_threshold):
    # print('performing RANSAC pose estimation ... ')
    # ransan 随机采样一致性算法，改进：记录变化趋势，减少随机性
    # 1.从所有的数据中进行随机采样，得到一组足以建模的一组数据
    # 2.判断模型质量
    # 重复上述步骤,最后选出置信度最高的

    # source: 模型  注意 这是匹配好的
    # target: 场景
    # iter_num: 迭代次数
    # inlier_threshold: 度量距离小于阈值的视为内点

    num_sample_points = 5

    if source.shape != target.shape:
        raise ValueError('ERROR INPUT DATA SHAPE: source:{}, target:{}'.format(source.shape, target.shape))

    if len(source) < num_sample_points:
        raise ValueError('ERROR DATA LENGTH: len(match)={} < #sample={}'.format(len(source), num_sample_points))

    source_deep = deepcopy(source)  # deepcopy to avoid change original data!
    data_len = len(source_deep)
    idx_list = list(range(data_len))  # 创建索引随机采样的列表

    # KNN 搜索最近点用于性能度量
    # 使用模型点与场景点的最近距离作为最终结果的度量标准
    # 待查询数据(变换完的模型)
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(target)  # 目标不变

    model_buff = []  # model parameter, dist
    score_buff = []  # evaluation buffer

    while iter_num > 0:
        iter_num -= 1

        # 1.sample 得到随机不重合的点对
        random_idx = sample_n(idx_list, num_sample_points)  # 每次随机采样几个
        sub_source = source_deep[random_idx]  # 取出匹配的子集
        sub_target = target[random_idx]

        # 2. model estimation
        W = np.eye(len(sub_source))

        # ICP这里不再设置过多的中间参数
        # for _ in range(1):  # ICP迭代次数(没有更新匹配的点，则迭代没有意义，但是实际上我使用了上次的作为初值)
        # ICP kernel, 假设点是已经匹配好的，通过最小二乘拟合出一个变换
        res_mat = icp_refine(sub_source, sub_target, W)  # 模型 场景 加权

        r_mat_res = res_mat[:3, :3]
        t_res = res_mat[:3, -1].reshape(-1)

        # 3. evaluation the estimated model, 将全部点云代进去，看哪个位姿的误差小
        # 改进：变成求最近距离的：模型上求到场景的最近距离，然后求和：根据距离（作为特征进行）匹配？？
        source_deep_trans = np.dot(r_mat_res, source_deep.T).T + t_res  # 根据最小二乘结果模型进行变换

        # 使用模型点与场景点的最近距离作为最终结果的度量标准
        distances, indices = nbrs.kneighbors(source_deep_trans)
        dist = sum(distances.T[1])

        model_buff.append(res_mat)  # 缓存模型以及评分
        score_buff.append(dist)
    model_buff = np.array(model_buff)  # iter-num x 3
    score_buff = np.array(score_buff)  # iter-num x 1

    # finally we find the min_dist from all model buffer
    idx = np.argmin(score_buff)  # 取置信度最高的模型参数
    model_param = model_buff[idx]  # in [k, b]

    # check model, and remove outlier
    r_mat_res = model_param[:3, :3]
    t_res = model_param[:3, 3:].reshape(-1)

    source_trans = np.dot(r_mat_res, source_deep.T).T + t_res  # 模型变换到场景上
    distances, indices = nbrs.kneighbors(source_trans)
    dist = linalg.norm(source_trans - target, axis=1) + distances.T[1]
    # 这样是对应变换后：由于本来的匹配错误，这样如果即使是正确的匹配也会被误认为是错误匹配
    # 要提高正确匹配的召回率，就得用KNN的方法对实际的数据进行操作

    # inlier and outlier determination
    # we select the one whose distance are higher than a set inlier threshold as outlier
    # lower threshold will produce more outlier
    outlier_buff = np.where(dist > inlier_threshold)[0]
    inlier_buff = np.array(list(set(idx_list) - set(outlier_buff))).astype(int)
    inlier_ratio = len(inlier_buff) / (len(inlier_buff) + len(outlier_buff))
    dist = np.sum(dist)  # overall estimated model's performance
    return model_param, dist, inlier_buff, outlier_buff, inlier_ratio


# test space
def gen_fake_line(k=2, b=3):
    # define a line model: y=k*x + b
    x = np.arange(0, 10, 0.1)
    y = k * x + b + np.random.rand(len(x))  # ok
    pts = np.array([x, y]).T

    # generate some noise
    mean = np.array([10, 15])  # 正态分布
    cov = np.eye(2) * 3
    noise1 = np.random.multivariate_normal(mean, cov, 15)

    mean = np.array([0, 15])  # 正态分布
    cov = np.eye(2) * 5
    noise2 = np.random.multivariate_normal(mean, cov, 25)
    # noise2 = np.random.uniform([0, 4], [10, 14], [10, 2])  # 均匀分布

    # add noise to our line dataset
    pts = np.vstack((pts, noise1))
    pts = np.vstack((pts, noise2))
    print('pts_shape:', pts.shape)

    return pts


if __name__ == '__main__':
    print('line detection demo using ransac ')
    # generate toy fake data
    pts = gen_fake_line(k=2, b=3)

    # ransac parameters
    iter_num = 30

    # estimate a line model
    s_time = time.time()
    model_para, dist, inlier, outlier, inlier_ratio = ransac_line(pts, iter_num)
    e_time = time.time()
    print('time cost: {0:.4f} s'.format(e_time - s_time))
    print('inlier ratio: {:.4f}'.format(inlier_ratio))

    # visualization
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # ax.scatter(pts.T[0], pts.T[1])  # 原始数据
    ax.scatter(pts[inlier].T[0], pts[inlier].T[1], color='g', label='inlier')  # 区分内外点的
    ax.scatter(pts[outlier].T[0], pts[outlier].T[1], color='b', label='outlier')  # 区分内外点的

    # generate draw data
    x_test = np.array([-1, 10])
    y_test = model_para[0] * x_test + model_para[1]

    ax.plot(x_test, y_test, color='r')
    ax.plot(x_test, y_test, color='r')
    plt.legend()
    plt.show()
