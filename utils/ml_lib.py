# machine learning lib

from sklearn.cluster import DBSCAN
import numpy as np


# dbscan based object segmentation
def dbscan_segment(scene_down_np, search_radius=0.005, min_sample=5, threshold_n=50, class_num=5):
    # 场景分割
    # 输入：待分割的点云
    # 返回：分割后点云簇的索引

    # search_radius  # 聚类搜索半径
    # min_sample  # 半径内的包含的点
    # threshold_n  # 每个点集最少的点数
    # class_num  # 5

    # Compute DBSCAN
    db = DBSCAN(eps=search_radius, min_samples=min_sample).fit(scene_down_np)

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    unique_labels = set(labels)  # 列表变成没有重复元素的集合
    # print('unique_labels:', unique_labels)

    # Number of clusters in labels, ignoring noise if present.
    # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # n_noise_ = list(labels).count(-1)  # 噪声点数
    unique_labels = list(unique_labels)[:class_num]  # 只取出前多少类 (后面的是符合分割标准但不是想要的)

    # 返回值缓存器
    seg_idx_buff = []

    # 对不同的类别添加不同的颜色  实现类别级分割
    for k in unique_labels:  # 通过匹配这些label，找到mask  k并不能用于类别区分
        if k == -1:  # Black used for noise.
            continue
        class_member_mask = np.where(labels == k)[0]
        # print('class_member_mask:', len(class_member_mask))

        # 分割后通过点的个数进行滤波  设置阈值适应不同的遮挡
        if len(class_member_mask) < threshold_n:  # 通过点的个数进行滤波
            continue
        # 添加进来的个数就是所有分割结果的个数
        # 索引参考对象是输入点云
        seg_idx_buff.append(class_member_mask)
    return seg_idx_buff


# point cloud patch classification
def get_class_res(res_in):
    # 根据所有的结果 找多类别
    # 输入: point-wise classification
    # 输出: 类别
    res_num = len(res_in)  # 所有的预测结果
    bin_count = np.bincount(res_in.astype(int))  # 按照索引 第几个元素有几个
    class_max = max(bin_count)
    class_res = np.where(bin_count == class_max)[0][0]  # 最大元素所在索引 每次只返回最有可能的类
    score = class_max / res_num  # 置信度，预测最可能的结果的可能性
    return int(class_res), score


def get_class_res_static(res_in):
    # 根据所有的结果 找多类别
    # 输入: 分类结果
    # 输出: 类别
    res_num = len(res_in)  # 所有的预测结果
    bin_count = np.bincount(res_in.astype(int))  # 按照索引 第几个元素有几个
    score_list = bin_count / res_num
    return score_list
