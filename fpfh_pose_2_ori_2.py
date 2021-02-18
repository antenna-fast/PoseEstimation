from o3d_impl import *
import cv2
from o3d_pose_lib import *

from ransac import *

import time
from joblib import dump, load  # 保存模型

# ML lib
from sklearn.cluster import DBSCAN


# 预处理: 下采样，计算FPFH
def preprocess_point_cloud(pcd, voxel_size):
    # print(":: Downsample with a voxel size %.3f." % voxel_size)
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


# 要不要再拆开，后面再看需求
# 先把整个pipeline完成
def get_fpfh(pcd_in, voxel_size):

    radius_normal = voxel_size * 2
    pcd_in.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    # print("Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_in,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    return pcd_fpfh


# 加载数据
# 读取场景和目标，下采样，计算FPFH

def prepare_dataset(model_path, scene_path, voxel_size):
    print(":: Load two point clouds and disturb initial pose.")
    source = o3d.io.read_point_cloud(model_path)
    target = o3d.io.read_point_cloud(scene_path)

    # trans_init = np.asarray([[1.0, 0.0, 0.0, 5.0],
    #                          [0.0, 1.0, 0.0, -18.0],
    #                          [0.0, 0.0, 1.0, 0.0],
    #                          [0.0, 0.0, 0.0, 1.0]])
    # source.transform(trans_init)

    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    return source, target, source_down, target_down, source_fpfh, target_fpfh


# FLANN  进行特征匹配，找到关键点对应
FLANN_INDEX_LSH = 5
index_params = dict(algorithm=FLANN_INDEX_LSH,
                    table_number=33,  # 12
                    key_size=20,  # 20
                    multi_probe_level=1)  # 2
search_params = dict(checks=50)  # or pass empty dictionary  这是第二个字典，指定了索引里的树应该被递归遍历的次数

flann = cv2.FlannBasedMatcher(index_params, search_params)

class_map = {0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1], 3: [1, 1, 0], 4: [0, 1, 1], 5: [1, 0, 1]}


# 输入分类结果
# 输出类别
def get_class_res(res_in):
    bin_count = bincount(res_in.astype(int))
    class_max = max(bin_count)
    class_res = where(bin_count == class_max)[0][0]  # 每次只返回一类
    return class_res


if __name__ == '__main__':

    # 加载分类器模型
    clf = load('model_lib/adaboost.clf')

    root_path = 'D:/SIA/Dataset/'
    # source是模型
    # target是场景
    # 虚拟数据
    # scene_path = root_path + 'Stanford/RandomScene1/Scene0.ply'  # 场景有很多！
    # scene_path = root_path + 'Stanford/RandomScene1/Scene1.ply'
    scene_path = root_path + 'Stanford/RandomScene1/Scene2.ply'

    # model_path = root_path + 'Stanford/3D models/buddha/happy_vrip_res3.ply'
    # model_path = root_path + 'Stanford/3D models/armadillo/Armadillo_vres2_small_scaled.ply'
    model_path = root_path + 'Stanford/3D models/bunny/bun_zipper.ply'

    # 真实场景
    # scene_path = root_path + 'CVLab/2009-10-27/Scene1.ply'
    # model_path = root_path + 'CVLab/2009-10-27/model1.ply'

    # scene_path = root_path + 'CVLab/2010-06-12/Scene1/Scene1.ply'  # 可乐瓶 kinect
    # model_path = root_path + 'CVLab/2010-06-12/Scene1/model1.ply'

    # scene_path = root_path + 'CVLab/2010-03-03/Scena1/scene1.ply'  # 小人  kinect
    # model_path = root_path + 'CVLab/2010-03-03/Scena1/model1.ply'

    is_rgb = 0

    # voxel_size = 0.155  # means 5cm for the dataset
    voxel_size = 0.005  # means 5cm for the dataset
    # voxel_size = 0.4  # means 5cm for the dataset
    # voxel_size = 0.21  # means 5cm for the dataset

    threshold_n = 100  # 每个点集最少的点数

    s_time = time.time()  # 从什么时候开始计时？

    source, target, source_down, target_down, source_fpfh, target_fpfh = \
        prepare_dataset(model_path, scene_path, voxel_size)

    if not is_rgb:
        target_down.paint_uniform_color([0.4, 0.4, 0.4])  # 否则无法添加颜色
        source_down.paint_uniform_color([0.4, 0.4, 0.4])
    # 可视化原始数据
    # draw_registration_result(source, target, np.identity(4), is_rgb)
    # draw_registration_result(source_down, target_down, np.identity(4), is_rgb)  # 采样 特征提取后的数据

    # 转换成np
    model_np = array(source_down.points)  # 模型点
    scene_np = array(target_down.points)  # 场景点

    # 场景分割
    # Compute DBSCAN

    db = DBSCAN(eps=0.0054, min_samples=5).fit(scene_np)

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    unique_labels = set(labels)  # 列表变成没有重复元素的集合

    print('labels:', labels)
    print('unique_labels:', unique_labels)

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)  # 噪声点数

    unique_labels = list(unique_labels)[:5]

    # 对不同的类别添加不同的颜色
    for k in unique_labels:  # 通过匹配这些label，找到mask  k并不能用于类别区分
        if k == -1:  # Black used for noise.
            col = [0, 0, 0, 1]
            continue
        # print('k:', k)

        class_member_mask = where(labels == k)[0]  # where把类别掩膜转换成True的位置 方便索引
        # print('class_member_mask:', len(class_member_mask))

        if len(class_member_mask) < threshold_n:  # 通过点的个数进行滤波
            continue

        # 点云分类 提取所有点的特征点，放到分类器里面
        seg_pcd = target_down.select_by_index(class_member_mask)  # true false的不行
        print('seg_pcd:', seg_pcd)  # 这里就已经不对了

        # 分割后通过点的个数进行滤波

        seg_fpfh = get_fpfh(seg_pcd, voxel_size)  # 输入分割后的点云，输出FPFH

        seg_fpfh_T = seg_fpfh.data.T
        # print('seg_fpfh:', seg_fpfh_T.shape)

        res = clf.predict(seg_fpfh_T)
        # print('res:', res)

        # 统计res里面最多的元素，作为类别输出
        class_res = get_class_res(res)  # 可以再看看置信度：最多的类别占全部的多少？
        print('class_res:', class_res)

        # 以下找到当前索引的特征对应的哪个类别  接下来就是加载对应类别的特征 对应起来
        col = class_map[class_res]  # 不同类别的颜色  应当加上分类器之后  把k换掉
        asarray(target_down.colors)[class_member_mask, :] = col

        # 找到当前类别的特征向量(们)，然后和随影类别的模型上面的特征向量进行匹配
        # load_model_feature()  # 输入类别，输出模型上所有的特征

        # 特征匹配

    # 可视化基于DBSCAN分割之后的结果
    is_rgb = 1
    # draw_registration_result(source_down, target_down, eye(4), is_rgb)
    draw_registration_result(target_down, target_down, eye(4), is_rgb)

    # print(source_fpfh.data[0])
    source_fpfh_T = float32(source_fpfh.data.T)
    target_fpfh_T = float32(target_fpfh.data.T)
    # source_fpfh_T = uint8(source_fpfh.data.T)  # 模型
    # target_fpfh_T = uint8(target_fpfh.data.T)  # 场景
    print('source_fpfh:\n', source_fpfh_T.shape)
    print('target_fpfh:\n', target_fpfh_T.shape)

    # 特征分类，找到对应类别的之后再各自进行匹配  ********** ours work

    # 只拿出某一个类别的，进行匹配，看能不能降低错误率  优势在于多类别！

    # 特征匹配  现在的思路是从模型往回找
    # 等分割的时候，就是从场景往模型找
    # （因为事先分类好了，对于重合模型的情况，聚类就好了
    # 具体来讲，如果差别不超过模型的尺寸半径就聚集在一起）
    # queryDescriptors, trainDescriptors, k, mask, compactResult
    # matches = flann.knnMatch(target_fpfh_T, source_fpfh_T, k=2)  # 场景 模型 近邻个数
    matches = flann.knnMatch(source_fpfh_T, target_fpfh_T, k=2)  # 场景 模型 近邻个数

    # print(matches)
    match_model_idx = []
    match_scene_idx = []

    for (m, n) in matches:  # m是最小距离  n是次小距离（或者一会加上过滤）
        # print(m.distance)
        if m.distance < 0.8 * n.distance:  # 什么原理？ 最小的<0.7次小的
            queryIdx = m.queryIdx  # 模型上
            trainIdx = m.trainIdx  # 场景中

            match_scene_idx.append(trainIdx)  # 场景中 stack起来
            match_model_idx.append(queryIdx)  # 模型上

            asarray(target_down.colors)[trainIdx, :] = [1, 0, 0]  # 场景
            asarray(source_down.colors)[queryIdx, :] = [0, 1, 0]  # 模型

    e_time = time.time()
    print('match_time:', e_time - s_time)

    # 匹配点索引
    match_model_idx = array(match_model_idx)
    match_scene_idx = array(match_scene_idx)  # 场景点索引
    # print('match_model_idx:', match_model_idx)
    # print('match_scene_idx:', match_scene_idx)
    # print('match_scene_idx:', len(match_scene_idx))
    # print('match_scene_unique_idx:', len(unique(match_scene_idx)))

    # 匹配的点
    model_paired = model_np[match_model_idx]
    scene_paired = scene_np[match_scene_idx]

    ##################
    # 根据匹配点进行位姿估计  3D-3D  RANSAC构造模型的核还需要改
    res_ransac, _, _, _ = ransac_pose(model_paired, scene_paired, 50)
    print('res_ransac:\n', res_ransac)
    draw_registration_result(source, target, res_ransac, is_rgb)  # 粗配准后
    # 问题是 如果本来就重合，但是特征配准错了，就会偏移! 比如CVLab的Kinect数据集

    # 为了可视化
    # print('scene_vtx:', len(scene_vtx))
    model_vtx_num = len(model_np)
    print('model_vtx_num:', model_vtx_num)  # md 纠结好久，原来是场景家的不对！索引不是顶点的
    line_vtx = r_[model_np, scene_np]  # 顶点坐标stack起来
    line_idx = c_[match_model_idx, match_scene_idx + model_vtx_num]  # idx的场景那一半也得加上对应的长度
    print('line_vtx:', line_vtx.shape)
    print('line_idx:', line_idx.shape)

    color = array([0, 0.3, 1])
    colors = tile(color, (len(line_idx), 1))  # 复制数组
    # print(colors)

    line_set = draw_line(line_vtx, line_idx, colors)  # 对应点的匹配线  # points lines color
    # line_set = draw_line(ab, idx, colors)  # 对应点的匹配线  # points lines color 颜色个数=点对个数

    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # # 将两个点云放入visualizer
    # vis.add_geometry(source)
    # # vis.add_geometry(target)
    # vis.get_render_option().point_size = 2  # 点云大小
    # vis.get_render_option().background_color = np.asarray([0, 0, 0])  # 背景颜色

    o3d.visualization.draw_geometries([
        source_down,  # 重合的原因：source是直接抠出来的
        target_down,
        line_set
    ],
        window_name='ANTenna3D',
    )

    # 根据FPFH特征 粗配准
    # result_ransac = execute_global_registration(source_down, target_down,
    #                                             source_fpfh, target_fpfh,
    #                                             voxel_size)
    #
    # result_ransac = execute_fast_global_registration(source_down, target_down,
    #                                                  source_fpfh, target_fpfh,
    #                                                  voxel_size)
    #
    # print('result_ransac:\n', result_ransac)
    # draw_registration_result(source_down, target_down, result_ransac.transformation)
    #
    # # 执行ICP精细配准
    # result_icp = refine_registration(source, target, result_ransac,
    #                                  voxel_size)
    #
    # print('result_icp:\n', result_icp)
    # # print('result_icp:', dir(result_icp))
    # # print('result_icp:', result_icp.transformation)
    # draw_registration_result(source, target, result_icp.transformation)
