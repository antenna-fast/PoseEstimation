from o3d_impl import *
import cv2
from o3d_pose_lib import *

from ransac import *

from path_lib import *

import time
from joblib import dump, load  # 保存模型

# ML lib
from sklearn.cluster import DBSCAN


# 输入分类结果
# 输出类别
def get_class_res(res_in):
    bin_count = bincount(res_in.astype(int))
    class_max = max(bin_count)
    class_res = where(bin_count == class_max)[0][0]  # 每次只返回一类
    return class_res


def load_model_feature(class_idx):
    feature_path = feature_bag_root + class_encode[class_idx] + '.npy'
    # print('feature_path:', feature_path)
    model_fpfh_feature = np.load(feature_path)
    return model_fpfh_feature


# 通过类别加载对应的模型，并进行采样等等

# FLANN  进行特征匹配，找到关键点对应
FLANN_INDEX_LSH = 5
index_params = dict(algorithm=FLANN_INDEX_LSH,
                    table_number=33,  # 12
                    key_size=20,  # 20
                    multi_probe_level=1)  # 2
search_params = dict(checks=50)  # or pass empty dictionary  这是第二个字典，指定了索引里的树应该被递归遍历的次数

flann = cv2.FlannBasedMatcher(index_params, search_params)

class_encode = ['armadillo', 'buddha', 'bunny', 'chinese_dragon', 'dragon', 'statuette']
color_map = {0: [1, 0, 0], 1: [0.2, 1, 0], 2: [0.1, 0.8, 0.5], 3: [1, 1, 0], 4: [0.2, 0.8, 1], 5: [1, 0, 1]}

if __name__ == '__main__':

    model_root_stanford = 'D:/SIA/Dataset/Stanford/3D models/'

    # 输入索引，输出模型的fpfh  * 根据索引保存
    feature_bag_root = 'D:/SIA/Dataset/FeatureBag/StanfordFPFH/'

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
    # model_path = root_path + 'Stanford/3D models/bunny/bun_zipper.ply'
    model_path = root_path + 'Stanford/3D models/dragon/dragon_vrip_res2.ply'

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

    # source, target, source_down, target_down, source_fpfh, target_fpfh = \
    #     prepare_dataset(model_path, scene_path, voxel_size)

    # 这里不必要求出场景的fpfh，加速点*
    target, target_down, target_fpfh = prepare_dataset(scene_path, voxel_size)

    if not is_rgb:
        target_down.paint_uniform_color([0.4, 0.4, 0.4])  # 否则无法添加颜色
        # source_down.paint_uniform_color([0.4, 0.4, 0.4])  # 暂时不加载模型

    # 可视化原始数据
    # draw_registration_result(source, target, np.identity(4), is_rgb)
    # draw_registration_result(source_down, target_down, np.identity(4), is_rgb)  # 采样 特征提取后的数据

    # 转换成np
    # model_np = array(source_down.points)  # 模型点
    scene_down_np = array(target_down.points)  # 场景点

    # 场景分割
    # Compute DBSCAN
    db = DBSCAN(eps=0.0054, min_samples=5).fit(scene_down_np)

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

    # 对不同的类别添加不同的颜色  实现类别级分割
    for k in unique_labels:  # 通过匹配这些label，找到mask  k并不能用于类别区分
        if k == -1:  # Black used for noise.
            col = [0, 0, 0, 1]
            continue
        # print('k:', k)

        class_member_mask = where(labels == k)[0]  # where把类别掩膜转换成True的位置 方便索引
        # print('class_member_mask:', len(class_member_mask))

        # 分割后通过点的个数进行滤波
        if len(class_member_mask) < threshold_n:  # 通过点的个数进行滤波
            continue

        # 点云分类 提取所有点的特征点，放到分类器里面
        seg_target = target_down.select_by_index(class_member_mask)  # true false的不行
        # print('seg_pcd:', seg_pcd)  # 这里就已经不对了  已解决，不能用 TrueFalse索引

        seg_target_fpfh_T = get_fpfh(seg_target, voxel_size)  # 输入分割后的点云，输出FPFH
        # print('seg_target_fpfh:', seg_target_fpfh_T.shape)

        res = clf.predict(seg_target_fpfh_T)  # 分类器预测
        # print('res:', res)

        # 统计res里面最多的元素，计算类别并输出
        class_res = get_class_res(res)  # 可以再看看置信度：最多的类别占全部的多少？
        print('class_res:', class_res)

        # 以下找到当前索引的特征对应的哪个类别  接下来就是加载对应类别的特征 对应起来
        col = color_map[class_res]  # 不同类别的颜色  应当加上分类器之后  把k换掉
        asarray(target_down.colors)[class_member_mask, :] = col
        # seg_pcd_np = array(target_down.points)[class_member_mask, :]

        # 加载对应类别的模型
        # 根据类别找到对应的模型位置
        m_path = model_root_stanford + class_encode[class_res] + '/'
        model_path = m_path + get_file_list(m_path)[0]
        # print('model_path:', model_path)

        model_trans_init = eye(4)  # 初始化模型位姿，更好地可视化
        model_trans_init[:3, 3:] = array([1.5, 0.5, 1]).reshape(3, -1) * 0.2
        source, source_down = load_model(model_path, model_trans_init, voxel_size)

        source_down.paint_uniform_color([0.4, 0.4, 0.4])  # 否则没有颜色
        model_down_np = array(source_down.points)  # 模型点

        # 找到当前类别的特征向量(们)，然后和随影类别的模型上面的特征向量进行匹配
        model_feature = load_model_feature(class_idx=class_res)  # 输入类别，输出模型上所有的特征  这些特征的索引就是对应点的索引

        # 格式转换 否则flann不支持
        model_feature = float32(model_feature)
        # target_feature_seg = float32(seg_target_fpfh_T)
        target_feature = float32(target_fpfh.data.T)  # 不能用分割之后的！ 要不然哪有什么区别？？

        # 可以替代上面的一堆  为了迎合o3d的注册函数  为了区别，这里用shource表示模型
        # _, _, source_fpfh = prepare_dataset(model_path, voxel_size)

        # 特征匹配，得到匹配的点  输入模型特征和当前分割后的特征，输出分割后对应的索引
        matches = flann.knnMatch(model_feature, target_feature, k=2)  # 场景 模型 近邻个数

        match_model_idx = []
        match_scene_idx = []

        for (m, n) in matches:  # m是最小距离  n是次小距离（或者一会加上过滤）
            # print(m.distance)
            if m.distance < 0.8 * n.distance:  # 什么原理？ 最小的<0.7次小的
                queryIdx = m.queryIdx  # 模型上
                trainIdx = m.trainIdx  # 场景中

                match_model_idx.append(queryIdx)  # 模型上
                match_scene_idx.append(trainIdx)  # 场景中 target_down的索引！

        match_model_idx = array(match_model_idx)
        match_scene_idx = array(match_scene_idx)  # 场景点索引

        # 匹配的对应点添加颜色
        asarray(source_down.colors)[match_model_idx, :] = [0, 1, 0]  # 模型
        asarray(target_down.colors)[match_scene_idx, :] = [1, 0, 0]  # 场景 对应点着色

        # 根据索引找到对应的三维坐标  匹配的坐标点 用于位姿估计
        model_paired = model_down_np[match_model_idx]
        scene_paired = scene_down_np[match_scene_idx]  # 此时，match不改变索引

        # 进行粗位姿估计 (毕竟就算出去也得循环)
        # 根据匹配点进行位姿估计  3D-3D  RANSAC构造模型的核还需要改
        res_ransac, _, _, _ = ransac_pose(model_paired, scene_paired, 10)

        # res_ransac = execute_global_registration(source_down, seg_target,
        #                             source_fpfh, target_fpfh, voxel_size)
        # # print('res_ransac:\n', res_ransac)
        # res_ransac = res_ransac.transformation  # 库函数的
        # print('res_ransac:\n', res_ransac)

        draw_registration_result(source, target, res_ransac, is_rgb, 'Global')  # 粗配准后

        # 位姿精细估计
        res_icp = refine_registration(source, target, res_ransac, voxel_size)
        draw_registration_result(source, target, res_icp.transformation, is_rgb, 'Global ICP')  # 粗配准后

        # 单物体的  （这里要根据具体的类别加载一下）
        axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

        line_set = get_line_set(model_down_np, scene_down_np, match_model_idx, match_scene_idx)
        draw_line_down(source_down, target_down, line_set, axis_pcd)

        # 缓存起来，方便后面进行可视化（加载所有模型 划线）

    # 跳出来之后，进行对应点的可视化，一看，呦，没有多少误匹配了！

    # 可视化基于DBSCAN分割之后的结果
    # is_rgb = 1
    # draw_segmentation_result(target_down, is_rgb)
    # draw_registration_result(target_down, target_down, eye(4), is_rgb)

    # print(source_fpfh.data[0])
    # source_fpfh_T = float32(source_fpfh.data.T)
    # target_fpfh_T = float32(target_fpfh.data.T)
    # source_fpfh_T = uint8(source_fpfh.data.T)  # 模型
    # target_fpfh_T = uint8(target_fpfh.data.T)  # 场景
    # print('source_fpfh:\n', source_fpfh_T.shape)
    # print('target_fpfh:\n', target_fpfh_T.shape)

    # 特征分类，找到对应类别的之后再各自进行匹配  ********** ours work

    # 只拿出某一个类别的，进行匹配，看能不能降低错误率  优势在于多类别！

    # 特征匹配  现在的思路是从模型往回找
    # 等分割的时候，就是从场景往模型找
    # （因为事先分类好了，对于重合模型的情况，聚类就好了
    # 具体来讲，如果差别不超过模型的尺寸半径就聚集在一起）
    # queryDescriptors, trainDescriptors, k, mask, compactResult
    # matches = flann.knnMatch(target_fpfh_T, source_fpfh_T, k=2)  # 场景 模型 近邻个数
    # matches = flann.knnMatch(source_fpfh_T, target_fpfh_T, k=2)  # 场景 模型 近邻个数
    # # print(matches)
    #
    # match_model_idx = []
    # match_scene_idx = []
    #
    # for (m, n) in matches:  # m是最小距离  n是次小距离（或者一会加上过滤）
    #     # print(m.distance)
    #     if m.distance < 0.8 * n.distance:  # 什么原理？ 最小的<0.7次小的
    #         queryIdx = m.queryIdx  # 模型上
    #         trainIdx = m.trainIdx  # 场景中
    #
    #         match_scene_idx.append(trainIdx)  # 场景中 stack起来
    #         match_model_idx.append(queryIdx)  # 模型上
    #
    #         asarray(target_down.colors)[trainIdx, :] = [1, 0, 0]  # 场景
    #         asarray(source_down.colors)[queryIdx, :] = [0, 1, 0]  # 模型
    #
    # e_time = time.time()
    # print('match_time:', e_time - s_time)

    # 匹配点索引
    # match_model_idx = array(match_model_idx)
    # match_scene_idx = array(match_scene_idx)  # 场景点索引
    # print('match_model_idx:', match_model_idx)
    # print('match_scene_idx:', match_scene_idx)
    # print('match_scene_idx:', len(match_scene_idx))
    # print('match_scene_unique_idx:', len(unique(match_scene_idx)))

    # 匹配的点
    # model_paired = model_np[match_model_idx]
    # scene_paired = scene_np[match_scene_idx]

    ##################
    # 根据匹配点进行位姿估计  3D-3D  RANSAC构造模型的核还需要改
    # res_ransac, _, _, _ = ransac_pose(model_paired, scene_paired, 50)
    # print('res_ransac:\n', res_ransac)
    # draw_registration_result(source, target, res_ransac, is_rgb)  # 粗配准后
    # 问题是 如果本来就重合，但是特征配准错了，就会偏移! 比如CVLab的Kinect数据集

    # 为了可视化
    # print('scene_vtx:', len(scene_vtx))
    # model_vtx_num = len(model_np)
    # print('model_vtx_num:', model_vtx_num)  # md 纠结好久，原来是场景家的不对！索引不是顶点的
    # line_vtx = r_[model_np, scene_np]  # 顶点坐标stack起来
    # line_idx = c_[match_model_idx, match_scene_idx + model_vtx_num]  # idx的场景那一半也得加上对应的长度
    # print('line_vtx:', line_vtx.shape)
    # print('line_idx:', line_idx.shape)
    #
    # color = array([0, 0.3, 1])
    # colors = tile(color, (len(line_idx), 1))  # 复制数组
    # print(colors)

    # line_set = draw_line(line_vtx, line_idx, colors)  # 对应点的匹配线  # points lines color
    # line_set = draw_line(ab, idx, colors)  # 对应点的匹配线  # points lines color 颜色个数=点对个数

    # 设置背景 ？？ 未完成
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # # 将两个点云放入visualizer
    # vis.add_geometry(source)
    # # vis.add_geometry(target)
    # vis.get_render_option().point_size = 2  # 点云大小
    # vis.get_render_option().background_color = np.asarray([0, 0, 0])  # 背景颜色

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
