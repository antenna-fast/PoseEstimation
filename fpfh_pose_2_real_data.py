from o3d_pose_lib import *
from ransac import *

from path_lib import *
from joblib import dump, load  # 保存模型

from ml_lib import *
from cv_lib import *

import time


# 对比真实数据集
# 单独拿出来是因为，真实数据集有很多噪声，这在分割与分类上会造成困扰

# class_encode = ['armadillo', 'buddha', 'bunny', 'chinese_dragon', 'dragon', 'statuette']
class_encode = ['bear.ply', 'dinasaur.ply', 'face.ply', 'ghost.ply', 'mboy.ply', 'robot.ply']
color_map = {0: [1, 0, 0], 1: [0.2, 1, 0],  2: [0.1, 0.8, 0.5],
             3: [1, 1, 0], 4: [0.2, 0.8, 1], 5: [1, 0, 1]}


if __name__ == '__main__':

    # 数据集根目录
    model_root_cvlab = 'D:/SIA/Dataset/CVLab/'
    # 输入索引，输出模型的fpfh  * 根据索引保存
    feature_bag_root = 'D:/SIA/Dataset/FeatureBag/CVLabFPFH/'
    # 加载分类器模型
    id_str = '2009-10-27'
    model_save_path = 'model_lib/real_model_' + id_str + '_adaboost.clf'

    clf = load(model_save_path)

    # 真实场景
    scene_path = model_root_cvlab + '2009-10-27/Scene1.ply'
    # model_path = model_root_cvlab + '2009-10-27/model1.ply'  # 小人

    # scene_path = model_root_cvlab + '2010-06-12/Scene1/Scene1.ply'  # 可乐瓶 kinect
    # model_path = model_root_cvlab + '2010-06-12/Scene1/model1.ply'

    # scene_path = model_root_cvlab + '2010-03-03/Scene1/scene1.ply'  # 标注只有 小人 恐龙
    # scene_path = model_root_cvlab + '2010-03-03/Scene1/scene2.ply'  # 标注只有 小人 恐龙
    # scene_path = model_root_cvlab + '2010-03-03/Scene1/scene3.ply'  # 标注只有 小人 恐龙
    # model_path = model_root_cvlab + '2010-03-03/Scene1/model1.ply'

    # 快速匹配
    flann = flann_init()

    # 参数
    is_rgb = 1

    # voxel_size = 0.005  # means 5cm for the dataset
    voxel_size = 0.4  # means 5cm for the dataset
    # voxel_size = 0.21  # means 5cm for the dataset

    # seg
    threshold_n = 50  # 通过点的个数进行滤波 每个点集最少的点数
    # ml
    threshold_score = 0.23  # 分类器置信度阈值

    # 场景数据加载
    # source, target, source_down, target_down, source_fpfh, target_fpfh = \
    #     prepare_dataset(model_path, scene_path, voxel_size)

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

    s_time = time.time()  # 从什么时候开始计时？
    # 场景分割

    # 去除大平面

    # 1.scene_down_np, 2.search_radius, 3.min_sample, 4.threshold_n, 5.class_num
    seg_res_idx = dbscan_segment(scene_down_np, 0.5, 5, 50, 5)

    for class_member_mask in seg_res_idx:
        print('for loop')  # 这里注意，还没有训练模型呢
        # 点云分类 提取所有点的特征点，放到分类器里面
        seg_target = target_down.select_by_index(class_member_mask)  # true false的不行
        seg_pcd_np = array(seg_target.points)
        # print('seg_pcd:', seg_pcd)  # 这里就已经不对了  已解决，不能用 TrueFalse索引

        seg_target_fpfh_T = get_fpfh(seg_target, voxel_size)  # 输入分割后的点云，输出FPFH
        # print('seg_target_fpfh:', seg_target_fpfh_T.shape)

        res = clf.predict(seg_target_fpfh_T)  # 分类器预测
        # print('res:', res)

        # 统计res里面最多的元素，计算类别并输出
        class_res, score_res = get_class_res(res)  # 可以再看看置信度：最多的类别占全部的多少？
        print('class_res:{0}  score:{1}'.format(class_encode[class_res], score_res))

        if score_res < threshold_score:  # 分类置信度小于阈值，直接跳过
            continue

        # 以下找到当前索引的特征对应的哪个类别  接下来就是加载对应类别的特征 对应起来
        # asarray(target_down.colors)[class_member_mask, :] = color_map[class_res]  # 根据类别着色
        # seg_pcd_np = array(target_down.points)[class_member_mask, :]

        # 加载对应类别的模型
        # 根据类别找到对应的模型位置
        # model_path = model_root_cvlab + '3D Models/2010-03-03/' + class_encode[class_res]
        if class_encode[class_res] is not 'bear.ply':
            continue

        model_path = model_root_cvlab + '3D Models/2009-10-27/' + class_encode[class_res]
        # print('model_path:', model_path)

        model_trans_init = eye(4)  # 初始化模型位姿，更好地可视化
        # model_trans_init[:3, 3:] = array([1.5, 0.5, 1]).reshape(3, -1) * 0.2  ######
        source, source_down = load_model(model_path, model_trans_init, voxel_size)

        if not is_rgb:
            source_down.paint_uniform_color([0.4, 0.4, 0.4])  # 否则没有颜色

        source_down_np = array(source_down.points)  # 模型点

        # 找到当前类别的特征向量(们)，然后和相应类别的模型上面的特征向量进行匹配
        feature_path = feature_bag_root + os.path.splitext(class_encode[class_res])[0] + '.npy'
        model_feature = np.load(feature_path)  # 输入类别，输出模型上所有的特征  这些特征的索引就是对应点的索引

        # 格式转换 否则flann不支持
        model_feature = float32(model_feature)
        target_feature_seg = float32(seg_target_fpfh_T)

        # 可以替代上面的一堆  为了迎合o3d的注册函数  为了区别，这里用source表示模型
        # _, _, source_fpfh = prepare_dataset(model_path, voxel_size)

        # 特征匹配，得到匹配的点  输入模型特征和当前分割后的特征，输出匹配的索引
        # 输出分割后对应的索引
        matches = flann.knnMatch(model_feature, target_feature_seg, k=2)  # 场景 模型 近邻个数

        match_model_idx = []
        match_scene_idx = []

        for (m, n) in matches:  # m是最小距离  n是次小距离（或者一会加上过滤）
            # print(m.distance)
            if m.distance < 0.8 * n.distance:  # 什么原理？ 最小的<0.7次小的
                queryIdx = m.queryIdx  # 模型上
                trainIdx = m.trainIdx  # 场景中

                match_model_idx.append(queryIdx)  # 模型上
                match_scene_idx.append(trainIdx)  # 场景中 stack起来  seg的索引！！

                asarray(source_down.colors)[queryIdx, :] = [0, 1, 0]  # 模型
                # asarray(target_down.colors)[trainIdx, :] = [1, 0, 0]  # 场景
                asarray(seg_target.colors)[trainIdx, :] = [1, 0, 0]  # 场景

        match_model_idx = array(match_model_idx)
        match_scene_idx = array(match_scene_idx)  # 场景点索引

        # 根据索引找到对应的三维坐标  用于位姿估计
        model_paired = source_down_np[match_model_idx]
        scene_target_np = array(seg_target.points)
        scene_paired = scene_target_np[match_scene_idx]  # match改变了索引

        # 进行粗粒度位姿估计 (毕竟就算出去也得循环)
        # 根据匹配点进行位姿估计  3D-3D  RANSAC构造模型的核还需要改
        res_ransac, _, _, _ = ransac_pose(model_paired, scene_paired, 10)  # 这里必须是配对好的--带着误差进行的迭代
        print('res_ransac:\n', res_ransac)  # 变换矩阵

        # 直接这样，相当于模型的点云和场景中的对齐后 错乱了  不行！
        # res_ransac = execute_global_registration(source_down, seg_target,
        #                             source_fpfh, target_fpfh, voxel_size)
        # # print('res_ransac:\n', res_ransac)
        # res_ransac = res_ransac.transformation  # 库函数的
        # print('res_ransac:\n', res_ransac)

        draw_registration_result(source, target, res_ransac, is_rgb, 'Ours')  # 粗配准后

        # 位姿精细估计
        # res_icp = refine_registration(source_down, seg_target, res_ransac, voxel_size)
        # res_icp = refine_registration(source_down, target_down, res_ransac, voxel_size)
        res_icp = refine_registration(source, target, res_ransac, voxel_size)
        draw_registration_result(source, target, res_icp.transformation, is_rgb, 'Ours ICP')  # 粗配准后

        # 单物体的  （这里要根据具体的类别加载一下）
        axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=12, origin=[0, 0, 0])

        line_set = get_line_set(source_down_np, scene_target_np, match_model_idx, match_scene_idx)
        draw_line_down(source_down, target_down, line_set, axis_pcd)

        # 缓存起来，方便后面进行可视化（加载所有模型 划线）

    # 跳出来之后，进行对应点的可视化，一看，呦，没有多少误匹配了！

    # 设置背景 ？？ 未完成
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # # 将两个点云放入visualizer
    # vis.add_geometry(source)
    # # vis.add_geometry(target)
    # vis.get_render_option().point_size = 2  # 点云大小
    # vis.get_render_option().background_color = np.asarray([0, 0, 0])  # 背景颜色
