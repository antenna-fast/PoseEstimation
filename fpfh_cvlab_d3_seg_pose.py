import time
from joblib import dump, load  # 保存模型

from utils.o3d_pose_lib import *
from utils.ransac import *
from utils.path_lib import *
from utils.ml_lib import *
from utils.cv_lib import *


# CVLab dataset3

if __name__ == '__main__':
    color_map = {0: [1, 0, 0], 1: [0.2, 1, 0], 2: [0.1, 0.8, 0.5],
                 3: [1, 1, 0], 4: [0.2, 0.8, 1], 5: [1, 0, 1]}
    precision_dict = {}  # 用于保存精确率

    # 数据集根目录
    model_root_cvlab = 'D:/SIA/Dataset/CVLab/'
    # 输入索引，输出模型的fpfh  * 根据索引保存

    # 加载分类器模型
    # id_str = '2009-10-27'
    id_str = '2010-03-03'
    SceneFile = 'Scene3/'
    Scene = 'Scene1.ply'
    # Scene = 'Scene2.ply'
    # Scene = 'Scene3.ply'

    feature_bag_root = 'D:/SIA/Dataset/FeatureBag/CVLabFPFH/' + id_str + '/'  # 仅有两类的

    # 精确度保存路径
    precision_path = 'C:/Users/yaohua-win/PycharmProjects/pythonProject/pose_sys/measure/precision/'

    model_root = 'D:/SIA/Dataset/CVLab/3D Models/' + id_str + '/filted/'  # 真实数据集
    class_encode = get_file_list(model_root, '.ply')  # 过滤
    print('class_encode:', class_encode)

    # 分类器
    # model_save_path = 'feature_classify/model_lib/real_model_' + id_str + '_adaboost.clf'
    model_save_path = 'feature_classify/model_lib/cvlib_adaboost_' + id_str + '.clf '
    clf = load(model_save_path)

    # 真实场景
    scene_path = model_root_cvlab + id_str + '/' + SceneFile + Scene

    # 可视化参数
    is_rgb = 1
    inlier_color = array([0, 0.3, 1])
    outlier_color = array([1, 0.3, 0])

    bbox_gt_color = [1, 0, 0]  # 红色是GT
    bbox_pred_color = [0, 1, 0]  # 绿色是预测

    voxel_size = 0.4  # means 5cm for the dataset
    # voxel_size = 0.21  # means 5cm for the dataset

    # DBSCAN papameter
    dbscan_r = 0.55
    # dbscan_r = 0.65
    dbscan_sample_num = 5
    dbscan_clus_thres = 60  # 通过点的个数进行滤波 每个点集最少的点数  相当于,又找了一个下线
    dbscan_class_num = 8  # 前5类

    # FLANN
    flann_ratio = 0.85

    # RANSAC parameter
    ransac_iter_num = 360  # 评估距离大于此数视为外点
    ransac_inlier_threshold = 12  # 评估距离大于此数视为外点

    # Machine Learning parameter
    threshold_score = 0.23  # 分类器置信度阈值

    # 场景数据加载
    target, target_down, target_fpfh = prepare_dataset(scene_path, voxel_size)

    if not is_rgb:
        target_down.paint_uniform_color([0.4, 0.4, 0.4])  # 否则无法添加颜色

    # 可视化原始数据
    # draw_registration_result(source, target, np.identity(4), is_rgb)
    # draw_registration_result(source_down, target_down, np.identity(4), is_rgb)  # 采样 特征提取后的数据

    # 转换成np
    scene_down_np = array(target_down.points)  # 场景点

    # 快速匹配
    flann = flann_init()

    s_time = time.time()  # 从什么时候开始计时？

    # 去除大平面（背景）

    # 场景分割(是场景理解的基础)
    # 1.scene_down_np, 2.search_radius, 3.min_sample, 4.threshold_n, 5.class_num
    seg_res_idx = dbscan_segment(scene_down_np, dbscan_r, dbscan_sample_num, dbscan_clus_thres, dbscan_class_num)

    for class_member_mask in seg_res_idx:  # [2:]:  # 去掉的是背景
        print('for loop')

        # 点云分类 提取分割后的特征点，放到分类器里面
        seg_target = target_down.select_by_index(class_member_mask)  # true false的不行
        seg_pcd_np = array(seg_target.points)
        # print('seg_pcd:', seg_pcd)  # 这里就已经不对了  已解决，不能用 TrueFalse索引
        # show_pcd(seg_target)

        seg_target_fpfh_T = get_fpfh(seg_target, voxel_size)  # 输入分割后的点云，输出FPFH

        res = clf.predict(seg_target_fpfh_T)  # 分类器预测
        # print('res:', res)

        # 统计res里面最多的元素，计算类别并输出
        class_res, score_res = get_class_res(res)  # 可以再看看置信度：最多的类别占全部的多少？
        model_name = class_encode[class_res]
        print('class_res:{0}  score:{1}'.format(model_name, score_res))

        if score_res < threshold_score:  # 分类置信度小于阈值，直接跳过
            continue

        # if class_encode[class_res] is not 'mboy.ply':  # 专门看哪一类的
        #     continue

        # 加载对应的模型
        model_path = model_root_cvlab + '3D Models/' + id_str + '/filted/' + model_name
        # print('model_path:', model_path)

        model_trans_init = eye(4)  # 初始化模型位姿，更好地可视化
        # model_trans_init[:3, 3:] = array([1.5, 0.5, 1]).reshape(3, -1) * 0.2  ######
        # source, source_down = load_model(model_path, model_trans_init, voxel_size)
        source, source_down, source_fpfh = prepare_dataset(model_path, voxel_size, model_trans_init)

        if not is_rgb:
            source_down.paint_uniform_color([0.4, 0.4, 0.4])  # 否则没有颜色

        source_down_np = array(source_down.points)  # 模型点

        # 找到当前类别的特征向量，然后和相应类别的模型上面的特征向量进行匹配
        # feature_path = feature_bag_root + os.path.splitext(model_name)[0] + '.npy'
        # model_feature = np.load(feature_path)  # 输入类别，输出模型上所有的特征  这些特征的索引就是对应点的索引

        # 格式转换 否则flann不支持
        # model_feature = float32(model_feature)
        model_feature = float32(source_fpfh.data.T)
        target_feature_seg = float32(seg_target_fpfh_T)

        # 可以替代上面的一堆  为了迎合o3d的注册函数  为了区别，这里用source表示模型
        # _, _, source_fpfh = prepare_dataset(model_path, voxel_size)

        # 特征匹配，得到匹配的点  输入模型特征和当前分割后的特征，输出匹配的索引
        # 输出分割后对应的索引
        matches = flann.knnMatch(model_feature, target_feature_seg, k=2)  # 场景 模型 近邻个数

        print('target_feature_seg:', len(target_feature_seg))
        print('model_feature:', len(model_feature))

        match_model_idx = []
        match_scene_idx = []

        for (m, n) in matches:  # m是最小距离  n是次小距离（或者一会加上过滤）
            # print(m.distance)
            if m.distance < flann_ratio * n.distance:  # 什么原理？ 最小的<0.7次小的
                queryIdx = m.queryIdx  # 模型上
                trainIdx = m.trainIdx  # 场景中

                match_model_idx.append(queryIdx)  # 模型上
                match_scene_idx.append(trainIdx)  # 场景中 stack起来  seg的索引！！

                # asarray(source_down.colors)[queryIdx, :] = [0, 1, 0]  # 模型
                # asarray(target_down.colors)[trainIdx, :] = [1, 0, 0]  # 场景
                # asarray(seg_target.colors)[trainIdx, :] = [1, 0, 0]  # 场景

        match_model_idx = array(match_model_idx).astype(int)
        match_scene_idx = array(match_scene_idx).astype(int)  # 场景点索引

        # 根据索引找到对应的三维坐标  用于位姿估计
        model_paired = source_down_np[match_model_idx]
        scene_target_np = array(seg_target.points)
        scene_paired = scene_target_np[match_scene_idx]  # match改变了索引
        print('model_paired:', len(model_paired))  # 检验匹配的个数

        # 进行粗粒度位姿估计 (毕竟就算出去也得循环)
        # 根据匹配点进行位姿估计  3D-3D  RANSAC构造模型的核还需要改
        # source, target, iter_num, inlier_threshold
        # model_param, confidence, inlier_buff, outlier_buff

        res_ransac, confidence, inlier, outlier = \
            ransac_pose(model_paired, scene_paired, ransac_iter_num, ransac_inlier_threshold)  # 这里必须是配对好的--带着误差进行的迭代

        inlier_num = len(inlier)
        outlier_num = len(outlier)
        precision = inlier_num / (inlier_num + outlier_num)  # TP/TP+FP
        print('匹配准确率: {0:.3f}'.format(precision))
        precision_dict[model_name] = precision  # 如果不想带ply  就在这去掉

        # 根据RANSAC得到 类内匹配时的 内外点
        # line_set = get_line_set(source_down_np, seg_target_np, match_model_idx, match_scene_idx)
        inlier_line_set = get_line_set(model_paired, scene_paired, inlier, inlier, array([0, 0.3, 1]))
        outlier_line_set = get_line_set(model_paired, scene_paired, outlier, outlier, array([1, 0.3, 0]))

        # 直接这样，相当于模型的点云和场景中的对齐后 错乱了  不行！
        # res_ransac = execute_global_registration(source_down, seg_target,
        #                             source_fpfh, target_fpfh, voxel_size)
        # # print('res_ransac:\n', res_ransac)
        # res_ransac = res_ransac.transformation  # 库函数的
        # print('res_ransac:\n', res_ransac)

        # GT
        gt_mat = loadtxt('D:/SIA/Dataset/CVLab/' + id_str + '/' + SceneFile + '/ground_truth.xf')
        # print('gt_mat:\n', gt_mat)

        # 加载bbox
        # bbox_path = 'D:/SIA/Dataset/CVLab/3D Models/' + id_str + '/bbox/' + 'obb_mboy.txt'
        # bbox_path = 'D:/SIA/Dataset/CVLab/3D Models/' + id_str + '/bbox/' + 'obb_face.txt'
        bbox_path = 'D:/SIA/Dataset/CVLab/3D Models/' + id_str + '/bbox/' + 'obb_robot.txt'
        bbox_np = loadtxt(bbox_path)  # 不同模型的  没有和数据集的命名关联起来  错了！本来想弄robot

        # 变换bbox
        bbox_gt = get_bbox(bbox_np, gt_mat, bbox_gt_color)  # gt
        bbox_pred = get_bbox(bbox_np, res_ransac, bbox_pred_color)  # 预测的bbox

        # print('draw_coase_registration_result')
        # draw_registration_result(source, target, res_ransac, is_rgb, 'Ours')  # 粗配准后
        draw_registration_with_bbox(source, target, bbox_gt, bbox_pred, res_ransac, is_rgb, 'Ours')  # 粗配准后

        # 位姿精细估计
        # res_icp = refine_registration(source_down, seg_target, res_ransac, voxel_size)
        # res_icp = refine_registration(source_down, target_down, res_ransac, voxel_size)
        res_icp = refine_registration(source, target, res_ransac, voxel_size)
        res_icp = res_icp.transformation
        bbox_pred = get_bbox(bbox_np, res_icp, bbox_pred_color)

        # draw_registration_result(source, target, res_icp.transformation, is_rgb, 'Ours ICP')  # 粗配准后
        # draw_registration_with_bbox(source, target, bbox_gt, bbox_pred, res_icp, is_rgb, 'Ours ICP')  # 粗配准后

        # 单物体的  （这里要根据具体的类别加载一下）
        axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=12, origin=[0, 0, 0])

        # draw_line_down(source_down, target_down, line_set, axis_pcd)
        # draw_line_down(source, target, line_set, axis_pcd)
        draw_line_down(source, target, inlier_line_set, outlier_line_set)

        # 缓存起来，方便后面进行可视化（加载所有模型 划线）

    # 跳出来之后，进行对应点的可视化，一看，呦，没有多少误匹配了！
    save(precision_path + 'cvlab_n' + '.npy', precision_dict)
