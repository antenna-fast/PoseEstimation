from o3d_impl import *
from cv_lib import *
from ml_lib import *
from o3d_pose_lib import *
from ransac import *
from path_lib import *
from joblib import dump, load  # 保存模型
import time

# 通过类别加载对应的模型，并进行采样等等

color_map = {0: [1, 0, 0], 1: [0.2, 1, 0], 2: [0.1, 0.8, 0.5],
             3: [1, 1, 0], 4: [0.2, 0.8, 1], 5: [1, 0, 1]}

trans_list = [[0.24, -0.15, 0],  # Amad
              [0.24, -0.25, 0],  # happy
              [0.24, -0.25, 0],  # 兔
              [0.24, -0.15, 0],  # 恶龙
              [0.24, -0.20, 0],  #
              [0.24, -0.15, 0],  #
              ]

precision_dict = {}


if __name__ == '__main__':

    # 加载地址
    model_root_stanford = 'D:/SIA/Dataset/Stanford/3D models/'
    class_encode = os.listdir(model_root_stanford)

    # 输入索引，输出模型的fpfh  * 根据索引保存
    feature_bag_root = 'D:/SIA/Dataset/FeatureBag/StanfordFPFH/'

    # 加载分类器模型
    model_save_path = 'C:/Users/yaohua-win/PycharmProjects/pythonProject/' \
                      'pose_sys/feature_classify/stanford_syn_adaboost.clf'

    clf = load(model_save_path)

    root_path = 'D:/SIA/Dataset/'

    scene_path = 'D:/SIA/Dataset/Stanford/ANTennnaScene/6modelScene.ply'

    bbox_path = 'D:/SIA/Dataset/Stanford/model bbox/'  # bbox 模型坐标系下的顶点
    gt_path = 'D:/SIA/Dataset/Stanford/ANTennnaScene/gt_pose/'

    # 保存地址
    test_result_path = 'D:/SIA/Dataset/Stanford/ANTennnaScene/precision/'

    # 可视化参数
    is_rgb = 0
    inlier_color = array([0, 0.3, 1])
    outlier_color = array([1, 0.3, 0])
    source_color = [1, 0.706, 0]
    target_color = [0, 0.651, 0.929]

    voxel_size = 0.006  # means 5cm for the dataset

    dbscan_r = 0.01
    dbscan_num = 6
    dbscan_clus_thres = 30  # 通过点的个数进行滤波 每个点集最少的点数
    dbscan_class_num = 6  # 前_类

    # RANSAC
    ransac_iter_num = 55  # 评估距离大于此数视为外点
    ransac_inlier_threshold = 0.13  # 评估距离大于此数视为外点

    flann = flann_init()  # fast lib

    s_time = time.time()  # 从什么时候开始计时? 数据加载??

    # 场景数据加载
    target, target_down, target_fpfh = prepare_dataset(scene_path, voxel_size)

    if not is_rgb:
        target_down.paint_uniform_color([0.4, 0.4, 0.4])  # 否则无法添加颜色

    # 可视化原始数据
    # draw_registration_result(source, target, np.identity(4), is_rgb)
    # draw_registration_result(target_down, target_down, np.identity(4), is_rgb)  # 采样 特征提取后的数据

    # 转换成np
    scene_down_np = array(target_down.points)  # 场景点

    # 分割
    # 1.scene_down_np, 2.search_radius, 3.min_sample, 4.threshold_n, 5.class_num
    seg_res_idx = dbscan_segment(scene_down_np, dbscan_r, dbscan_num, dbscan_clus_thres, dbscan_class_num)
    # print('seg_cluster:', len(seg_res_idx))

    frame_count = 0  # 帧标志位

    # OPEN3D begain  ---------------------  可视化
    # vis = o3d.visualization.Visualizer()
    # vis.create_window(window_name='ANTenna3D')
    #
    # # 设置窗口背景颜色
    # opt = vis.get_render_option()
    # # opt.background_color = np.asarray([0, 0., 0.0])  # up to 1
    # opt.background_color = np.asarray([1, 1., 1.0])  # up to 1

    show_list = []
    bbox_list_gt = []
    bbox_list_pred = []

    for class_member_mask in seg_res_idx:

        # 点云分类 提取所有点的特征点，放到分类器里面进行分类
        seg_target = target_down.select_by_index(class_member_mask)  # true false的不行
        seg_target_np = array(seg_target.points)

        seg_target_fpfh_T = get_fpfh(seg_target, voxel_size)  # 输入分割后的点云，输出FPFH
        # print('seg_target_fpfh:', seg_target_fpfh_T.shape)

        res = clf.predict(seg_target_fpfh_T)  # 分类器预测

        # 统计res里面最多的元素，计算类别并输出
        class_res, score_res = get_class_res(res)  # 可以再看看置信度：最多的类别占全部的多少？
        model_name = class_encode[class_res]
        print('class_res:{0}  score:{1}'.format(model_name, score_res))

        # 以下找到当前索引的特征对应的哪个类别  接下来就是加载对应类别的特征 对应起来
        # asarray(seg_target.colors) = color_map[class_res]  # 不同类别的颜色  应当加上分类器之后
        # asarray(target_down.colors)[class_member_mask, :] = color_map[class_res]  # 下采样后的

        # 加载对应类别的模型 根据类别找到对应的模型位置
        m_path = model_root_stanford + model_name+ '/'
        model_path = m_path + get_file_list(m_path, '.ply')[0]
        # print('model_path:', model_path)

        model_trans_init = eye(4)  # 初始化模型位姿，更好地可视化
        model_trans_init[:3, 3:] = array(trans_list[class_res]).reshape(3, -1)

        source, source_down = load_model(model_path, model_trans_init, voxel_size)
        # source, source_down, source_fpfh = prepare_dataset(model_path, voxel_size)

        # source_down.paint_uniform_color([0.4, 0.4, 0.4])  # 否则后面无法添加颜色：报错  然而根本就没有可视化
        source_down_np = array(source_down.points)  # 模型点

        # 加载对应模型的bbox
        obb_vtx = loadtxt(bbox_path + 'aabb_' + model_name + '.txt')  # 三维点 可对其进行平移旋转变换

        # 还要根据GT进行变换！！！  直接使用get_bbox
        gt_pose = loadtxt(gt_path + model_name + '.txt')
        obb = get_bbox(obb_vtx, gt_pose, array([1, 0, 0]))  # 这里的gt_pose就是trans_list

        # 问题：不能适应不同的采样率，所以换掉
        # 找到当前类别的特征向量(们)，然后和随影类别的模型上面的特征向量进行匹配
        # feature_path = feature_bag_root + os.path.splitext(class_encode[class_res])[0] + '.npy'
        # model_feature = np.load(feature_path)  # 输入类别，输出模型上所有的特征  这些特征的索引就是对应点的索引

        # 为了区别，这里用source表示模型
        _, _, source_fpfh = prepare_dataset(model_path, voxel_size)
        model_feature = source_fpfh.data.T

        # 格式转换 否则flann不支持
        model_feature = float32(model_feature)
        target_feature_seg = float32(seg_target_fpfh_T)

        # 特征匹配，得到匹配的点  输入模型特征和当前分割后的特征，输出匹配的索引
        # 输出分割后对应的索引
        matches = flann.knnMatch(model_feature, target_feature_seg, k=2)  # 场景 模型 近邻个数

        match_model_idx = []  # FLANN 经过 比例滤波的匹配结果
        match_scene_idx = []

        for (m, n) in matches:  # m是最小距离  n是次小距离（或者一会加上过滤）
            # print(m.distance)
            if m.distance < 0.85 * n.distance:  # 什么原理？ 最小的<0.7次小的
                queryIdx = m.queryIdx  # 模型上
                trainIdx = m.trainIdx  # 场景中

                match_model_idx.append(queryIdx)  # 模型上
                match_scene_idx.append(trainIdx)  # 场景中 stack起来  seg的索引！！

        match_model_idx = array(match_model_idx)
        match_scene_idx = array(match_scene_idx)  # 场景点索引

        # asarray(source_down.colors)[match_model_idx, :] = [0, 1, 0]  # 模型
        # asarray(target_down.colors)[trainIdx, :] = [1, 0, 0]  # 场景  索引不是这上面的
        # asarray(seg_target.colors)[match_scene_idx, :] = [1, 0, 0]  # 分割后的场景  但没有进行可视化

        # 根据索引找到对应的三维坐标 用于位姿估计
        model_paired = source_down_np[match_model_idx]
        scene_paired = seg_target_np[match_scene_idx]  # match改变了索引

        # 进行粗粒度位姿估计 -- 得到类内匹配一致性
        # 根据匹配点进行位姿估计  3D-3D  RANSAC构造模型的核还需要改
        # model_param, confidence, inlier, outlier
        # inlier outlier是相对于输入数据 model_paired scene_paired的索引
        res_ransac, confi, inlier_buff, outlier_buff = \
            ransac_pose(model_paired, scene_paired, ransac_iter_num, ransac_inlier_threshold)  # 这里必须是配对好的--但是带着误差进行的迭代 --就是这样用的

        inlier_num = len(inlier_buff)
        outlier_num = len(outlier_buff)
        # print('inlier_num:', inlier_num)
        # print('outlier_num:', outlier_num)
        precision = inlier_num / (inlier_num + outlier_num)  # TP/TP+FP
        print('匹配准确率: {0:.3f}'.format(precision))
        precision_dict[model_name] = precision

        obb_pred = get_bbox(obb_vtx, dot(res_ransac, model_trans_init), [0, 1, 0])  # 相对于模型坐标系！

        # 根据RANSAC得到 类内匹配时的 内外点
        # line_set = get_line_set(source_down_np, seg_target_np, match_model_idx, match_scene_idx)
        inlier_line_set = get_line_set(model_paired, scene_paired, inlier_buff, inlier_buff, inlier_color)
        outlier_line_set = get_line_set(model_paired, scene_paired, outlier_buff, outlier_buff, outlier_color)

        # 直接这样，相当于模型的点云和场景中的对齐后 错乱了  不行！
        # res_ransac = execute_global_registration(source_down, seg_target,
        #                             source_fpfh, target_fpfh, voxel_size)
        # res_ransac = res_ransac.transformation  # 库函数的
        # print('res_ransac:\n', res_ransac)

        # draw_registration_result(source, target, res_ransac, is_rgb, 'Ours')  # 粗配准后
        # draw_registration_with_bbox(source, target, obb, obb_pred, res_ransac, is_rgb, 'Ours')  # 粗配准后

        # 位姿精细估计
        # res_icp = refine_registration(source_down, target_down, res_ransac, voxel_size)
        res_icp = refine_registration(source, target, res_ransac, voxel_size)
        res_icp = res_icp.transformation

        # res_icp = res_ransac  # 粗配准
        obb_pred = get_bbox(obb_vtx, dot(res_icp, model_trans_init), [0, 1, 0])  # 相对于模型坐标系！
        # draw_registration_result(source, target, res_icp.transformation, is_rgb, 'Ours ICP')  # 粗配准后

        # draw_registration_with_bbox(source, target, obb, obb_pred, res_icp, is_rgb, 'Ours ICP')  # 粗配准后

        # 单物体的  （这里要根据具体的类别加载一下）
        axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

        draw_line_down(source, target, inlier_line_set, outlier_line_set)  # 好看！

        # 缓存
        # bbox_list_gt.append(obb)  # 已经变换完的
        bbox_list_gt.append(obb_pred)

        # 干脆也对模型进行变换
        source_trans = deepcopy(source).transform(res_icp)
        source_trans.paint_uniform_color(source_color)  # 颜色
        show_list.append(source_trans)

    show_list.append(target)

    # 保存测量值  精确度
    save(test_result_path + 'NB_syn_6model.npy', precision_dict)

    # show_batch_pcd(target, bbox_list_gt, bbox_list_pred)
    show_batch_pcd(show_list, bbox_list_gt, bbox_list_pred, 'Ours')
