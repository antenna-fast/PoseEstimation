import time
from joblib import dump, load  # 保存模型

from utils.o3d_impl import *
from utils.cv_lib import *
from utils.ml_lib import *
from utils.o3d_pose_lib import *
from utils.ransac import *
from utils.path_lib import *


# CVLab3D - dataset 1&2
# 通过类别加载对应的模型，并进行采样等等

if __name__ == '__main__':
    class_encode = ['armadillo', 'buddha', 'bunny', 'chinese_dragon', 'dragon', 'statuette']
    color_map = {0: [1, 0, 0], 1: [0.2, 1, 0], 2: [0.1, 0.8, 0.5],
                 3: [1, 1, 0], 4: [0.2, 0.8, 1], 5: [1, 0, 1]}

    precision_dict = {}

    # scene = 'Scene1'
    scene = 'Scene2'
    # scene = 'Scene31'
    # scene = 'Scene32'
    # scene = 'Scene37'
    model_root_stanford = 'D:/SIA/Dataset/Stanford/3D models/'

    # 输入索引，输出模型的fpfh  * 根据索引保存  已经提取好的特征：坏处就是要改俩参数，不过可以写到文件里面
    # feature_bag_root = 'D:/SIA/Dataset/FeatureBag/StanfordFPFH/'

    root_path = 'D:/SIA/Dataset/'
    scene_path = root_path + 'Stanford/RandomScene1/' + scene + '.ply'

    bbox_path = 'D:/SIA/Dataset/Stanford/model bbox/'

    # model_idx_list = [0, 3, 4]  # Scene1
    model_idx_list = [1, 2, 4]  # Scene2
    # model_idx_list = [0, 1, 3, 4, 5]  # Scene31
    # model_idx_list = [0, 1, 2, 4, 5]  # Scene31
    # model_idx_list = [0, 1, 2, 4, 5]  # Scene37

    # 精确度保存路径
    precision_path = 'C:/Users/yaohua-win/PycharmProjects/pythonProject/pose_sys/measure/precision/'

    # 可视化参数
    is_rgb = 0
    inlier_color = array([0, 0.3, 1])
    outlier_color = array([0, 0.3, 1])  #
    # outlier_color = array([1, 0.3, 0])

    voxel_size = 0.006  #
    # voxel_size = 0.005  #
    # voxel_size = 0.008  #

    threshold_n = 100  # 每个点集最少的点数
    ransac_iter_num = 105
    ransac_inlier_threshold = 0.09  # 评估距离大于此数视为外点

    flann = flann_init()

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

    # 改换成加载不同的模型 在场景中进行匹配
    for class_res in model_idx_list:
        # 根据类别找到对应的模型位置
        model_name = class_encode[class_res]
        m_path = model_root_stanford + model_name + '/'
        model_path = m_path + get_file_list(m_path, '.ply')[0]
        # print('model_path:', model_path)

        # 加载模型
        model_trans_init = eye(4)  # 初始化模型位姿，更好地可视化
        model_trans_init[:3, 3:] = array([0.3, 0.4, 0]).reshape(3, -1)
        source, source_down = load_model(model_path, model_trans_init, voxel_size)

        # 加载对应模型的bbox  这样不行
        obb_vtx = loadtxt(bbox_path + 'aabb_' + model_name + '.txt')  # 三维点 可对其进行平移旋转变换

        obb = np2bbox(obb_vtx, 0.01, array([0, 1, 0]))  # points, lines, colors  lineset格式
        # print('obb:', obb)

        source_down.paint_uniform_color([0.4, 0.4, 0.4])  # 否则没有颜色
        model_down_np = array(source_down.points)  # 模型点

        # draw_registration_result(source, target, np.identity(4), is_rgb)

        # 问题：不能适应不同的采样率，所以换掉
        # 找到当前类别的特征向量(们)，然后和随影类别的模型上面的特征向量进行匹配
        # feature_path = feature_bag_root + os.path.splitext(class_encode[class_res])[0] + '.npy'
        # model_feature = np.load(feature_path)  # 输入类别，输出模型上所有的特征  这些特征的索引就是对应点的索引

        # 可以替代上面的一堆  为了迎合o3d的注册函数  为了区别，这里用shource表示模型
        _, _, source_fpfh = prepare_dataset(model_path, voxel_size)
        model_feature = np.float32(source_fpfh.data.T)

        # 格式转换 否则flann不支持
        target_feature = np.float32(target_fpfh.data.T)  # 不能用分割之后的！ 要不然哪有什么区别？？

        # 特征匹配，得到匹配的点  输入模型特征和当前分割后的特征，输出分割后对应的索引
        matches = flann.knnMatch(model_feature, target_feature, k=2)  # 场景 模型 近邻个数

        # ratio test
        match_model_idx = []
        match_scene_idx = []
        for (m, n) in matches:  # m是最小距离  n是次小距离
            # print(m.distance)
            if m.distance < 0.85 * n.distance:  # 什么原理？ 最小的<0.7次小的
                queryIdx = m.queryIdx  # 模型上
                trainIdx = m.trainIdx  # 场景中
                match_model_idx.append(queryIdx)  # 模型上
                match_scene_idx.append(trainIdx)  # 场景中 target_down的索引！
        match_model_idx = np.array(match_model_idx)
        match_scene_idx = np.array(match_scene_idx)  # 场景点索引

        # 匹配的对应点添加颜色
        np.asarray(source_down.colors)[match_model_idx, :] = [0, 1, 0]  # 模型
        np.asarray(target_down.colors)[match_scene_idx, :] = [1, 0, 0]  # 场景 对应点着色

        # 根据索引找到对应的三维坐标  匹配的坐标点 用于位姿估计
        model_paired = model_down_np[match_model_idx]
        scene_paired = scene_down_np[match_scene_idx]  # 此时，match不改变索引

        # 进行粗位姿估计 (毕竟就算出去也得循环)
        # 根据匹配点进行位姿估计  3D-3D  RANSAC构造模型的核还需要改(但是其实也已经是匹配过的)
        # 迭代内部：度量标准有问题
        res_ransac, confi, inlier_buff, outlier_buff = \
            ransac_pose(model_paired, scene_paired, ransac_iter_num, ransac_inlier_threshold)

        # 此处输出准确率
        inlier_num = len(inlier_buff)
        outlier_num = len(outlier_buff)
        precision = inlier_num / (inlier_num + outlier_num)  # TP/TP+FP
        print('模型: {0}  匹配准确率: {1:.3f}'.format(model_name, precision))

        precision_dict[model_name] = precision

        # 根据RANSAC得到 全局匹配时的 内外点
        # model_np, scene_np, match_model_idx, match_scene_idx, color
        inlier_line_set = get_line_set(model_paired, scene_paired, inlier_buff, inlier_buff, inlier_color)
        outlier_line_set = get_line_set(model_paired, scene_paired, outlier_buff, outlier_buff, outlier_color)

        # 库函数粗配准
        # res_ransac = execute_global_registration(source_down, target_down,
        #                             source_fpfh, target_fpfh, voxel_size)
        # # print('res_ransac:\n', res_ransac)
        # res_ransac = res_ransac.transformation  # 库函数的
        # print('res_ransac:\n', res_ransac)

        draw_registration_result(source, target, res_ransac, is_rgb, 'Global')  # 粗配准后
        # draw_registration_with_bbox(source, target, obb, obb, res_ransac, is_rgb, 'Global')  # 粗配准后

        # 位姿精细估计
        res_icp = refine_registration(source, target, res_ransac, voxel_size)
        draw_registration_result(source, target, res_icp.transformation, is_rgb, 'Global ICP')  # 粗配准后

        # 单物体的  （这里要根据具体的类别加载一下）
        axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

        # draw_line_down(source_down, target_down, line_set, axis_pcd)
        draw_line_down(source, target, inlier_line_set, outlier_line_set)
        # draw_line_down(source_down, target_down, inlier_line_set, outlier_line_set)

    # 保存测量值  精确度
    save(precision_path + 'syn_global_' + scene + '.npy', precision_dict)
