from cv_lib import *
from o3d_pose_lib import *
from ransac import *
from path_lib import *

import time
from joblib import dump, load  # 保存模型

# ML lib
from ml_lib import *


# 通过类别加载对应的模型，并进行采样等等
color_map = {0: [1, 0, 0], 1: [0.2, 1, 0], 2: [0.1, 0.8, 0.5],
             3: [1, 1, 0], 4: [0.2, 0.8, 1], 5: [1, 0, 1]}

if __name__ == '__main__':

    root_path = 'D:/SIA/Dataset/UWA/'
    class_encode = os.listdir(root_path + 'model/unzip')

    scene_path = root_path + 'scene/unzip/rs1.ply'
    model_path = root_path + 'model/unzip/cheff.ply'

    model_idx_list = [0, 1, 2]  # 都有哪些模型？编码后的

    is_rgb = 0

    # voxel_size = 3.4  # means 5cm for the dataset
    voxel_size = 1.8  # means 5cm for the dataset

    flann = flann_init()

    s_time = time.time()  # 从什么时候开始计时？

    # 场景数据读取
    target, target_down, target_fpfh = prepare_dataset(scene_path, voxel_size)
    print('target:', target)
    print('target_down:', target_down)

    if not is_rgb:
        target.paint_uniform_color([0.5, 0.4, 0.1])  # 否则无法添加颜色
        # source_down.paint_uniform_color([0.4, 0.4, 0.4])  # 暂时不加载模型

    # 可视化原始数据
    # draw_registration_result(, target_down, np.identity(4), is_rgb)  # 采样 特征提取后的数据

    # 转换成np
    # model_np = array(source_down.points)  # 模型点
    scene_down_np = array(target_down.points)  # 场景点

    # 场景分割  全局的 不必分割，这里保留了是因为也可以用来做可视化(也被我去掉了)
    for class_res in model_idx_list:
        # 加载对应类别的模型
        # 根据类别找到对应的模型位置
        model_path = root_path + 'model/unzip/' + class_encode[class_res]  # + '.ply'
        # print('model_path:', model_path)

        model_trans_init = eye(4)  # 初始化模型位姿，更好地可视化  这个在加载的时候就要做
        model_trans_init[:3, 3:] = array([0, -100.5, 1]).reshape(3, -1)
        # source, source_down = load_model(model_path, model_trans_init, voxel_size)  # 加载模型
        source, source_down, source_fpfh = prepare_dataset(model_path, voxel_size, model_trans_init)  # 加载模型

        source_down.paint_uniform_color([0.5, 0.5, 0.65])  # 否则没有颜色
        model_down_np = array(source_down.points)  # 模型点

        # 可视化加载模型和场景的效果，查看初始位姿
        draw_registration_result(source_down, target_down, np.identity(4), is_rgb)  # 采样 特征提取后的数据

        # 找到当前类别的特征向量(们)，然后和随影类别的模型上面的特征向量进行匹配
        # feature_path = feature_bag_root + os.path.splitext(class_encode[class_res])[0] + '.npy'
        # model_feature = np.load(feature_path)  # 输入类别，输出模型上所有的特征  这些特征的索引就是对应点的索引

        model_feature = float32(source_fpfh.data.T)  # 格式转换 否则flann不支持
        target_feature = float32(target_fpfh.data.T)  # 不能用分割之后的！ 要不然哪有什么区别？？

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

        # print(match_scene_idx)
        # 匹配的对应点添加颜色
        # asarray(source_down.colors)[match_model_idx, :] = [0, 1, 0]  # 模型
        # asarray(target_down.colors)[match_scene_idx, :] = [1, 0, 0]  # 场景 对应点着色

        # 根据索引找到对应的三维坐标  匹配的坐标点 用于位姿估计
        model_paired = model_down_np[match_model_idx]
        scene_paired = scene_down_np[match_scene_idx]  # 此时，match不改变索引

        # 进行粗位姿估计 (毕竟就算出去也得循环)
        # 根据匹配点进行位姿估计  3D-3D  RANSAC构造模型的核还需要改
        # source, target, iter_num, inlier_threshold
        # res_ransac, confi, inlier_buff, outlier_buff = ransac_pose(model_paired, scene_paired, 150, 0.09)
        _, confi, inlier_buff, outlier_buff = ransac_pose(model_paired, scene_paired, 50, 0.09)

        res_ransac = execute_global_registration(source_down, target_down,
                                    source_fpfh, target_fpfh, voxel_size)
        # # print('res_ransac:\n', res_ransac)
        res_ransac = res_ransac.transformation  # 库函数的
        # print('res_ransac:\n', res_ransac)

        # 内点索引？
        print('inlier_buff:', len(inlier_buff))
        new_idx = array(range(len(inlier_buff)))

        # 根据RANSAC得到 全局匹配时的 内外点
        inlier_line_set = get_line_set(model_paired, scene_paired, inlier_buff, inlier_buff, array([0, 0.3, 1]))
        outlier_line_set = get_line_set(model_paired, scene_paired, outlier_buff, outlier_buff, array([1, 0.3, 0]))

        draw_registration_result(source_down, target_down, res_ransac, is_rgb, 'Global')  # 粗配准后

        # 位姿精细估计
        res_icp = refine_registration(source, target, res_ransac, voxel_size)
        # draw_registration_result(source, target, res_icp.transformation, is_rgb, 'Global ICP')  # 粗配准后

        # 单物体的  （这里要根据具体的类别加载一下）
        axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

        line_color = [0.9, 0.2, 0]
        line_set = get_line_set(model_down_np, scene_down_np, match_model_idx, match_scene_idx, line_color)
        # draw_line_down(source_down, target_down, line_set, axis_pcd)
        draw_line_down(source_down, target_down, line_set, axis_pcd)
        # draw_line_down(source_down, target_down, inlier_line_set, outlier_line_set)

        # 缓存起来，方便后面进行可视化（加载所有模型 划线）
