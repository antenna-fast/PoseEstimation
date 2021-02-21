from o3d_pose_lib import *
from ransac import *
from path_lib import *
from cv_lib import *

import time
from joblib import dump, load  # 保存模型


# 真实数据
# 基于全局对齐，不需要加载模型的特征，可以先直接去（注释）掉他们？

# 需要解析ini文件，里面包含了模型的路径
# 当前策略：一个场景那里面就那么点东西，把模型和场景分别拿到就行了

# 这个保留，因为也可以加速匹配
def load_model_feature(feature_path):
    model_fpfh_feature = np.load(feature_path)
    return model_fpfh_feature


class_encode = ['armadillo', 'buddha', 'bunny', 'chinese_dragon', 'dragon', 'statuette']
color_map = {0: [1, 0, 0], 1: [0.2, 1, 0], 2: [0.1, 0.8, 0.5], 3: [1, 1, 0], 4: [0.2, 0.8, 1], 5: [1, 0, 1]}


if __name__ == '__main__':

    flann = flann_init()

    root_path = 'D:/SIA/Dataset/CVLab/'
    # source是模型
    # target是场景

    # 真实场景
    scene_path = root_path + '2009-10-27/Scene1.ply'
    model_path = root_path + '2009-10-27/model1.ply'  # 小人

    # scene_path = root_path + '2010-06-12/Scene1/Scene1.ply'  # 可乐瓶 kinect
    # model_path = root_path + '2010-06-12/Scene1/model1.ply'

    # scene_path = root_path + '2010-03-03/Scene1/scene1.ply'  # 小人  kinect
    # model_path = root_path + '2010-03-03/Scene1/model2.ply'  # 直接改这里吧，大批测试就改成自动加载
    # 很奇怪，为什么模型上面多了平面

    is_rgb = 1

    # voxel_size = 0.155  # means 5cm for the dataset
    # voxel_size = 0.005  # means 5cm for the dataset
    voxel_size = 0.4  # means 5cm for the dataset
    # voxel_size = 0.21  # means 5cm for the dataset

    s_time = time.time()  # 从什么时候开始计时？

    # 这里不必要求出场景的fpfh，加速点*
    target, target_down, target_fpfh = prepare_dataset(scene_path, voxel_size)
    source, source_down, source_fpfh = prepare_dataset(model_path, voxel_size)
    # 转换成np
    scene_down_np = array(target_down.points)  # 场景点
    model_down_np = array(source_down.points)  # 模型点

    if not is_rgb:
        target_down.paint_uniform_color([0.4, 0.4, 0.4])  # 否则无法添加颜色
        source_down.paint_uniform_color([0.4, 0.4, 0.4])  # 暂时不加载模型

    # 可视化原始数据
    draw_registration_result(source, target, np.identity(4), is_rgb)
    # draw_registration_result(source_down, target_down, np.identity(4), is_rgb)  # 采样 特征提取后的数据

    # # 场景分割（这时候用不到了）
    # 对于多类别已知模型，这里可以遍历加载所有的模型
    # 测试阶段，可以先加载单模型

    if 1:
        # 以下找到当前索引的特征对应的哪个类别  接下来就是加载对应类别的特征 对应起来
        # col = color_map[class_res]  # 不同类别的颜色  应当加上分类器之后
        # asarray(target_down.colors)[class_member_mask, :] = col
        # seg_pcd_np = array(target_down.points)[class_member_mask, :]

        # 加载对应类别的模型
        # 根据类别找到对应的模型位置
        # m_path = model_root_ + class_encode[class_res] + '/'
        # model_path = m_path + get_file_list(m_path)[0]
        # print('model_path:', model_path)

        model_trans_init = eye(4)  # 初始化模型位姿，更好地可视化
        # model_trans_init[:3, 3:] = array([1.5, 0.5, 1]).reshape(3, -1) * 0.2
        # source, source_down = load_model(model_path, model_trans_init, voxel_size)
        source, source_down, source_fpfh = prepare_dataset(model_path, voxel_size)

        # source_down.paint_uniform_color([0.4, 0.4, 0.4])  # 否则没有颜色
        # model_down_np = array(source_down.points)  # 模型点

        # 找到当前类别的特征向量(们)，然后和随影类别的模型上面的特征向量进行匹配
        # model_feature = load_model_feature(class_idx=class_res)  # 输入类别，输出模型上所有的特征  这些特征的索引就是对应点的索引

        # 格式转换 否则flann不支持
        # target_feature_seg = float32(seg_target_fpfh_T)
        target_feature = float32(target_fpfh.data.T)  # 不能用分割之后的！ 要不然哪有什么区别？？
        source_feature = float32(source_fpfh.data.T)

        # 可以替代上面的一堆  为了迎合o3d的注册函数  为了区别，这里用shource表示模型
        # _, _, source_fpfh = prepare_dataset(model_path, voxel_size)

        # 特征匹配，得到匹配的点  输入模型特征和当前分割后的特征，输出分割后对应的索引
        matches = flann.knnMatch(source_feature, target_feature, k=2)  # 场景 模型 近邻个数

        match_model_idx = []
        match_scene_idx = []

        for (m, n) in matches:  # m是最小距离  n是次小距离（或者一会加上过滤）
            # print(m.distance)
            if m.distance < 0.8 * n.distance:  # 什么原理？ 最小的<0.7次小的
                queryIdx = m.queryIdx  # 模型上
                trainIdx = m.trainIdx  # 场景中

                match_model_idx.append(queryIdx)  # 模型上
                match_scene_idx.append(trainIdx)  # 场景中 target_down的索引！

                asarray(source_down.colors)[queryIdx, :] = [0, 1, 0]  # 模型
                asarray(target_down.colors)[trainIdx, :] = [1, 0, 0]  # 场景 对应点着色
                # asarray(seg_target.colors)[trainIdx, :] = [1, 0, 0]  # 场景

        match_model_idx = array(match_model_idx)
        match_scene_idx = array(match_scene_idx)  # 场景点索引

        # 根据索引找到对应的三维坐标  匹配的坐标点 用于位姿估计
        model_paired = model_down_np[match_model_idx]
        # scene_target_np = array(seg_target.points)  # 用了分割后的 ？？？ 这里应当用全局的
        scene_paired = scene_down_np[match_scene_idx]  # match改变了索引

        # 进行粗粒度位姿估计 (毕竟就算出去也得循环)
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

    # 跳出来之后，进行对应点的可视化，一看，呦，没有多少误匹配了！ 您可真搞笑
