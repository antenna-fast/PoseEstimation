import glob
import os
import time
import copy
import numpy as np

from utils.cv_lib import *
from utils.ransac import *
from utils.o3d_pose_lib import *
from utils.o3d_impl import np2pcd, show_pcd, show_batch_pcd, draw_line_down
from utils.path_lib import *
from utils.logger_utils import get_seg_line, create_logger


# from metric.add_metric import get_add

# from joblib import dump, load  # 保存模型

# UWA dataset
# Naive RANSAC for multi-model pose estimation

# for debug
def show_points_3d(points_3d, points_3d2):
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure()
    ax1 = plt.axes(projection='3d')

    ax1.scatter3D(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], cmap='Blues')  # 绘制散点图
    ax1.scatter3D(points_3d2[:, 0], points_3d2[:, 1], points_3d2[:, 2], cmap='Red')  # 绘制散点图

    plt.show()


if __name__ == '__main__':
    color_map = {0: [1, 0, 0], 1: [0.2, 1, 0], 2: [0.1, 0.8, 0.5],
                 3: [1, 1, 0], 4: [0.2, 0.8, 1], 5: [1, 0, 1]}

    # sys parameters
    # is_vis = 1
    is_vis = 0

    is_trans_init = 0
    # is_trans_init = 1

    log_level = 'INFO'

    # dataset properties
    is_rgb = 0
    set_scene_points_threshold = 50  # we do not perform ransac if the scene
    scene = 'rs30'

    # matching parameters
    match_ratio = 0.7

    # parameters
    # voxel_size = 0.8
    voxel_size = 1.8
    # voxel_size = 2.5
    # voxel_size = 2.8

    # ransac_iter_num = 600
    # ransac_iter_num = 1800
    ransac_iter_num = 2800
    # inlier_threshold = 25
    # inlier_threshold = 30
    inlier_threshold = 50

    inlier_ratio_filter = 0.01  # filter matching result which inlier ratio is lower than this
    # inlier_ratio_filter = 0.15  # filter matching result which inlier ratio is lower than this
    # inlier_ratio_filter = 0.2  # filter matching result which inlier ratio is lower than this

    # visualization parameters
    # paint models and scene
    models_color = [0.9, 0.4, 0.1]
    scene_color = [0.1, 0.5, 0.8]

    init_model_pose = np.eye(4)
    if is_trans_init:
        init_model_pose[:3, -1] = [0, -290, 0]

    # line set color
    inlier_lineset_color = np.array([0, 0.1, 1])
    outlier_lineset_color = np.array([1, 0.1, 0])

    # data path
    root_path = '/Users/aibee/Downloads/Dataset/3DDataset/6DPose/UWA'
    model_path_list = sorted(get_file_list(os.path.join(root_path, 'model')))
    scene_path_list = sorted(get_file_list(os.path.join(root_path, 'scene')))
    gt_root = os.path.join(root_path, 'gt', 'GroundTruth_3Dscenes')

    # name mapping
    gt_to_model_name = {'chicken': 'chicken_high',
                        'parasaurolophus': 'parasaurolophus_high',
                        'T-rex': 'T-rex_high',
                        'chef': 'cheff'}
    model_name_to_gt = {'chicken_high': 'chicken',
                        'parasaurolophus_high': 'parasaurolophus',
                        'T-rex_high': 'T-rex',
                        'cheff': 'chef'}

    pose_ransac_result = {}  # {scene: {model_gt: res_ransac}}
    pose_icp_result = {}  # {scene: {model_gt: icp_ransac}}
    inlier_ratio_result = {}  # {scene: {model_gt: inlier ratio}}
    pose_ransac_result_path = os.path.join(root_path, 'predict_result', 'uwa_global_pose_ransac.npy')
    pose_icp_result_path = os.path.join(root_path, 'predict_result', 'uwa_global_pose_icp.npy')
    inlier_ratio_result_path = os.path.join(root_path, 'predict_result', 'uwa_global_inlier_ratio.npy')

    # feature matching
    matcher = FeatureMatch(feature_dim=33)

    # logger path
    logger_path = os.path.join(root_path, 'uwa_global_pose.log')
    # logger = create_logger(logger_path, log_level='INFO')
    logger = create_logger(logger_path, log_level=log_level)
    logger.info('root_path: '.format(root_path))

    logger.info('PARAMETER ')
    logger.info('match ratio: {}'.format(match_ratio))

    # load all model to ram, for feature matching
    logger.info('loading models ... ')
    model_info = {}
    for model_path in model_path_list:
        model_path = os.path.join(root_path, 'model', model_path)
        # load each model to RAM buffer
        model, model_down, model_down_fpfh = prepare_dataset(model_path, voxel_size=voxel_size, trans=init_model_pose)
        model_name = os.path.basename(model_path).split('.')[0]
        model_info[model_name] = {'model': model,
                                  'model_down': model_down,
                                  'model_down_np': np.array(model_down.points),
                                  'model_down_fpfh': np.float32(model_down_fpfh.data.T)}
    num_models = len(model_path_list)
    num_scenes = len(scene_path_list)

    # POSE ESTIMATION START
    s_time = time.time()  # sys start
    for s in range(num_scenes):
        scene_path = scene_path_list[s]
        scene_name = scene_path.split('.')[0]
        # print('processing scene {}: [{}/{}] ... '.format(scene_path, s + 1, num_scenes))
        logger.info(get_seg_line(val='processing scene {}: [{}/{}] '.format(scene_path, s + 1, num_scenes), seg_len=10, seg_marker='*'))

        # if scene not in scene_name:
        #     continue

        # init result buffer
        pose_ransac_result[scene_name] = {}
        pose_icp_result[scene_name] = {}

        # transformed model list to visualize current scene's result
        model_list = []

        # load gt of current scene
        gt_files = glob.glob(gt_root + '/*' + scene_name + '.xf')  # gt in current scene
        gts = {}
        gt_models = []
        for g in gt_files:
            gt_model_name = os.path.basename(g).split('-rs')[0]
            gt_models.append(gt_model_name)
            gts[gt_model_name] = np.loadtxt(g)

        # load and pre-process scene
        scene_path = os.path.join(root_path, 'scene', scene_path)
        scene, scene_down, scene_down_fpfh = prepare_dataset(scene_path, voxel_size=voxel_size, trans=np.eye(4))
        scene_down_np = np.array(scene_down.points)  # np format
        scene_down_fpfh = np.float32(scene_down_fpfh.data.T)
        # pain point cloud
        if not is_rgb:
            scene.paint_uniform_color(scene_color)  # 否则无法添加颜色
            scene_down.paint_uniform_color(scene_color)
        if is_vis:  # 可视化加载模型和场景的效果，查看初始位姿
            draw_registration_result(scene_down, scene_down, np.identity(4), win_name=scene_name)  # 采样 特征提取后的数据

        # while len(scene_down_np) > set_scene_points_threshold:  # to guarantee recall
        if 1:
            # iter all possible models
            for i, model_name_rt in enumerate(model_info):  # get key
                logger.info('matching model {}: [{}/{}] ... '.format(model_name_rt, i + 1, num_models))
                model_rt = model_info[model_name_rt].get('model')
                model_down_rt = model_info[model_name_rt].get('model_down')
                model_down_np_rt = model_info[model_name_rt].get('model_down_np')
                model_down_fpfh_rt = model_info[model_name_rt].get('model_down_fpfh')

                if not is_rgb:
                    model_rt.paint_uniform_color(models_color)  # 否则无法添加颜色
                    model_down_rt.paint_uniform_color(models_color)  # 否则无法添加颜色

                # matching test: take scene as the transform of model
                # scene_path = os.path.join(root_path, 'scene', scene_path)
                # trans = np.eye(4)
                # trans[:, 3] = [5, 50, 0, 1]
                # scene, scene_down, scene_down_fpfh = \
                #     0, copy.deepcopy(model_down_rt).transform(trans), model_down_fpfh_rt
                # matching test OK, DO NOT put transform within deepcopy!

                # model-scene feature matching
                match_model_idx, match_scene_idx = matcher.flann_matching(model_down_fpfh_rt, scene_down_fpfh, ratio=match_ratio)

                # 根据匹配的索引找到对应的三维坐标  匹配的坐标点用于位姿估计求解
                model_paired = model_down_np_rt[match_model_idx]
                scene_paired = scene_down_np[match_scene_idx]  # 此时，match不改变索引

                if is_vis:  # 可视化匹配的对应点，对model和scene添加不同颜色
                    model_pcd = np2pcd(model_paired, color=[0, 1, 0])
                    scene_pcd = np2pcd(scene_paired, color=[1, 0, 0])
                    show_batch_pcd(pcd_in=[model_pcd, scene_pcd], win_name=scene_name + '_pairedPCD')

                # single model ransac
                res_ransac, loss, inlier_buff, outlier_buff, inlier_ratio = \
                    ransac_pose(model_paired, scene_paired, iter_num=ransac_iter_num, inlier_threshold=inlier_threshold)
                # res_ransac = execute_global_registration(model_down_rt, scene_down, model_down_fpfh_rt, scene_down_fpfh, voxel_size)
                # res_ransac = res_ransac.transformation  # 库函数的
                logger.info('RANSAC coarse pose-estimation down')
                logger.debug('res_ransac: \n {}'.format(res_ransac))

                if inlier_ratio < inlier_ratio_filter:
                    logger.info('No model matched in current scene, inlier ratio:{:.4f} ... '.format(inlier_ratio))
                    continue
                logger.info('inlier_ratio: {:.4f}'.format(inlier_ratio))  # beyond inlier ratio

                if is_vis:  # vis coarse registration result
                    draw_registration_result(model_down_rt, scene_down, res_ransac, scene_name + '_RansacGlobal')
                    # axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                    inlier_line_set = get_line_set(model_paired, scene_paired, inlier_buff, inlier_buff, inlier_lineset_color)
                    outlier_line_set = get_line_set(model_paired, scene_paired, outlier_buff, outlier_buff, outlier_lineset_color)
                    draw_line_down(model_down_rt, scene_down, inlier_line_set, outlier_line_set, win_name=scene_name + '_LineSet')

                # ICP
                res_icp = refine_registration(model_down_rt, scene_down, res_ransac, voxel_size)
                res_icp = res_icp.transformation
                if is_vis:  # visualize icp result
                    draw_registration_result(model_down_rt, scene_down, res_icp, scene_name + '_RansacGlobal+ICP')  # RANSAC + ICP

                # delete point from scene
                if model_name_rt == 'rhino':
                    continue

                # the scene may not exist current model
                try:
                    model_name_in_gt = model_name_to_gt[model_name_rt]
                    gt_rt = gts[model_name_in_gt]
                    logger.debug('gt_rt: \n{}'.format(gt_rt))
                    # get add
                    # pose_result[model_name_rt] = {scene_name: inlier_ratio}
                    pose_ransac_result[scene_name][model_name_in_gt] = res_ransac  # the same 'model_name' as in gt
                    pose_icp_result[scene_name][model_name_in_gt] = res_icp  # the same 'model_name' as in gt
                    # inlier_ratio_result[scene_name][model_name_in_gt] = inlier_ratio
                    inlier_ratio_result.setdefault(model_name_in_gt, {})
                    inlier_ratio_result[model_name_in_gt][scene_name] = inlier_ratio
                except:
                    logger.info('model {} not exist in the current scene '.format(model_name_to_gt[model_name_rt]))

                model_down_rt_trans = copy.deepcopy(model_down_rt).transform(res_ransac)
                model_list.append(model_down_rt_trans)

            if is_vis:
                show_batch_pcd(pcd_in=model_list + [scene_down], win_name=scene_name + '_GlobalMultiPoseRansac')

    logger.info('ALL SCENE DONE')

    # save metric result
    np.save(pose_ransac_result_path, pose_ransac_result)
    np.save(pose_icp_result_path, pose_icp_result)
    np.save(inlier_ratio_result_path, inlier_ratio_result)

    # inlier ratio statistic for each model
    mean_inlier_ratio = {}

    for m in inlier_ratio_result:
        inlier_ratio_model_sum = 0
        for s in inlier_ratio_result[m]:
            inlier_ratio_model_sum += inlier_ratio_result[m][s]
        mean_inlier_ratio_tmp = inlier_ratio_model_sum / len(inlier_ratio_result[m])
        mean_inlier_ratio[m] = mean_inlier_ratio_tmp
        write_line = 'model: {}  overall inlier ratio: {}'.format(m, mean_inlier_ratio_tmp)
        print(write_line)
