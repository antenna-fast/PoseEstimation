import copy
import os
import time
import numpy as np
import open3d as o3d
from joblib import dump, load  # load and save model
import glob

from utils.o3d_impl import *
from utils.ml_lib import *
from utils.cv_lib import *
from utils.o3d_pose_lib import *
from utils.ransac import *
from utils.path_lib import *

from utils.logger_utils import create_logger, get_seg_line


if __name__ == '__main__':
    # 通过实例分割结果，加载对应的模型，直接进行匹配，可并行
    color_map = {0: [1, 0, 0], 1: [0.2, 1, 0], 2: [0.1, 0.8, 0.5],
                 3: [1, 1, 0], 4: [0.2, 0.8, 1], 5: [1, 0, 1]}

    # sys parameters
    # is_vis = 1
    is_vis = 0

    is_trans_init = 0
    # is_trans_init = 1

    log_level = 'INFO'

    is_save_patch = 0

    # dataset properties
    is_rgb = 0

    # read diameter and adaptively delete corresponding points
    # diameters_path = os.path.join(, '')
    set_scene_points_threshold = 2800  # remove these points below this threshold

    # feature matching parameters
    match_ratio = 0.7
    match_num_threshold = 6  # pass condition

    voxel_size = 1.8
    # voxel_size = 1.9  # means 5cm for the dataset
    # voxel_size = 2.4
    # voxel_size = 2.8
    # voxel_size = 3  # means 5cm for the dataset

    max_nn = 100

    # pose parameter
    ransac_iter_num = 1800

    # inlier_dist_threshold = 25
    # inlier_dist_threshold = 30
    inlier_dist_threshold = 50

    # filter matching result which inlier ratio is lower than this
    # inlier_ratio_filter = 0.02  # pass condition
    inlier_ratio_filter = 0.15  # pass condition
    # inlier_ratio_filter = 0.2  # pass condition

    # machine learning parameter
    # 1. DBSCAN
    threshold_n = 100  # 通过点的个数进行滤波 每个点集最少的点数
    search_radius = 8
    min_sample = 20
    class_num = 10
    patch_len_threshold = 500  # pass condition

    # classifier
    classifier = 'RandomForest'
    # classifier = 'AdaBoost'

    score_threshold = 0.3  # pass condition

    # visualization parameters
    # paint models and scene
    models_color = [0.9, 0.4, 0.1]
    scene_color = [0.1, 0.5, 0.8]

    init_model_pose = np.eye(4)
    if is_trans_init:
        init_model_pose[:3, -1] = [0, -290, 0]  # set a transformation

    # line set color
    inlier_lineset_color = np.array([0, 0.1, 1])
    outlier_lineset_color = np.array([1, 0.1, 0])

    # data path
    root_path = '/Users/aibee/Downloads/Dataset/3DDataset/6DPose/UWA'
    model_path_list = sorted(get_file_list(os.path.join(root_path, 'model')))
    scene_path_list = sorted(get_file_list(os.path.join(root_path, 'scene')))
    gt_root = os.path.join(root_path, 'gt', 'GroundTruth_3Dscenes')

    # machine learning path
    classifier_path = os.path.join(root_path, 'model_lib', classifier + '_uwa.clf')
    clf = load(classifier_path)  # load classifier
    decode_class_path = os.path.join(root_path, 'model_lib', 'decode_class.npy')
    decode_class = np.load(decode_class_path, allow_pickle=True).item()  # mapping idx into model name

    patches_path = os.path.join(root_path, 'patches')

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
    pose_ransac_result_path = os.path.join(root_path, 'predict_result', 'uwa_seg_pose_ransac.npy')
    pose_icp_result_path = os.path.join(root_path, 'predict_result', 'uwa_seg_pose_icp.npy')
    inlier_ratio_result_path = os.path.join(root_path, 'predict_result', 'uwa_seg_inlier_ratio.npy')

    # logger path
    logger_path = os.path.join(root_path, 'uwa_seg_pose.log')
    # logger = create_logger(logger_path, log_level='INFO')
    logger = create_logger(logger_path, log_level=log_level)
    logger.info('root_path: '.format(root_path))
    logger.info('PARAMETER ')
    logger.info('match ratio: {}'.format(match_ratio))

    # load all model to ram, for feature matching
    logger.info('loading models ... ')

    # feature matching
    matcher = FeatureMatch(feature_dim=33)
    s_time = time.time()

    # overlap points delete
    point_matcher = FeatureMatch(feature_dim=3)

    # prepare models library
    logger.info('loading models ... ')
    model_info = {}  # key=model_name, value={model, model_down, model_down_np, model_down_fpfh}
    for i, model_path in enumerate(model_path_list):
        model_path = os.path.join(root_path, 'model', model_path)
        # load each model to RAM buffer, to accelerate the following process
        model_name = os.path.basename(model_path).split('.')[0]
        model, model_down, model_down_fpfh = prepare_dataset(model_path, voxel_size=voxel_size, trans=init_model_pose)
        model_info[model_name] = {'model': model,
                                  'model_down': model_down,
                                  'model_down_np': np.array(model_down.points),
                                  'model_down_fpfh': np.float32(model_down_fpfh.data.T)}
    num_models = len(model_path_list)
    num_scenes = len(scene_path_list)

    # scene_path = root_path + 'model/parasaurolophus_high.ply'  # 直接使用模型的对齐看看！！！
    # scene_path = os.path.join(root_path, 'scene', 'rs1.ply')  # 遮挡  分开了
    # scene_path = os.path.join(root_path, 'scene', 'rs2.ply')  # 有粘连
    # scene_path = os.path.join(root_path, 'scene', 'rs4.ply')  # 遮挡

    s_time = time.time()

    # 场景数据加载
    patch_idx = 0
    for s in range(num_scenes):
        scene_path = scene_path_list[s]
        scene_name = scene_path.split('.')[0]

        # if not 'rs30' in scene_name:
        # if 'rs30' > scene_name:
        #     continue

        # logger.info('processing scene {}: [{}/{}] ... '.format(scene_path, s + 1, num_scenes))
        logger.info(get_seg_line(val='processing scene {}: [{}/{}] '.format(scene_path, s + 1, num_scenes), seg_len=10, seg_marker='*'))

        # init result buffer
        pose_ransac_result[scene_name] = {}
        pose_icp_result[scene_name] = {}

        # transformed model list
        model_list = []

        # load gt of current scene
        gt_files = glob.glob(gt_root + '/*' + scene_name + '.xf')  # gt in current scene
        gts = {}
        gt_models = []
        for g in gt_files:
            gt_model_name = os.path.basename(g).split('-rs')[0]
            gt_models.append(gt_model_name)
            gts[gt_model_name] = np.loadtxt(g)

        # 1. load and pre-process scene
        scene_path = os.path.join(root_path, 'scene', scene_path)
        scene, scene_down, scene_down_fpfh = prepare_dataset(scene_path, voxel_size=voxel_size)
        scene_down_np = np.array(scene_down.points)  # np format scene points
        scene_down_fpfh = np.float32(scene_down_fpfh.data.T)
        # scene, scene_down, target_fpfh = prepare_dataset(scene_path, voxel_size)
        if not is_rgb:
            scene.paint_uniform_color(scene_color)  # 否则无法添加颜色
            scene_down.paint_uniform_color(scene_color)
        # if is_vis:  # 可视化scene
        #     draw_registration_result(scene_down, scene_down, np.identity(4), win_name=scene_name)  # 采样 特征提取后的数据

        # 2. point cloud segmentation using DBSCAN
        seg_res = dbscan_segment(scene_down_np, search_radius, min_sample, threshold_n, class_num)
        logger.info('num object cluster: {}'.format(len(seg_res)))

        # perform patch pose, mask is corresponding to scene_down
        for class_member_mask in seg_res:  # for each point patch, get class label
            # if 1:
            inlier_ratio = 1
            patch_recall_num = 2
            current_recall_idx = 0
            while len(class_member_mask) > patch_len_threshold and inlier_ratio > inlier_ratio_filter:  # the mask will be update later
                current_recall_idx += 1
                if current_recall_idx > patch_recall_num:
                    break

                print('len_mask: {}'.format(len(class_member_mask)))
                scene_patch = scene_down.select_by_index(class_member_mask)
                scene_patch_np = np.array(scene_patch.points)  # np format patch for matching

                if is_save_patch:
                    print('saving patch .. ')
                    patch_idx += 1
                    patch_file_name = os.path.join(patches_path, str(patch_idx) + '.ply')
                    o3d.io.write_point_cloud(patch_file_name, scene_patch)

                # logger.info('patch len: {}'.format(len(scene_patch_np)))
                if is_vis:
                    show_batch_pcd(pcd_in=scene_patch, win_name=scene_name + '_SegResult')  # 显示当前分割出来的

                # 3. extract fpfh local feature of segmented scene patch, for feature matching
                seg_scene_patch_fpfh = get_fpfh(scene_patch, voxel_size, max_nn=max_nn)

                # 4. point-wise patch classification
                res = clf.predict(seg_scene_patch_fpfh)

                # 统计res里面最多的元素，计算类别并输出
                model_name_idx_res, score_res = get_class_res(res)  # get the final class result

                model_name_res = decode_class.get(model_name_idx_res)
                logger.info('class_result:{0}  score:{1:.4f}'.format(model_name_res, score_res))

                # load corresponding model information according to classification result
                model_down_rt = model_info[model_name_res]['model_down']
                model_down_rt_np = np.array(model_down_rt.points)  # np format
                model_down_fpfh_rt = model_info[model_name_res]['model_down_fpfh']

                if not is_rgb:
                    model_down_rt.paint_uniform_color(models_color)  # 否则没有/无法paint颜色

                # 5. feature matching
                # 输入模型特征和当前分割后的特征，输出 model-scene patch之间匹配的索引
                match_model_idx, match_scene_patch_idx = matcher.flann_matching(model_down_fpfh_rt, seg_scene_patch_fpfh, ratio=match_ratio)
                # logger.info('match idx len: {}'.format(len(match_model_idx)))

                if len(match_model_idx) < match_num_threshold:
                    logger.info('matched number of key points is not enough to estimate 6DoF pose .. continue')
                    continue  # cause loop

                # add different colors to model and scene
                np.asarray(model_down_rt.colors)[match_model_idx, :] = [0, 1, 0]  # 模型
                np.asarray(scene_patch.colors)[match_scene_patch_idx, :] = [1, 0, 0]  # 分割后的场景  但没有进行可视化

                # 根据索引找到对应的三维坐标 用于位姿估计
                model_paired = model_down_rt_np[match_model_idx]
                scene_patch_paired = scene_patch_np[match_scene_patch_idx]

                if is_vis:  # 可视化匹配的对应点，对model和scene添加不同颜色
                    model_pcd = np2pcd(model_paired, color=[0, 1, 0])
                    scene_pcd = np2pcd(scene_patch_paired, color=[1, 0, 0])
                    # show_batch_pcd(pcd_in=[model_pcd, scene_pcd], win_name=scene_name + '_MatchedPoints')

                # 6. RANSAC coarse pose estimation
                # inlier outlier是相对于输入数据 model_paired scene_paired的索引
                res_ransac, loss, inlier_buff, outlier_buff, inlier_ratio = ransac_pose(model_paired, scene_patch_paired, ransac_iter_num, inlier_dist_threshold)
                # res_ransac = execute_global_registration(model_down_rt, scene_patch, source_fpfh, target_fpfh, voxel_size)
                # res_ransac = res_ransac.transformation  # 库函数的
                logger.info('RANSAC coarse pose-estimation down')
                # print('res_ransac:\n', res_ransac)

                # if inlier_ratio < inlier_ratio_filter:
                #     logger.info('No model matched in current scene, inlier ratio:{:.4f} ... '.format(inlier_ratio))
                    # TODO: in this case, we need to recall it in a greedy way
                    # continue  # cause loop
                if inlier_ratio > inlier_ratio_filter:
                    patch_recall_num = 1
                logger.info('inlier_ratio: {:.4f}'.format(inlier_ratio))  # beyond inlier ratio

                if is_vis:
                    # axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20.1, origin=[0, 0, 0])
                    # get inlier and outlier lin-set to visualize RANSAC matching result
                    # inlier_line_set = get_line_set(model_paired, scene_patch_paired, inlier_buff, inlier_buff, inlier_lineset_color)
                    inlier_line_set = get_line_set(model_paired, scene_patch_paired, inlier_buff, inlier_buff, inlier_lineset_color)
                    outlier_line_set = get_line_set(model_paired, scene_patch_paired, outlier_buff, outlier_buff, outlier_lineset_color)
                    # line-set
                    # when visualization, we use full scene to see matching result
                    draw_line_down(model_down_rt, scene_down, inlier_line_set, outlier_line_set, win_name=scene_name + '_DrawLine')
                    # RANSAC pose estimation result
                    draw_registration_result(model_down_rt, scene_down, res_ransac, scene_name + '_OursCoarse')  # 粗配准后

                # 7. ICP pose refine
                res_icp = refine_registration(model_down_rt, scene_patch, res_ransac, voxel_size)
                res_icp = res_icp.transformation
                # if is_vis:  # visualize icp result
                #     draw_registration_result(model_down_rt, scene_down, res_icp, scene_name + '_Ours+ICP')  # 粗配准后

                # result evaluation according to gt
                # if model_name_res == 'rhino':
                #     continue

                # the scene may not exist current model
                try:
                    model_name_in_gt = model_name_to_gt[model_name_res]
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
                    logger.info('model {} not exist in the current scene '.format(model_name_res))

                model_down_rt_trans = copy.deepcopy(model_down_rt).transform(res_ransac)
                model_down_rt_trans_np = np.array(model_down_rt_trans.points)
                model_list.append(model_down_rt_trans)

                # to remove mask in the scene down
                nbrs = NearestNeighbors(n_neighbors=1).fit(scene_down_np)  # in all scene
                distances, indices = nbrs.kneighbors(model_down_rt_trans_np)  # return idx in scene_down
                distances = distances.flatten()
                overlap_scene_idx = list(indices[np.where(distances < set_scene_points_threshold)[0]].flatten())
                print('len overlap: {}'.format(len(overlap_scene_idx)))
                # delete estimated model's nerast index from scene_down
                # remainder
                class_member_mask = list(set(class_member_mask) - (set(class_member_mask) & set(overlap_scene_idx)))

        if is_vis:
            show_batch_pcd(pcd_in=model_list + [scene_down], win_name=scene_name + '_SegMultiPoseRansac')

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
