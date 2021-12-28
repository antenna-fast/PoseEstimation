"""
Author: ANTenna on 2021/12/25 5:50 下午
aliuyaohua@gmail.com

Description:

"""
import copy
import os
import sys
import numpy as np

sys.path.insert(0, os.getcwd() + '/..')
from utils.o3d_impl import *
from utils.o3d_pose_lib import prepare_dataset
from utils.path_lib import get_file_list


if __name__ == '__main__':

    is_rgb = 0
    models_color = [0.9, 0.4, 0.1]
    scene_color = [0.1, 0.5, 0.8]

    root_path = '/Users/aibee/Downloads/Dataset/3DDataset/6DPose/UWA'
    model_path_list = sorted(get_file_list(os.path.join(root_path, 'model')))

    gt_path = os.path.join(root_path, 'gt')
    # pred_ransac_path = os.path.join(root_path, 'predict_result', 'uwa_global_pose_ransac.npy')
    # pred_icp_path = os.path.join(root_path, 'predict_result', 'uwa_global_pose_icp.npy')

    pred_ransac_path = os.path.join(root_path, 'predict_result', 'uwa_seg_pose_ransac.npy')
    pred_icp_path = os.path.join(root_path, 'predict_result', 'uwa_seg_pose_icp.npy')

    pred_ransac_pose = np.load(pred_ransac_path, allow_pickle=True).item()  # {scene: {model_name: pose}}
    pred_icp_pose = np.load(pred_icp_path, allow_pickle=True).item()

    model_recall = 0
    voxel_size = 2.8

    gt_to_model_name = {'chicken': 'chicken_high',
                        'parasaurolophus': 'parasaurolophus_high',
                        'T-rex': 'T-rex_high',
                        'chef': 'cheff'}
    model_name_to_gt = {'chicken_high': 'chicken',
                        'parasaurolophus_high': 'parasaurolophus',
                        'T-rex_high': 'T-rex',
                        'cheff': 'chef'}

    init_model_pose = np.eye(4)

    # load all model to ram, for feature matching
    # logger.info('loading models ... ')
    model_info = {}
    for model_path in model_path_list:
        model_path = os.path.join(root_path, 'model', model_path)
        # load each model to RAM buffer
        model, model_down, model_down_fpfh = prepare_dataset(model_path, voxel_size=voxel_size, trans=init_model_pose)
        model_name = os.path.basename(model_path).split('.')[0]
        if not is_rgb:
            model.paint_uniform_color(models_color)  # 否则无法添加颜色
            model_down.paint_uniform_color(models_color)

        model_info[model_name] = {'model': model,
                                  'model_down': model_down,
                                  'model_down_np': np.array(model_down.points),
                                  'model_down_fpfh': np.float32(model_down_fpfh.data.T)}
    num_models = len(model_path_list)
    print('loaded models ')

    # ransac_pose
    for s in pred_ransac_pose:
        # load scene
        scene_path = os.path.join(root_path, 'scene', s + '.ply')
        scene, scene_down, scene_down_fpfh = prepare_dataset(scene_path, voxel_size=voxel_size)
        if not is_rgb:
            scene.paint_uniform_color(scene_color)  # 否则无法添加颜色
            scene_down.paint_uniform_color(scene_color)

        # load all poses, and trans models according to its predicted pose
        model_list = []
        for m in pred_ransac_pose[s]:
            m_pose = pred_ransac_pose[s][m]

            model_name_rt = gt_to_model_name[m]  # must!
            model_down_rt = copy.deepcopy(model_info[model_name_rt]['model_down'])
            model_down_rt.transform(m_pose)
            model_list.append(model_down_rt)

        show_batch_pcd(pcd_in=[scene_down] + model_list, win_name=s + '_SegMultiResult')  # 显示当前分割出来的
