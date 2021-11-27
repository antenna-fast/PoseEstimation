from o3d_pose_lib import *

bbox_gt_color  = [1, 0, 0]

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
