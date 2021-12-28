import copy
import os

import numpy as np
import numpy.random
import torch
from sklearn.neighbors import NearestNeighbors


def get_add(model, diameter, obj_class, sym_list, pred_pose, gt_pose):
    """
    :param model: 3D model, Nx3 points
    :param diameter: diameter of input model
    :param obj_class: object class
    :param sym_list: symmetry list
    :param pred_pose: predicted pose by algorithm
    :param gt_pose: gt pose
    :return: is pass tolerance
    """
    # transform model
    pred = np.dot(model, pred_pose[:3, :3]) + pred_pose[:3, -1]
    target = np.dot(model, gt_pose[:3, :3]) + gt_pose[:3, -1]

    if obj_class in sym_list:  # sym loss: ADD-S
        pred = pred.astype(np.float32)
        target = target.astype(np.float32)
        # build knn search tree
        nbrs = NearestNeighbors(n_neighbors=1).fit(target)
        distances, indices = nbrs.kneighbors(pred)  # for each predicted point, return idx in the TARGET
        target_matched = target[indices.reshape(-1)]
        # index of the source's nearest point in target
        dis = np.mean(np.linalg.norm((pred - target_matched), axis=1), axis=0)
    else:  # non-sym loss: ADD
        dis = np.mean(np.linalg.norm(pred - target, axis=1))

    if dis < diameter:
        print('Pass! Distance: {0}'.format(dis))
        return 1
    else:
        print('NOT Pass! Distance: {0}'.format(dis))
        return 0


# for i in range(num_objects):
#     print('Object {0} success rate: {1}'.format(objlist[i], float(success_count[i]) / num_count[i]))
#     # fw.write('Object {0} success rate: {1}\n'.format(objlist[i], float(success_count[i]) / num_count[i]))

# print('Overall success rate: {0}'.format(float(sum(success_count)) / sum(num_count)))
# fw.write('ALL success rate: {0}\n'.format(float(sum(success_count)) / sum(num_count)))
# fw.close()


if __name__ == '__main__':
    model = numpy.random.randn(10, 3)
    diameter = 1
    pred = np.eye(4)
    gt = np.eye(4)

    root_path = '/Users/aibee/Downloads/Dataset/3DDataset/6DPose/UWA'
    diam_save_path = os.path.join(root_path, 'model', 'model_diameter.npy')  # {model_name: diam}

    # model, diameter, obj_class, sym_list, pred_pose, gt_pose
    res = get_add(model=model, diameter=diameter, obj_class='1', sym_list=['1'], pred_pose=pred, gt_pose=gt)
    print('res: {}'.format(res))
