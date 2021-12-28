import os
import sys
import time
from joblib import dump, load  # 保存模型  pickle lib

import numpy as np
import random as rsample  # 产生随机数

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

sys.path.insert(0, os.getcwd() + '/..')
from utils.o3d_pose_lib import prepare_dataset
from utils.path_lib import get_file_list
from utils.ml_lib import get_class_res


def train_model(model_info, feature_path, sample_num):
    # 1. load models
    # 2. extract and save features
    # 3. uniform feature sampling
    # 4. train classifier
    # 5. save classifier
    iter_num = 10

    sample_num_tmp = sample_num
    for s in range(iter_num):
        print('training iter: [{}/{}]'.format(s+1, iter_num))
        sample_num = int(sample_num_tmp / (s+1))
        print('sample_num: {}'.format(sample_num))

        # one iteration
        for i, model_name in enumerate(model_info):
            print('model_name: {0}  idx: {1}'.format(model_name, i))
            model_down_feature = model_info[model_name]['model_down_fpfh']
            print('pcd_fpfh_shape:', model_down_feature.shape)

            # save feature
            # np.save(feature_path, model_down_feature)

            # 叠加起来, 训练多类别分类器
            data_len = len(model_down_feature)
            label_temp = np.ones(data_len).reshape(-1, 1) * i  # len行1列

            # 生成采样索引
            sample_idx = rsample.sample(list(range(data_len)), sample_num)
            # sample to avoid unbalance, and stack them
            if i == 0:  # init buffer: feature and label
                feature_buff = model_down_feature[sample_idx]
                label_buff = label_temp[sample_idx]
            else:
                feature_buff = np.concatenate((feature_buff, model_down_feature[sample_idx]), axis=0)  # 叠加特征
                label_buff = np.concatenate((label_buff, label_temp[sample_idx]), axis=0)  # 叠加标签
        print('feature_buff:', feature_buff.shape)
        label_buff = label_buff.flatten()

        # 以这些特征作为训练数据X 标签为模型对应的类别
        print('training classifier ... ')
        s_time = time.time()
        clf.fit(feature_buff, label_buff)  # 第几个模型对应第几个类别
        delta_t = time.time() - s_time
        print('fit classifier in {0:.3f} second'.format(delta_t))

        # save classifier
        print('saving model ... ')
        dump(clf, model_save_file_path)

        print('TRAIN CLASSIFIER FINISHED ')


# 查看分类效果, 可视化相关矩阵
# 测试: 加载模型，提取特征  单模型分类 OK 能够分出
def test_model(classifier_path, model_info, decode_path):
    # load class decode
    decode_class = np.load(decode_path, allow_pickle=True).item()
    print('loading classifier ... ')
    clf = load(classifier_path)  # load classifier

    # classify each model
    for i, model_name in enumerate(model_info):
        print('model_name: {0}  idx: {1}'.format(model_name, i))
        model_down_fpfh = model_info[model_name]['model_down_fpfh']
        # print('pcd_fpfh:', model_down_fpfh.shape)
        # 以这些特征作为训练数据X 标签为模型对应的类别
        s_time = time.time()
        res = clf.predict(model_down_fpfh)  # 第几个模型对应第几个类别
        delta_t = time.time() - s_time
        # print('predicted in {0:.3f} second'.format(delta_t))

        # Return the mean accuracy on the given test data and labels.
        class_res, score = get_class_res(res)
        print('predict model name: {} score: {}'.format(decode_class[class_res], score))
        print('*' * 20)


if __name__ == '__main__':
    # sys parameter
    is_train = 1
    # is_train = 0

    is_get_decode_class = 1

    # classifier = 'AdaBoost'
    classifier = 'RandomForest'

    # data path
    root_path = '/Users/aibee/Downloads/Dataset/3DDataset/6DPose/UWA'
    model_path_list = get_file_list(os.path.join(root_path, 'model'))

    # model feature path
    object_feature_path = os.path.join(root_path, 'feature/model_feature')
    if not os.path.exists(object_feature_path):
        os.makedirs(object_feature_path)

    # machine learning path
    # 分类器保存位置
    model_root_path = os.path.join(root_path, 'model_lib')
    if not os.path.exists(model_root_path):
        os.makedirs(model_root_path)
    model_save_file_path = os.path.join(model_root_path, classifier + '_uwa.clf')

    # color encode  [红 绿 蓝 黄 天蓝 洋红]
    class_map = {0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1],
                 3: [1, 1, 0], 4: [0, 1, 1], 5: [1, 0, 1]}

    voxel_size = 2.8
    sample_num = 4300  # 所有模型上均匀采样 特征点最小数量
    init_model_pose = np.eye(4)

    if classifier == 'AdaBoost':  # AdaBoost分类器
        ada_estimator = 160
        random_state = 1
        lr = 0.8
        clf = AdaBoostClassifier(n_estimators=ada_estimator, random_state=random_state, learning_rate=lr)
    elif classifier == 'RandomForest':  # random forest
        max_depth = 30
        random_state = 0
        n_estimators = 100
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    else:
        raise KeyError('ERROR CLASSIFIER: {}'.format(classifier))

    # load object models
    print('loading models ...')
    model_info = {}
    decode_class = {}
    for i, model_path in enumerate(model_path_list):
        model_path = os.path.join(root_path, 'model', model_path)
        # load each model to RAM buffer
        model, model_down, model_down_fpfh = prepare_dataset(model_path, voxel_size=voxel_size, trans=init_model_pose)
        model_name = os.path.basename(model_path).split('.')[0]
        model_info[model_name] = {'model': model,
                                  'model_down': model_down,
                                  'model_down_np': np.array(model_down.points),
                                  'model_down_fpfh': np.float32(model_down_fpfh.data.T)}
        decode_class[i] = model_name
    num_models = len(model_path_list)

    if is_get_decode_class:
        print('decode_class: \n', decode_class)
        decode_class_path = os.path.join(root_path, 'model_lib', 'decode_class.npy')
        np.save(decode_class_path, decode_class)
        print('class decode information saved: {}'.format(decode_class_path))

    if is_train:  # train classifier
        print('training classifier ... ')
        train_model(model_info, object_feature_path, sample_num)
    else:  # test classifier
        print('testing classifier ... ')
        test_model(classifier_path=model_save_file_path, model_info=model_info, decode_path=decode_class_path)
        print()
