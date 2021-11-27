from fpfh_pose_2_syn_n import *
from path_lib import *
from o3d_impl import *
from np_lib import *

from sklearn.ensemble import AdaBoostClassifier

from joblib import dump, load  # 保存模型  pickle lib
import time

import random as rsample  # 产生随机数

# 根据全局FPFH进行点云分类

# 1.加载模型 提特征
model_root = 'D:/SIA/Dataset/Stanford/3D models/'
# model_list = ['armadillo', 'buddha', 'bunny', 'chinese_dragon', 'dragon', 'statuette']

model_list = get_file_list(model_root)
print(model_list)
model_num = len(model_list)

# 分类器保存位置
model_save_path = 'C:/Users/yaohua-win/PycharmProjects/pythonProject/' \
                  'pose_sys/feature_classify/stanford_syn_adaboost.clf'

# 保存地址
feature_bag_path = 'D:/SIA/Dataset/FeatureBag/StanfordFPFH/'

# 分类器
clf = AdaBoostClassifier(n_estimators=260,
                         random_state=1,
                         learning_rate=0.8)

# color encode
# [红 绿 蓝 黄 天蓝 洋红]
class_map = {0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1],
             3: [1, 1, 0], 4: [0, 1, 1], 5: [1, 0, 1]}

voxel_size = 0.006

train = 1

if train:
    # 在所有的模型上  构建特征
    for i in range(model_num):
        model_name = model_list[i]
        print('model_name: {0}  idx: {1}'.format(model_name, i))

        file_path = model_root + model_name + '/' + os.listdir(model_root+model_name)[0]
        # print(file_path)

        pcd, pcd_down, pcd_fpfh = read_pcd(file_path, voxel_size)

        fpfh_data = pcd_fpfh.data.T
        print('pcd_fpfh:', fpfh_data.shape)

        # 保存特征
        save_path = feature_bag_path + os.path.splitext(model_name)[0] + '.npy'
        save(save_path, fpfh_data)

        # 叠加起来一起训练
        data_len = len(fpfh_data)
        label_temp = ones(data_len).reshape(-1, 1) * i  # len行1列

        # 如果不截取： 全部的样本 用于训练
        feature_num = data_len

        if i == 0:
            feature_buff = fpfh_data[:feature_num]  # 避免类别不平衡  超参数？
            label_buff = label_temp[:feature_num]

        else:
            feature_buff = r_[feature_buff, fpfh_data[:feature_num]]  # 叠加特征
            label_buff = r_[label_buff, label_temp[:feature_num]]  # 叠加标签

    print('feature_buff:', feature_buff.shape)
    label_buff = label_buff.flatten()

    # 以这些特征作为训练数据X 标签为模型对应的类别  直接在这里面,是不是上次的被覆盖了？（所以每次都是最后一个类别）
    # s_time = time.time()
    clf.fit(feature_buff, label_buff)  # 第几个模型对应第几个类别
    # delta_t = time.time() - s_time
    # print('fit ok in {0:.2f} second'.format(delta_t))

    # 保存模型
    dump(clf, model_save_path)

# 测试
# 加载模型，提取特征  单模型分类 OK 能够分出
# 如何应对杂乱场景多模型呢？

# 加载模型
clf = load(model_save_path)

score_matrix = []

for i, model_name in enumerate(model_list):
    print('model_name: {0}  idx: {1}'.format(model_name, i))

    file_path = model_root + model_name + '/' + os.listdir(model_root+model_name)[0]
    pcd, pcd_down, pcd_fpfh = read_pcd(file_path, voxel_size)

    fpfh_data = pcd_fpfh.data.T
    print('pcd_fpfh:', fpfh_data.shape)

    # 以这些特征作为训练数据X 标签为模型对应的类别
    s_time = time.time()
    res = clf.predict(fpfh_data)  # 第几个模型对应第几个类别
    delta_t = time.time() - s_time
    print('predicted in {0:.3f} second'.format(delta_t))

    # 运行时，场景中的特征预测使用单个点
    # Return the mean accuracy on the given test data and labels.
    # class_res, score = get_class_res(res)
    score_list = get_class_res_static(res, model_num)
    print('class_label={0}  score={1}'.format(1, score_list))

    score_matrix.append(score_list)

score_matrix = array(score_matrix)

# 保存之
# 单独可视化出来！！！分类效果

np.savetxt('matrix.txt', score_matrix.T)
