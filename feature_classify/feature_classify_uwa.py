from fpfh_pose_2_syn_n import *
from np_lib import *

# ML lib
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

import time
from joblib import dump, load  # 保存模型  pickle lib

import random as rsample  # 产生随机数

# 1.加载模型 提特征

root_path = 'D:/SIA/Dataset/UWA/'
# source是模型 target是场景

# class_encode = os.listdir(root_path + 'model/unzip')
class_encode = ['cheff.ply', 'chicken_high.ply', 'parasaurolophus_high.ply', 'rhino.ply', 'T-rex_high.ply']
print(class_encode)

# scene_path = root_path + 'scene/unzip/rs1.ply'
# model_path = root_path + 'model/unzip/cheff.ply'

# 分类器保存位置
model_save_path = 'model_lib/adaboost_uwa.clf'

model_num = len(class_encode)
# print(model_list)

# 保存地址
feature_bag_path = feature_bag_root = 'D:/SIA/Dataset/FeatureBag/UWA/'

# AdaBoost分类器
# clf = AdaBoostClassifier(n_estimators=160,
#                          random_state=1,
#                          learning_rate=0.8)

# 随机森林
clf = RandomForestClassifier(max_depth=20, random_state=0)

# color encode
# ['armadillo', 'buddha', 'bunny', 'chinese_dragon', 'dragon', 'statuette']
# ['bear.ply', 'dinasaur.ply', 'face.ply', 'ghost.ply', 'mboy.ply', 'robot.ply']
# [红 绿 蓝 黄 天蓝 洋红]
class_map = {0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1],
             3: [1, 1, 0], 4: [0, 1, 1], 5: [1, 0, 1]}

voxel_size = 2.8
sample_num = 4300  # 所有模型上 特征点最小数量  调整

# TODO：根据个数，生成均匀采样序列  用sanple库即可（无重复）
train = 1

model_idx_list = [1, 2]  # 都有哪些模型？编码后的

if train:
    # 在所有的模型上  构建特征
    for i in range(model_num):
        print('_____')

        model_name = class_encode[i]
        print('model_name: {0}  idx: {1}'.format(model_name, i))

        model_path = root_path + 'model/unzip/' + class_encode[i]
        # print(file_path)

        pcd, pcd_down, pcd_fpfh = read_pcd(model_path, voxel_size)

        fpfh_data = pcd_fpfh.data.T
        print('pcd_fpfh:', fpfh_data.shape)

        # 保存特征
        save_path = feature_bag_path + os.path.splitext(model_name)[0] + '.npy'
        np.save(save_path, fpfh_data)

        # 叠加起来一起训练
        data_len = len(fpfh_data)
        label_temp = ones(data_len).reshape(-1, 1) * i  # len行1列

        # 生成采样索引（解决样本不平衡）  衰减第0个的？
        sample_idx = rsample.sample(list(range(data_len)), sample_num)

        if i == 0:
            feature_buff = fpfh_data[sample_idx]  # 避免类别不平衡  超参数？
            label_buff = label_temp[sample_idx]

        else:
            feature_buff = r_[feature_buff, fpfh_data[sample_idx]]  # 叠加特征
            label_buff = r_[label_buff, label_temp[sample_idx]]  # 叠加标签

        print()

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

# 查看分类效果
# 可视化出来  相关矩阵？？或者是

for i in range(model_num):
    # i = 1
    model_name = class_encode[i]
    print('model_name: {0}  idx: {1}'.format(model_name, i))
    model_path = root_path + 'model/unzip/' + model_name

    pcd, pcd_down, pcd_fpfh = read_pcd(model_path, voxel_size)

    fpfh_data = pcd_fpfh.data.T
    # print('pcd_fpfh:', fpfh_data.shape)

    # 以这些特征作为训练数据X 标签为模型对应的类别   训练
    s_time = time.time()
    # res = clf.predict([fpfh_data[10]])  # 第几个模型对应第几个类别
    res = clf.predict(fpfh_data)  # 第几个模型对应第几个类别
    delta_t = time.time() - s_time
    print('predicted in {0:.3f} second'.format(delta_t))

    # 运行时，场景中的特征预测使用单个点
    # Return the mean accuracy on the given test data and labels.
    class_res, score = get_class_res(res)
    print('class_label={0}  score={1}'.format(class_res, score))

    print()