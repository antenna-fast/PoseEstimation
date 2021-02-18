from fpfh_pose_2 import *
from path_service import *
from o3d_impl import *

from np_lib import *

# ML lib
from sklearn.ensemble import AdaBoostClassifier
from sklearn.manifold import TSNE

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.ticker import NullFormatter

import time

from joblib import dump, load  # 保存模型

# 根据全局FPFH进行点云分类

# 加载所有模型点云
# 提取特征
# 训练分类器（类别标注）
# 加载场景点云，提取特征，进行特征分类(得到点云分割)
# 利用分割结果进行匹配并进一步配准


# 读取单个点云
# 下采样，计算FPFH
def read_pcd(model_path, voxel_size):
    pcd = o3d.io.read_point_cloud(model_path)

    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    pcd_down, pcd_fpfh = preprocess_point_cloud(pcd, voxel_size)

    return pcd, pcd_down, pcd_fpfh


# 1.加载模型 提特征
model_root = 'D:/SIA/Dataset/Stanford/3D models/'
# model_list = get_file_list(model_root)
model_list = ['armadillo', 'buddha', 'bunny', 'chinese_dragon', 'dragon', 'statuette']
model_num = len(model_list)
print(model_list)

feature_bag_path = 'D:/SIA/Dataset/FeatureBag/StanfordFPFH/'

# 分类器
clf = AdaBoostClassifier(n_estimators=260,
                         random_state=1,
                         learning_rate=0.8)

# color encode
# ['armadillo', 'buddha', 'bunny', 'chinese_dragon', 'dragon', 'statuette']
# [红 绿 蓝 黄 天蓝 洋红]
class_map = {0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1],
             3: [1, 1, 0], 4: [0, 1, 1], 5: [1, 0, 1]}

voxel_size = 0.005
train = 1

if train:
    # 在所有的模型上  构建特征
    for i in range(model_num):
        model_name = model_list[i]
        print('model_name: {0}  idx: {1}'.format(model_name, i))

        file_name = get_file_list(model_root + model_name)[0]
        file_path = model_root + model_name + '/' + file_name
        # print(file_path)

        pcd, pcd_down, pcd_fpfh = read_pcd(file_path, voxel_size)

        fpfh_data = pcd_fpfh.data.T
        print('pcd_fpfh:', fpfh_data.shape)

        # 保存特征
        save_path = feature_bag_path + model_name + '.npy'
        save(save_path, fpfh_data)

        # 叠加起来一起训练
        data_len = len(fpfh_data)
        label_temp = ones(data_len).reshape(-1, 1) * i

        if i == 0:
            feature_buff = fpfh_data[:1900]  # 避免类别不平衡
            label_buff = label_temp[:1900]

        else:
            feature_buff = r_[feature_buff, fpfh_data[:1900]]  # 叠加特征
            label_buff = r_[label_buff, label_temp[:1900]]  # 叠加标签

    print('feature_buff:', feature_buff.shape)
    label_buff = label_buff.flatten()

    # 以这些特征作为训练数据X 标签为模型对应的类别  直接在这里面,是不是上次的被覆盖了？（所以每次都是最后一个类别）
    s_time = time.time()
    clf.fit(feature_buff, label_buff)  # 第几个模型对应第几个类别
    delta_t = time.time() - s_time
    print('fit ok in {0:.2f} second'.format(delta_t))

    # 保存模型
    dump(clf, 'model_lib/adaboost.clf')

# 测试
# 加载模型，提取特征  单模型分类 OK 能够分出
# 如何应对杂乱场景多模型呢？

# 加载模型
clf = load('model_lib/adaboost.clf')

# res = clf.predict([fpfh_data[0]])
# print('res:', res)

i = 3
model_name = model_list[i]
print('model_name: {0}  idx: {1}'.format(model_name, i))

file_name = get_file_list(model_root + model_name)[0]
file_path = model_root + model_name + '/' + file_name

pcd, pcd_down, pcd_fpfh = read_pcd(file_path, voxel_size)

fpfh_data = pcd_fpfh.data.T
print('pcd_fpfh:', fpfh_data.shape)

data_len = len(fpfh_data)  # 所有的特征数   这种方法没有多类别能力
labels = ones(data_len) * i

# 以这些特征作为训练数据X 标签为模型对应的类别
s_time = time.time()
# res = clf.predict([fpfh_data[10]])  # 第几个模型对应第几个类别
res = clf.predict(fpfh_data)  # 第几个模型对应第几个类别
delta_t = time.time() - s_time
print('predicted in {0:.3f} second'.format(delta_t))

# 最有可能的类别
# print('argmax res:', argmax(bincount(res)))

# Return the mean accuracy on the given test data and labels.
score = clf.score(fpfh_data, labels)  # 给定预测特征 得到效果
print('score:', score)

# 运行时，场景中的特征预测使用单个点
idx_temp = 0
for i in range(6):
    s = sum(res == i)  # 统计数量 对全部类别 (整个模型)
    print(s / data_len)

    # 某一类别的mask
    # idx_list = (res == i)
    # 对应的类别涂上对应的颜色 好像没什么意义

# print('res:', res)
