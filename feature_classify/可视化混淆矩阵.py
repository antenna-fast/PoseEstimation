import matplotlib.pyplot as plt
import numpy as np

import os

# 加载混淆矩阵
confusion = np.loadtxt('matrix.txt')
confusion = np.round(confusion, 2)

# model_list = ['armadillo', 'buddha', 'bunny', 'chinese_dragon', 'dragon', 'statuette']
model_list = ['Bear.ply', 'Dinosaur.ply', 'Face.ply', 'Claus.ply', 'Mario.ply', 'Robot.ply']
#
model_num = len(model_list)

for m in range(model_num):
    model_list[m] = model_list[m].capitalize()
    model_list[m] = os.path.splitext(model_list[m])[0]

# 热度图，后面是指定的颜色块，可设置其他的不同颜色
# plt.imshow(confusion, cmap=plt.cm.viridis)
# plt.imshow(confusion, cmap=plt.cm.cividis)
# plt.imshow(confusion, cmap=plt.cm.Blues)
plt.imshow(confusion, cmap=plt.cm.GnBu)  #
# ticks 坐标轴的坐标点
# label 坐标轴标签说明
indices = range(len(confusion))
# 第一个是迭代对象，表示坐标的显示顺序，第二个参数是坐标轴显示列表
# plt.xticks(indices, [0, 1, 2])
# plt.yticks(indices, [0, 1, 2])

plt.xticks(indices, model_list)  #
plt.yticks(indices, model_list)

plt.colorbar()

plt.xlabel('真实值', fontproperties='simsun')
plt.ylabel('预测值', fontproperties='simsun')
plt.title('混淆矩阵', fontproperties='simsun')

# plt.rcParams两行是用于解决标签不能显示汉字的问题
# plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# 显示数据
for i in range(len(confusion)):  # 第几行
    for j in range(len(confusion[i])):  # 第几列
        # first_index = first_index # 坐标中心不太好
        plt.text(i, j, confusion[j][i])
# 在matlab里面可以对矩阵直接imagesc(confusion)

plt.show()
