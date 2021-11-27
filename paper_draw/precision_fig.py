# 6 model上面的精度

from numpy import *
import matplotlib.pyplot as plt
import pandas as pd

# 分别加载npy，画出柱状图

# file_root = 'D:/SIA/Dataset/Stanford/ANTennnaScene/precision/'
# precision_path_ori = file_root + 'ori_syn_6model.npy'  # 两个字典
# precision_path_my = file_root + 'NB_syn_6model.npy'

scene = 'Scene1'
# scene = 'Scene2'
# scene = 'Scene31'
# scene = 'Scene32'
# scene = 'Scene37'  # 采样率  0.006

# 精确度保存路径
precision_path = 'C:/Users/yaohua-win/PycharmProjects/pythonProject/pose_sys/measure/precision/'

precision_path_ori = precision_path + 'syn_global_' + scene + '.npy'  # 两个字典
precision_path_my = precision_path + 'syn_my_' + scene + '.npy'

p_dict_1 = load(precision_path_ori, allow_pickle=True).item()  # .values()
p_dict_2 = load(precision_path_my, allow_pickle=True).item()  # .values()
print('p_dict_1:', p_dict_1)
print('p_dict_2:', p_dict_2)

s1 = sorted(p_dict_1)  # 所有的模型  存在list
print('s1:', s1)

# method_list = ['Origional', '122']
method_list = ['原始', '改进']

dict_data = {method_list[0]: [], method_list[1]: []}

# 统一模型名字
for m in s1:
    dict_data[method_list[0]].append(p_dict_1[m])
    dict_data[method_list[1]].append(p_dict_2[m])

data_len = len(s1)
index = arange(data_len)

bw = 0.3  # 偏移量

my_color = array([30, 144, 255]) / 255  # 道奇蓝
ori_color = array([255, 140, 0]) / 255   # 橙红

plt.title('bar', fontsize=16)
plt.bar(index, dict_data[method_list[0]], bw, color=ori_color)
plt.bar(index + bw, dict_data[method_list[1]], bw, color=my_color)  # 第二条

# plt.rcParams['font.sans-serif'] = ['Times New Roman']
# plt.rcParams['axes.unicode_minus'] = False

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

plt.legend(method_list)

plt.xticks(index + 0.5 * bw, s1)
plt.ylabel('Precision')
plt.show()
