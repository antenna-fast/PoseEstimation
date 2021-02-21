from path_lib import *
import configparser


#
# 里面包含了场景介绍  模型数量 模型名字 真实值对应的文件
#

config_path = 'D:/SIA/Dataset/SHOT/dataset1_scenes/3D models/Stanford/Random/Dataset1_configFiles/'
cingig_list = get_file_list(config_path)
f_name = config_path + cingig_list[0]
print(f_name)

config = configparser.ConfigParser()
config.read(f_name)

section_list = config.sections()
print(section_list)

sl = section_list[0]
option_list = config.options(sl)
print(option_list)

item_list = config.items(sl)
print(item_list)
