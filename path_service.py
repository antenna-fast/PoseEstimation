import os


# 输入根目录，得到(过滤的)文件列表
def get_file_list(root_path):
    f_list = os.listdir(root_path)
    return f_list
