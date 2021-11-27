import os


# 输入根目录，得到(过滤的)文件列表    不加就是默认文件夹
def get_file_list(root_path, file_type=''):
    f_list = os.listdir(root_path)
    # 如果符合后缀，就保留
    buff = []
    for f in f_list:
        if file_type in f:
            buff.append(f)
    return buff
