import os


# 输入根目录，得到(过滤的)文件列表 不加就是默认全部
def get_file_list(root_path, filter='.ply'):
    f_list = os.listdir(root_path)
    # 如果符合后缀，就保留
    buff = []
    for f in f_list:
        if filter in f:
            buff.append(f)
    return buff


def get_parent_dir(given_path):
    return '/'.join(given_path.split('/')[:-1])
