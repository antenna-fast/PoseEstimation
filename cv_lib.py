import cv2


# opencv lib for pose estimation

# 快速匹配
def flann_init():
    # FLANN  进行特征匹配，找到关键点对应
    FLANN_INDEX_LSH = 5
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=33,  # 12
                        key_size=20,  # 20
                        multi_probe_level=1)  # 2
    search_params = dict(checks=50)  # or pass empty dictionary  这是第二个字典，指定了索引里的树应该被递归遍历的次数

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    return flann
