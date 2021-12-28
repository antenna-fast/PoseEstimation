# opencv lib for pose estimation

import cv2
import numpy as np


class FeatureMatch(object):
    def __init__(self, feature_dim=33):
        # parameter
        self.feature_dim = feature_dim  # 33 for FPFH
        self.FLANN_INDEX_LSH = 5
        self.flann = self.flann_init()

    def flann_init(self):
        # FLANN  进行特征匹配，找到关键点对应
        index_params = dict(algorithm=self.FLANN_INDEX_LSH,
                            table_number=self.feature_dim,  # 12
                            key_size=20,  # 20
                            multi_probe_level=1)  # 2
        # 指定索引里的树应该被递归遍历的次数
        search_params = dict(checks=50)  # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        return flann

    def flann_matching(self, src, dst, ratio=0.65, k=2):
        matches = self.flann.knnMatch(src, dst, k=k)  # 场景 模型 近邻个数
        # perform ratio test
        match_model_idx = []
        match_scene_idx = []

        for (m, n) in matches:  # m是最小距离  n是次小距离（或者一会加上过滤）
            if m.distance < ratio * n.distance:  # 最小的<0.7次小的
                queryIdx = m.queryIdx  # 模型上
                trainIdx = m.trainIdx  # 场景中
                match_model_idx.append(queryIdx)  # 模型上
                match_scene_idx.append(trainIdx)  # 场景中
        match_model_idx = np.array(match_model_idx)
        match_scene_idx = np.array(match_scene_idx)  # 场景点索引

        return match_model_idx, match_scene_idx


if __name__ == '__main__':

    matcher = FeatureMatch(feature_dim=2)

    src_test = np.array([[0, 0],
                         [1, 0],
                         [1, 1],
                         [2, 1]], dtype=np.float32)

    dst_test = np.array([[1, 0],
                         [0, 0],
                         [2, 1],
                         [1, 1]], dtype=np.float32)

    src_idx, dst_idx = matcher.flann_matching(src_test, dst_test)

    print('src:', src_idx)
    print('dst:', dst_idx)
    print()
