# plane estimation library
# 可以归约到几何库

import numpy as np


# 从法向量和一点得到平面
def get_plan(normal, pt):
    D = -1 * np.dot(normal.T, pt)
    p = np.array([normal[0], normal[1], normal[2], D])
    return p


if __name__ == '__main__':
    n = np.array([1, 1, 1])
    pt = np.array([0, 0, 0])
    p = get_plan(n, pt)
    print(p)
