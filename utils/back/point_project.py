# from numpy import *
import numpy as np


# 投影点[XYZ] 投影线[ABC]
def pt_to_line(p, l):
    l_n = np.array([l[0], l[1]])

    # 求参数t
    t = -(l[2] + l[0] * p[0] + l[1] * p[1]) / (l[0] * l_n[0] + l[1] * l_n[1])
    print('t:', t)

    x = p + t * l_n
    print(x)

    return x


# 输入:点(nx3) 平面 平面法向量
# 输出:投影点
def pt_to_plane(pts_array, plan, p_n):
    # 参数t
    res = []
    for pts in pts_array:  # 这里可以矩阵化  加速计算
        t = -1 * (plan[0] * pts[0] + plan[1] * pts[1] + plan[2] * pts[2] + plan[3]) / (plan[0] * p_n[0] + plan[1] * p_n[1] + plan[2] * p_n[2])
        x = pts + t * p_n

        res.append(x)

    res = np.array(res)
    return res
