from ..utils.o3d_impl import *


# Demo
# 绘制直线

# 点
points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
          [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
# 线段索引
lines = [[0, 1], [0, 2], [1, 3], [2, 3],
         [4, 5], [4, 6], [5, 7], [6, 7],
         [0, 4], [1, 5], [2, 6], [3, 7]]

colors = [[i/12, i/12, 0] for i in range(len(lines))]


line_set = draw_line(points, lines, colors)

o3d.visualization.draw_geometries([line_set])
