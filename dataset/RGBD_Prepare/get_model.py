import open3d as o3d
import numpy as np

# 加载RGBD的数据集 并进行可视化

root_path = 'D:/SIA/Dataset/RGBD/rgbd-scenes-v2_pc/rgbd-scenes-v2/pc/'
pcd_path = root_path + '01.ply'

pcd = o3d.io.read_point_cloud(pcd_path)
pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=10))

pcd.paint_uniform_color([1, 0.4, 0.1])

# 加载标签，对不同类别的进行涂色
label_path = root_path + '01.label'
label = np.loadtxt(label_path)
print(label)

idx = np.where(label == 4)[0]

np.asarray(pcd.colors)[idx, :] = [0, 0, 1]

pcd_cap = pcd.select_by_index(idx)


def get_seg_idx():
    return 0


axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])

o3d.visualization.draw_geometries([
    # pcd,
    pcd_cap,
    axis_pcd],
    window_name='ANTenna3D',
    zoom=1,
    front=[0, -0.8, -0.6],  # 相机位置
    lookat=[0, 0, 0],  # 对准的点
    up=[0, 0.3, 0.3],  # 用于确定相机右x轴
)
