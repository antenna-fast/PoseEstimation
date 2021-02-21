from copy import deepcopy

from o3d_impl import *


###################全局 粗配准 #############
# 基于RANSAC得到初始位姿  这里面集成了匹配  直接使用特征了
def execute_global_registration(source_down, target_down,
                                source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    # print(":: RANSAC registration on downsampled point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % voxel_size)
    # print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
             distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(1000, 0.999)
    )

    return result


# 快速全局配准（M估计）
def execute_fast_global_registration(source_down, target_down,
                                     source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" \
          % distance_threshold)
    result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result


# 位姿refine
# 基于ICP 精细配准
def refine_registration(source, target, result_ransac, voxel_size):
    distance_threshold = voxel_size * 0.4
    # print(":: Point-to-plane ICP registration is applied on original point")
    # print("   clouds to refine the alignment. This time we use a strict")
    # print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        # source, target, distance_threshold, result_ransac.transformation,
        source, target, distance_threshold, result_ransac,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result


################## 可视化 ####################

def draw_registration_result(source, target, transformation, is_rgb=0, w_title='3D View'):
    source_temp = deepcopy(source)
    target_temp = deepcopy(target)

    if not is_rgb:
        source_temp.paint_uniform_color([1, 0.706, 0])  # 模型 对于rgb点云不进行绘制颜色
        target_temp.paint_uniform_color([0, 0.651, 0.929])  # 场景

    source_temp.transform(transformation)  # 变换source

    # source_temp.translate([-1.0, 0, 0, 0])
    # target_temp.flip()

    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      window_name=w_title,
                                      # zoom=1,
                                      # front=[0, 0, 0.001],  # 相机位置
                                      # lookat=[0, 0, 0],  # 对准的点
                                      # up=[0, -1, 0],  # 用于确定相机右x轴

                                      zoom=1,
                                      front=[0, -1, 0.1],  # 相机位置
                                      lookat=[0, 0, 0],  # 对准的点
                                      up=[0, 1, 0.1],  # 用于确定相机右x轴
                                      # point_show_normal=True
                                      )


# 场景分割效果
def draw_segmentation_result(target, is_rgb=0):
    target_temp = deepcopy(target)

    if not is_rgb:
        target_temp.paint_uniform_color([0, 0.651, 0.929])  # 场景

    # source_temp.translate([-1.0, 0, 0, 0])
    # target_temp.flip()

    o3d.visualization.draw_geometries([target_temp],
                                      zoom=1,
                                      front=[0, -1, 0.1],  # 相机位置
                                      lookat=[0, 0, 0],  # 对准的点
                                      up=[0, 1, 0.1],  # 用于确定相机右x轴
                                      )


# 在下采样的上面划线
def draw_line_down(source_down, target_down, line_set, axis_mesh=None):
    if axis_mesh is None:
        o3d.visualization.draw_geometries([
            source_down,  # 重合的原因：source是直接抠出来的
            target_down,
            line_set
        ],
            window_name='ANTenna3D',
            zoom=1,
            front=[0, -1, 0.1],  # 相机位置
            lookat=[0, 0, 0],  # 对准的点
            up=[0, 1, 0.1],  # 用于确定相机右x轴
            # point_show_normal=True
        )

    o3d.visualization.draw_geometries([
        source_down,  # 重合的原因：source是直接抠出来的
        target_down,
        line_set,
        axis_mesh
    ],
        window_name='ANTenna3D',
        zoom=1,
        front=[0, -1, 0.1],  # 相机位置
        lookat=[0, 0, 0],  # 对准的点
        up=[0, 1, 0.1],  # 用于确定相机右x轴
        # point_show_normal=True
    )


# 生成连接线
# 输入模型和场景，以及匹配的索引  即可得到连接线
def get_line_set(model_np, scene_np, match_model_idx, match_scene_idx):
    # 为了可视化
    # print('scene_vtx:', len(scene_vtx))
    model_vtx_num = len(model_np)
    # print('model_vtx_num:', model_vtx_num)  # md 纠结好久，原来是场景家的不对！索引不是顶点的
    line_vtx = r_[model_np, scene_np]  # 顶点坐标stack起来
    line_idx = c_[match_model_idx, match_scene_idx + model_vtx_num]  # idx的场景那一半也得加上对应的长度
    # print('line_vtx:', line_vtx.shape)
    # print('line_idx:', line_idx.shape)

    color = array([0, 0.3, 1])
    colors = tile(color, (len(line_idx), 1))  # 复制数组
    # print(colors)

    line_set = draw_line(line_vtx, line_idx, colors)  # 对应点的匹配线  # points lines color

    return line_set


# 特征工程
# 预处理: 下采样，计算FPFH
def preprocess_point_cloud(pcd, voxel_size):
    # print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_down_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    return pcd_down, pcd_down_fpfh


# 要不要再拆开，后面再看需求
# 先把整个pipeline完成
def get_fpfh(pcd_in, voxel_size):
    radius_normal = voxel_size * 2
    pcd_in.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    # print("Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_in,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    return pcd_fpfh.data.T


def prepare_dataset(model_path, voxel_size):
    # print(":: Load two point clouds and disturb initial pose.")
    source = o3d.io.read_point_cloud(model_path)

    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)

    return source, source_down, source_fpfh


# 单纯加载模型并下采样
def load_model(model_path, trans_init, voxel_size):
    # print(":: Load one point cloud.")
    source = o3d.io.read_point_cloud(model_path)
    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # trans_init = np.asarray([[1.0, 0.0, 0.0, 5.0],
    #                          [0.0, 1.0, 0.0, -18.0],
    #                          [0.0, 0.0, 1.0, 0.0],
    #                          [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)

    source_down = source.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    return source, source_down


# 读取单个点云
# 下采样，计算FPFH
def read_pcd(model_path, voxel_size):
    pcd = o3d.io.read_point_cloud(model_path)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    pcd_down, pcd_down_fpfh = preprocess_point_cloud(pcd, voxel_size)

    return pcd, pcd_down, pcd_down_fpfh
