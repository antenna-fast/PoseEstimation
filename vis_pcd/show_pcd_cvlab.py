import os
import numpy as np
import open3d as o3d


def show_pointcloud(pcd_path, is_mesh=0, is_down_sample=0):
    print("Load a ply point cloud, print it, and render it")
    if is_mesh:
        pcd = o3d.io.read_triangle_mesh(pcd_path)
    else:
        pcd = o3d.io.read_point_cloud(pcd_path)

        if is_down_sample:
            pcd = pcd.voxel_down_sample(voxel_size=0.01)

        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # pcd.paint_uniform_color([0, 0.40, 0.1])
    # pcd_np = np.asarray(pcd.points)  # 将点转换为numpy数组

    o3d.visualization.draw_geometries([pcd],
                                      window_name='ANTenna3D',
                                      # zoom=0.3412,
                                      # front=[0.4257, -0.2125, -0.8795],
                                      # lookat=[2.6172, 2.0475, 1.532],
                                      # up=[-0.0694, -0.9768, 0.2024]
                                      )


if __name__ == '__main__':
    # dataset 1 stanford dense
    root_path = '/Users/aibee/Downloads/Dataset/3DDataset/6DPose/CVLab'
    folder = 'dataset2'
    scene = 'dataset2_scenes/3D models/Stanford/Random/Scene0_1-8'
    pcd_path = os.path.join(root_path, folder, scene + '.ply')
    # show_pointcloud(pcd_path=pcd_path)

    # dataset 2 stanford spaese

    # dataset 3 Spacetime Stereo

    # dataset 4 Spacetime Stereo Texture

    # dataset 5 kinect
    model_path = os.path.join(root_path, 'dataset5/dataset5/3D models/CVLab/Kinect/MeshRegistration/Doll/Doll005.ply')
    # show_pointcloud(pcd_path=model_path)
