import os

import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import from_origin
from scipy.spatial import Delaunay
import matplotlib.colors as mcolors
from scipy.stats import linregress

elevation_min = -100

elevation_max = 50


def read_point_cloud(ply_file):
    # 读取点云数据
    pcd = o3d.io.read_point_cloud(ply_file)
    # 转换为numpy数组
    xyz = np.asarray(pcd.points)
    return xyz


def create_surface(xyz):
    # # 这里需要根据点云生成表面，可能需要使用三角化算法如Delaunay
    # # 这里仅提供了一个占位符，具体实现会依赖于点云的密度和分布
    # # 返回一个二维数组作为地形的高程矩阵
    # return np.zeros((xyz.shape[0], xyz.shape[1]))  # 这需要替换为实际的DEM生成代码

    # Step 1: Create the Delaunay triangulation of the point cloud
    tri = Delaunay(xyz[:, :2])  # We only take the x and y for triangulation

    # Step 2: Define the grid upon which to interpolate
    # For example, create a grid from the bounds of your point cloud
    x_min, x_max = np.min(xyz[:, 0]), np.max(xyz[:, 0])
    y_min, y_max = np.min(xyz[:, 1]), np.max(xyz[:, 1])
    grid_x, grid_y = np.mgrid[x_min:x_max:200j, y_min:y_max:800j]  # 100x100 grid

    # Step 3: Interpolate the z-values on the grid
    from scipy.interpolate import LinearNDInterpolator
    interpolator = LinearNDInterpolator(tri, xyz[:, 2])
    dem = interpolator(grid_x, grid_y)

    # Replace nan values with the minimum z value
    dem[np.isnan(dem)] = np.min(xyz[:, 2])

    return dem


def compare_dem(dem_before, dem_after):
    # 比较两个DEM数组并返回差异
    return dem_after - dem_before


def get_dem_image(dem_diff, elevation_min, elevation_max):
    '''
    将dem转为rgb图像
    :param elevation_max:
    :param elevation_min:
    :param dem_diff:
    :return:
    '''
    # 创建规范化对象
    norm = mcolors.Normalize(vmin=elevation_min, vmax=elevation_max)

    # 获取terrain颜色方案
    terrain_cmap = plt.get_cmap('terrain')

    # 将DEM数据规范化并应用颜色映射
    colored_dem = terrain_cmap(norm(dem_diff))

    # Matplotlib返回的是[0, 1]范围的数据，转换为[0, 255]的范围的uint8类型
    colored_dem_255 = (colored_dem[:, :, :3] * 255).astype(np.uint8)
    return colored_dem_255


def get_detailed_slope_erosion_image(paths_ply):
    '''
    获取每个坡面详细的侵蚀图像（rgb图像），一共5个子图像拼接而成，依次是：第一阶段侵蚀、第二阶段侵蚀、第三阶段侵蚀、第四阶段侵蚀、总共侵蚀
    :param paths_ply:
    :return:
    '''

    images_hstack = None

    # print(paths_ply)

    # 获取每个阶段的侵蚀dem
    last = 0
    for i in range(1, 5):
        path_base = paths_ply[last]
        path_target = paths_ply[i]

        # print(ply_base, ply_target)

        file_name_base = os.path.basename(path_base)
        file_name_target = os.path.basename(path_target)

        print("正在处理：", file_name_base, " - ", file_name_target, "的侵蚀地形dem")

        dem_diff = cal_volume_change(path_base, path_target)
        # 将dem图像进行旋转旋转90度，变成纵向
        dem_diff = np.rot90(dem_diff)

        colored_dem_255 = get_dem_image(dem_diff, elevation_min, elevation_max)
        # print(res.shape)

        if images_hstack is None:
            images_hstack = colored_dem_255
        else:
            images_hstack = img_hstack(images_hstack, colored_dem_255)

        last = i

    # 获取总侵蚀dem
    path_base = paths_ply[0]
    path_target = paths_ply[4]

    file_name_base = os.path.basename(path_base)
    file_name_target = os.path.basename(path_target)

    print("正在处理：", file_name_base, " - ", file_name_target, "的侵蚀地形dem")

    dem_diff = cal_volume_change(path_base, path_target)
    # 将dem图像进行旋转旋转90度，变成纵向
    dem_diff = np.rot90(dem_diff)

    colored_dem_255 = get_dem_image(dem_diff, elevation_min, elevation_max)
    images_hstack = img_hstack(images_hstack, colored_dem_255)

    return images_hstack


def get_all_slope_image():
    dir_plys = r"E:\Users\Moyear\Desktop\3d\slopes"

    slope_names = ['A0',
                   'B1', 'B2', 'B3',
                   'C1', 'C2', 'C3',
                   'D1', 'D2', 'D3',
                   'E1', 'E2', 'E3']

    # for slope_name in slope_names:
    #     # 初始化列表来保存文件路径
    #     file_paths_list = []
    #
    #     # 遍历文件夹中的所有文件
    #     for root, dirs, files in os.walk(dir_plys):
    #         for file in files:
    #             if file.startswith(slope_name) and file.endswith('.ply'):
    #                 ply_file_path = os.path.join(root, file)
    #                 file_paths_list.append(ply_file_path)
    #
    #     get_detailed_slope_erosion_image(file_paths_list)
    #
    #     return

    slope_name = 'E2'
    # 初始化列表来保存文件路径
    file_paths_list = []

    # 遍历文件夹中的所有文件
    for root, dirs, files in os.walk(dir_plys):
        for file in files:
            if file.startswith(slope_name) and file.endswith('.ply'):
                ply_file_path = os.path.join(root, file)
                file_paths_list.append(ply_file_path)

    slopes_erosion = get_detailed_slope_erosion_image(file_paths_list)

    # 使用opencv显示图像
    slopes_erosion_bgr = cv2.cvtColor(slopes_erosion, cv2.COLOR_RGB2BGR)

    # 使用OpenCV显示图像
    cv2.imshow(slope_name, slopes_erosion_bgr)
    cv2.waitKey(0)


def img_hstack(image1, image2):
    result = None
    # 确保两张图像的高度一致
    if image1.shape[0] == image2.shape[0]:
        # 横向拼接两张图像
        result = np.hstack((image1, image2))

        # # 显示拼接后的图像
        # cv2.imshow('Result', result)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    else:
        print('图像高不一致！！！！')

    return result


def cal_volume_change(path_base, path_target):
    # 读取侵蚀前后的点云数据
    xyz_before = read_point_cloud(path_base)
    xyz_after = read_point_cloud(path_target)

    # 创建DEM
    dem_before = create_surface(xyz_before)
    dem_after = create_surface(xyz_after)

    # 比较DEM以获取差异
    dem_diff = compare_dem(dem_before, dem_after)
    return dem_diff


def main():
    # 指定颜色映射两端对应的最小和最大高程值
    elevation_min = -100  # 指定最小高程值
    elevation_max = 50  # 指定最大高程值

    path_base = r"E:\Users\Moyear\Desktop\3d\slopes\B1_0_20240102_A.ply"
    path_target = r"E:\Users\Moyear\Desktop\3d\slopes\B1_4_20240102_A.ply"

    dem_diff = cal_volume_change(path_base, path_target)
    # 将dem图像进行旋转旋转90度，变成纵向
    dem_diff = np.rot90(dem_diff)

    colored_dem_255 = get_dem_image(dem_diff, elevation_min, elevation_max)
    # print(res.shape)

    dem_img_bgr = cv2.cvtColor(colored_dem_255, cv2.COLOR_RGB2BGR)

    # 使用OpenCV显示图像
    cv2.imshow('Colored DEM', dem_img_bgr)


    # 创建规范化对象
    norm = mcolors.Normalize(vmin=elevation_min, vmax=elevation_max)

    # 获取terrain颜色方案
    terrain_cmap = plt.get_cmap('terrain')

    # 绘制DEM图
    plt.imshow(dem_diff, cmap=terrain_cmap, norm=norm)

    # 可视化DEM差异
    # plt.imshow(dem_diff, cmap='terrain')  # 使用terrain颜色映射
    plt.colorbar(label='Elevation (m)')
    plt.title('Digital Elevation Model')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()

    # # 将DEM差异保存到文件，这里以.npy格式保存，也可以选择其他格式
    # np.save('dem_difference.npy', dem_diff)

    # dem_height = dem_diff.shape[0]
    # dem_width = dem_diff.shape[1]
    # # 导出DEM为TIFF文件
    # transform = from_origin(np.min(xyz_before[:, 0]), np.max(xyz_before[:, 1]), 1, 1)  # 栅格大小需要根据实际调整
    # with rasterio.open('DEM_output.tif', 'w', driver='GTiff',
    #                    height=dem_height, width=dem_width,
    #                    count=1, dtype=str(dem_diff.dtype),
    #                    crs='+proj=latlong',
    #                    transform=transform) as dst:
    #     dst.write(dem_diff, 1)


def boxcount(Z, k):
    S = np.add.reduceat(
        np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
        np.arange(0, Z.shape[1], k), axis=1)
    return len(np.where(S > 0)[0])


def fractal_dimension(Z, threshold=0.9):
    # Only for 2d image
    assert (len(Z.shape) == 2)

    # Transform Z into a binary array
    Z = (Z < threshold)

    p = min(Z.shape)
    n = 2 ** np.floor(np.log(p) / np.log(2))
    n = int(np.log(n) / np.log(2))
    sizes = 2 ** np.arange(n, 1, -1)

    # Box counting
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))

    # Linear regression
    coeffs = linregress(np.log(sizes), np.log(counts))
    return -coeffs[0]


if __name__ == "__main__":
    # main()
    # get_all_slope_image()

    path_base = r"E:\Users\Moyear\Desktop\3d\slopes\B1_0_20240102_A.ply"
    path_target = r"E:\Users\Moyear\Desktop\3d\slopes\B1_4_20240102_A.ply"

    dem_diff = cal_volume_change(path_base, path_target)

    # 这里假设高程差异的最大值是阈值
    fd = fractal_dimension(dem_diff, threshold=1)

    print(f"分形维数: {fd}")

