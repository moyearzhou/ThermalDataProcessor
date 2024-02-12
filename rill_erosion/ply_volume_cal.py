# coding = utf-8
# @Time : 2024/1/20 16:54
# @Author : moyear
# @File : ply_volume_cal.y
# @Software : PyCharm
import open3d as o3d
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial import Delaunay

path_ply_a = r"E:\Users\Moyear\Desktop\3d\TEST_0.ply"
path_ply_b = r"E:\Users\Moyear\Desktop\3d\TEST_1.ply"


def load_point_cloud(filename):
    print(f"========加载点云数据 {filename}")
    return o3d.io.read_point_cloud(filename)


def align_point_clouds(source, target):

    print("========对齐点云数据")
    # 这里需要一个对齐算法，例如ICP或其他
    # 这里只是一个占位符，实际实现将更复杂
    transformation = np.identity(4) # 假设的对齐矩阵
    source.transform(transformation)
    return source


def build_tin(point_cloud):
    print("========构建TIN")
    points = np.asarray(point_cloud.points)
    # 2D Delaunay三角化，假设点云已经投影到2D平面
    tri = Delaunay(points[:, :2])
    return tri


def calculate_volume_change(tin, points_a, points_b):
    print("========计算体积变化")
    volume_change = 0.0

    # 创建points_b的KD树
    tree = KDTree(points_b[:, :2])  # 只使用x和y坐标

    for simplex in tin.simplices:
        # 对于三角形的每个顶点，找到points_b中的最近点
        simplex_points_b_indices = tree.query(points_a[simplex, :2])[1]

        # 计算三角形顶点的高度变化
        height_change = np.mean(points_b[simplex_points_b_indices, 2] - points_a[simplex, 2])

        # 三角形面积\
        area = 0.5 * np.linalg.norm(np.cross(
            points_a[simplex[1]] - points_a[simplex[0]],
            points_a[simplex[2]] - points_a[simplex[0]]
        ))
        # 体积变化为底面积乘以高度变化
        volume_change += area * height_change

    return volume_change


# 加载点云数据
cloud_a = load_point_cloud(path_ply_a)
cloud_b = load_point_cloud(path_ply_b)

# 对齐点云数据
aligned_cloud_a = align_point_clouds(cloud_a, cloud_b)

# 构建TIN
tin = build_tin(aligned_cloud_a)

# 计算体积变化
volume_change = calculate_volume_change(tin, np.asarray(aligned_cloud_a.points), np.asarray(cloud_b.points))

print(f"总体积变化: {volume_change / 1000} ml")
