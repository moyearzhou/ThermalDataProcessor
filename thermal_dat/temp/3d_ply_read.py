# coding = utf-8
# @Time : 2023/12/21 18:42
# @Author : moyear
# @File : 3d_ply_read.y
# @Software : PyCharm
from matplotlib import pyplot as plt
from plyfile import PlyData
import open3d as o3d


def open3d(path):
    # 读取PLY文件
    pcd = o3d.io.read_point_cloud(path)

    # 可视化点云
    o3d.visualization.draw_geometries([pcd])


def plot_3d(path):
    # 打开PLY文件
    plydata = PlyData.read(path)

    # 获取顶点坐标数据
    vertices = plydata['vertex']
    x = vertices['x']
    y = vertices['y']
    z = vertices['z']

    # 创建3D图像对象
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制3D散点图
    ax.scatter(x, y, z, c='b', marker='o')

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 显示图像
    plt.show()


if __name__ == '__main__':
    path = r"E:\Users\Moyear\Desktop\3d\ysy_1_3.ply"
    open3d(path)
