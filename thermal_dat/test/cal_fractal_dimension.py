# coding = utf-8
# @Time : 2024/2/27 22:56
# @Author : moyear
# @File : cal_fractal_dimension.y
# @Software : PyCharm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress


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


# 假设dem_diff是一个二维高程差异数组
# dem_diff = ...

# 计算分形维数，需要选择一个适当的阈值
# 这里假设高程差异的最大值是阈值
fd = fractal_dimension(dem_diff, threshold=dem_diff.max())

print(f"分形维数: {fd}")
