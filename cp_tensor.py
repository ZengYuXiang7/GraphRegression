# coding : utf-8
# Author : Yuxiang Zeng

import numpy as np


def khatri_rao(matrices):
    """计算 Khatri-Rao 积 (用于辅助CP分解)"""
    n_col = matrices[0].shape[1]
    n_matrices = len(matrices)
    prod = matrices[-1]
    for i in range(n_matrices - 2, -1, -1):
        mat = matrices[i]
        prod = (mat[:, None, :] * prod[None, :, :]).reshape(-1, n_col)
    return prod


def unfold(tensor, mode):
    """将张量按某个维度展开成矩阵"""
    return np.moveaxis(tensor, mode, 0).reshape(tensor.shape[mode], -1)


def fold(matrix, shape, mode):
    """将矩阵折叠回张量"""
    full_shape = list(shape)
    mode_dim = full_shape.pop(mode)
    full_shape.insert(0, mode_dim)
    return np.moveaxis(matrix.reshape(full_shape), 0, mode)


def cp_reconstruction(factors):
    """根据CP因子还原张量"""
    A, B, C = factors
    # 核心重建公式
    kr = khatri_rao([C, B])
    unfolded = A @ kr.T
    return fold(unfolded, (A.shape[0], B.shape[0], C.shape[0]), 0)


def simple_cp_als(tensor, rank, max_iter=10):
    """简易的 CP-ALS 分解算法"""
    shape = tensor.shape
    # 随机初始化因子矩阵
    factors = [np.random.rand(s, rank) for s in shape]

    for _ in range(max_iter):
        for mode in range(3):
            # 固定其他两个维度，更新当前维度
            others = [factors[i] for i in range(3) if i != mode]

            # 1. 计算 V = (C.T @ C) * (B.T @ B) (* 是元素乘)
            v_mat = np.ones((rank, rank))
            for fac in others:
                v_mat *= (fac.T @ fac)

            # 2. 构造右端项
            kr_mat = khatri_rao(others[::-1])
            unfolded_X = unfold(tensor, mode)
            rhs = unfolded_X @ kr_mat

            # 3. 求解线性方程更新因子: Factor * V = RHS
            # 使用 solve 求解比求逆更稳定
            factors[mode] = np.linalg.solve(v_mat.T, rhs.T).T

    return factors


def impute_missing_data(tensor_with_nan, rank, max_iter=50, tol=1e-4):
    """
    主函数：针对含 NaN 的三维数据进行填充
    """
    # 1. 初始化：记录缺失位置，用均值初始填充
    mask = np.isnan(tensor_with_nan)
    filled_tensor = tensor_with_nan.copy()
    fill_value = np.nanmean(tensor_with_nan)
    filled_tensor[mask] = fill_value

    print(f"开始填充，数据尺寸: {tensor_with_nan.shape}, 设定秩(Rank): {rank}")

    for i in range(max_iter):
        prev_tensor = filled_tensor.copy()

        # 2. 对当前填充好的数据做 CP 分解
        factors = simple_cp_als(filled_tensor, rank, max_iter=5)

        # 3. 用分解得到的因子重建张量
        reconstructed = cp_reconstruction(factors)

        # 4. 核心步骤：只更新缺失值部分，保留原始观测值
        filled_tensor[mask] = reconstructed[mask]

        # 检查收敛 (变化率)
        diff = np.linalg.norm(filled_tensor - prev_tensor) / np.linalg.norm(filled_tensor)
        if (i + 1) % 10 == 0:
            print(f"Iter {i + 1}: 变化率 = {diff:.6f}")
        if diff < tol:
            print(f"收敛于第 {i + 1} 次迭代")
            break

    return filled_tensor


# --- 使用示例 ---
# 假设你有一个名为 data_3d 的 numpy 数组，里面有 np.nan
if __name__ == "__main__":
    # 生成模拟数据
    real_data = np.random.rand(10, 10, 10)
    data_with_missing = real_data.copy()
    data_with_missing[np.random.rand(*real_data.shape) < 0.01] = np.nan  # 挖掉20%

    # 运行填充
    filled_data = impute_missing_data(data_with_missing, rank=3)

    # 验证效果 (仅作演示)
    error = np.nanmean((real_data - filled_data) ** 2)
    print(f"填充均方误差 (MSE): {error:.5f}")