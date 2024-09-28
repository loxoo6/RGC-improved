import numpy as np
import pandas as pd

# 加载数据
features_pca = pd.read_csv('/home/zly/hjy_code/RGC/features_pca.csv')
minmax_shequ = pd.read_csv("/home/zly/hjy_code/RGC/dataset/shequ/minmax_shequ.csv")

# 提取特征
f1 = features_pca['f1'].values
f2 = features_pca['f2'].values

# 构建设计矩阵 X 和响应矩阵 Y
n = minmax_shequ.shape[1]
X = np.vstack([np.ones(n), f1, f2]).T
Y = minmax_shequ.values

# 计算回归系数
beta = np.linalg.inv(X.T @ X) @ X.T @ Y

# 提取系数
beta_0 = beta[0, :]
beta_1 = beta[1, :]
beta_2 = beta[2, :]

# 输出每个变量的 f1 和 f2 表达式
for i, column in enumerate(minmax_shequ.columns):
    print(f"{column}:")
    print(f"f1 = ({Y[:, i]} - {beta_0[i]} - {beta_2[i]} * f2) / {beta_1[i]}")
    print(f"f2 = ({Y[:, i]} - {beta_0[i]} - {beta_1[i]} * f1) / {beta_2[i]}")
    print("\n")
