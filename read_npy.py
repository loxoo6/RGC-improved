import numpy as np

# 指定npy文件的路径
file_path = "/home/zly/hjy_code/RGC/dataset/cora_adj.npy"
# 读取npy文件
data = np.load(file_path)

# 打印数据
print(data)
print(data.shape)
print(data.dtype)

