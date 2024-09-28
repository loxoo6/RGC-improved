import pandas as pd
import networkx as nx
from sklearn.preprocessing import MinMaxScaler

# 读取Excel文件，只读取数据部分（跳过表头）
file_path = "/home/zly/hjy_code/RGC/dataset/shequ/Shequ.xlsx"
data = pd.read_excel(file_path, header=None).iloc[2:, 2:]

# 创建无向图
G = nx.Graph()

# 标准化数据
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# 将标准化后的数据转换为DataFrame
normalized_data = pd.DataFrame(data)

# 保存标准化后的数据到新的Excel文件
output_path = "/home/zly/hjy_code/RGC/dataset/shequ/minmax_shequ.xlsx"
normalized_data.to_excel(output_path, index=False)

print(f"Normalized data saved to {output_path}")
