import pandas as pd
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('Agg')

# 读取Excel文件，只读取数据部分（跳过表头）
file_path = "/home/zly/hjy_code/RGC/dataset/shequ/minmax_shequ.csv"
data = pd.read_csv(file_path).iloc[:, 1:]

# 创建无向图
G = nx.Graph()

# 获取所有列名作为属性名
attribute_names = list(data.columns)

# 添加节点和节点特征
for index, row in data.iterrows():
    attributes = {str(key): value for key, value in row.to_dict().items()}  # 将键转换为字符串
    G.add_node(index, **attributes)

# 提取节点特征
node_features = np.array([[float(G.nodes[node][attr]) for attr in map(str, attribute_names)] for node in G.nodes])

# 检查并处理 NaN 和 inf 值
if not np.isfinite(node_features).all():
    # 处理 NaN 和 inf 值，例如用 0 替换或移除含有这些值的行
    node_features = np.nan_to_num(node_features, nan=0.0, posinf=0.0, neginf=0.0)

# 使用KNN找到最近的8个邻居
k = 8
tree = KDTree(node_features)
distances, indices = tree.query(node_features, k=k+1)  # k+1是因为第一个最近邻是节点自身

# 为每个节点添加边，连接到其最近的8个邻居
for i, neighbors in enumerate(indices):
    for neighbor in neighbors[1:]:  # 跳过第一个节点自身
        G.add_edge(i, neighbor)

# 导出边列表
edge_index = np.array(G.edges).T

# 转换为PyTorch Tensor
x = torch.tensor(node_features, dtype=torch.float)
edge_index = torch.tensor(edge_index, dtype=torch.long)

# 创建PyTorch Geometric数据对象
data = Data(x=x, edge_index=edge_index)

# 查看数据对象
print(data)

# 打印节点特征和边列表
print(node_features.shape)
print(edge_index.shape)
print(node_features[:5])
print(edge_index[:, :5])

# 保存邻接矩阵 (adj.npy)
adj_matrix = nx.to_numpy_array(G, nodelist=sorted(G.nodes))
np.save("/home/zly/hjy_code/RGC/dataset/shequ/adj.npy", adj_matrix)

# 保存节点特征矩阵 (feat.npy)
np.save('/home/zly/hjy_code/RGC/dataset/shequ/feat.npy', node_features)

# 生成降维后的特征矩阵 (feat_sm_2.npy)
pca = PCA(n_components=2)
feat_sm_2 = pca.fit_transform(node_features)
np.save('/home/zly/hjy_code/RGC/dataset/shequ/feat_sm_2.npy', feat_sm_2)

# 可视化图
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, seed=42)  # 使用spring布局进行节点位置布置
nx.draw(G, pos, with_labels=False, node_size=50, node_color="skyblue", font_size=8, edge_color="gray")
plt.title("Graph Visualization")
plt.savefig("/home/zly/hjy_code/RGC/dataset/shequ/graph_visualization.png")
print("Graph visualization saved as 'graph_visualization.png'")


