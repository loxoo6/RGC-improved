import pandas as pd
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from sklearn.decomposition import PCA

# 设置Matplotlib的后端
import matplotlib
matplotlib.use('Agg')

def calculate_gravity(i, j, data, graph):
    """计算两个点之间的万有引力"""
    distance = np.linalg.norm(data[i] - data[j])
    if distance == 0:
        return float('inf')  # 避免除以零
    gamma_i = set(graph.neighbors(i))
    gamma_j = set(graph.neighbors(j))
    if len(gamma_i) == 0 or len(gamma_j) == 0:
        return 0  # 如果任何一个节点没有邻居，结构相似性为0
    structural_similarity = len(gamma_i & gamma_j) / np.sqrt(len(gamma_i) * len(gamma_j))
    return structural_similarity / (distance ** 2)

def gnan_searching(data, graph):
    """基于万有引力的自然邻居搜索算法"""
    n = len(data)
    r = 1
    flag = True
    count = [n]
    nb = np.zeros(n, dtype=int)
    G = [list() for _ in range(n)]  # 使用列表而不是集合
    GNNr = [set() for _ in range(n)]
    GNaN = [set() for _ in range(n)]

    while flag:
        for i in range(n):
            for j in range(n):
                if i != j:
                    gravity = calculate_gravity(i, j, data, graph)
                    G[i].append((gravity, j))  # 添加到列表中

        for i in range(n):
            G[i].sort(reverse=True)  # 根据引力排序
            if len(G[i]) >= r:
                _, q = G[i][r-1]
                nb[q] += 1
                GNaN[q].add(i)
                GNNr[i].add(q)

        count_r = len([x for x in nb if x == 0])
        if count_r == 0 or (len(count) > 1 and count_r == count[-1]):
            flag = False
        else:
            count.append(count_r)
            r += 1

    return GNaN, GNNr, r, nb

# 读取CSV文件
file_path = "/home/zly/hjy_code/new/ang88/Ang88.csv"
data = pd.read_csv(file_path).iloc[:, 1:]

# 创建无向图
G = nx.Graph()

# 获取所有列名作为属性名
attribute_names = list(data.columns)

# 添加节点和节点特征
for index, row in data.iterrows():
    attributes = row.to_dict()
    G.add_node(index, **attributes)

# 使用KNN找到最近的8个邻居
k = 8
node_features = np.array([[G.nodes[node][attr] for attr in attribute_names] for node in G.nodes])
tree = KDTree(node_features)
distances, indices = tree.query(node_features, k=k+1)  # k+1是因为第一个最近邻是节点自身

# 为每个节点添加边，连接到其最近的8个邻居
for i, neighbors in enumerate(indices):
    for neighbor in neighbors[1:]:  # 跳过第一个节点自身
        G.add_edge(i, neighbor)

print("Initial graph created with KNN edges")

# 提取节点特征
data_points = node_features

# 运行GNaN-Searching算法
print("Running GNaN-Searching algorithm")
GNaN, GNNr, r, nb = gnan_searching(data_points, G)
print("GNaN-Searching algorithm finished")

# 添加边（使用自然邻居）
isolated_nodes = []
for i, neighbors in enumerate(GNaN):
    if neighbors:
        for neighbor in neighbors:
            G.add_edge(i, neighbor)
    else:
        isolated_nodes.append(i)

print("Graph structure updated with GNaN edges")

# 检查是否有边
if G.number_of_edges() == 0:
    print("No edges were added to the graph.")
else:
    print(f"Graph has {G.number_of_edges()} edges.")

# 打印孤立的节点
if isolated_nodes:
    print(f"The following nodes are isolated (no neighbors found): {isolated_nodes}")

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
np.save("/home/zly/hjy_code/RGC/dataset/Ang88/Ang88_adj.npy", adj_matrix)

# 保存节点特征矩阵 (feat.npy)
np.save('/home/zly/hjy_code/RGC/dataset/Ang88/Ang88_feat.npy', node_features)

# 生成降维后的特征矩阵 (feat_sm_2.npy)
pca = PCA(n_components=2)
feat_sm_2 = pca.fit_transform(node_features)
np.save('/home/zly/hjy_code/RGC/dataset/Ang88/Ang88_feat_sm_2.npy', feat_sm_2)

# 可视化图
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, seed=42)  # 使用spring布局进行节点位置布置
nx.draw(G, pos, with_labels=False, node_size=50, node_color="skyblue", font_size=8, edge_color="gray")
plt.title("Graph Visualization")
plt.savefig("/home/zly/hjy_code/RGC/graph_visualization.png")
print("Graph visualization saved as 'graph_visualization.png'")
