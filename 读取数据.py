# ---------------------------读取cora文件内容--------------------------------------
# import pickle
#
# # 文件路径
# graph_file_path = "/home/zly/hjy_code/RGC/cora_dataset/ind.cora.graph"
# y_file_path = "/home/zly/hjy_code/RGC/cora_dataset/ind.cora.y"
# x_file_path = "/home/zly/hjy_code/RGC/cora_dataset/ind.cora.x"
#
# # 使用 pickle 读取文件，并指定编码为 'latin1'
# with open(graph_file_path, 'rb') as f:
#     graph_data = pickle.load(f, encoding='latin1')
#
# with open(y_file_path, 'rb') as f:
#     y_data = pickle.load(f, encoding='latin1')
#
# with open(x_file_path, 'rb') as f:
#     x_data = pickle.load(f, encoding='latin1')
#
# # 显示每个数据的结构和内容
# graph_summary = {
#     'Type': type(graph_data),
#     'Keys': list(graph_data.keys())[:5],  # 显示前5个键
#     'Sample Data': [(key, graph_data[key]) for key in list(graph_data.keys())[:5]]  # 显示前5个条目
# }
#
# y_summary = {
#     'Shape': y_data.shape,
#     'Sample Data': y_data[:5]  # 显示前5个条目
# }
#
# x_summary = {
#     'Shape': x_data.shape,
#     'Sample Data': x_data[:5]  # 显示前5个条目
# }
#
# print(graph_summary, y_summary, x_summary)
#----------------------------------------------------------------------------------------------
#-------------------------------------将cora原始文件格式转换成npy格式------------------------------
import numpy as np
import pickle
import scipy.sparse as sp

# # 文件路径
# old_feat_file_path = "/home/zly/hjy_code/RGC/cora_dataset/ind.cora.x"
# old_label_file_path = "/home/zly/hjy_code/RGC/cora_dataset/ind.cora.y"
# old_adj_file_path = "/home/zly/hjy_code/RGC/cora_dataset/ind.cora.graph"
#
# # 使用 pickle 读取文件，并指定编码为 'latin1'
# with open(old_feat_file_path, 'rb') as f:
#     old_x_data = pickle.load(f, encoding='latin1')
#
# with open(old_label_file_path, 'rb') as f:
#     old_y_data = pickle.load(f, encoding='latin1')
#
# with open(old_adj_file_path, 'rb') as f:
#     old_graph_data = pickle.load(f, encoding='latin1')
#
# # 转换特征数据
# feat_data = old_x_data.toarray()
#
# # 转换标签数据
# label_data = np.argmax(old_y_data, axis=1)
#
# # 转换邻接矩阵数据
# adj_data = np.zeros((len(old_graph_data), len(old_graph_data)), dtype=np.int32)
# for key, value in old_graph_data.items():
#     adj_data[key, value] = 1
#
# # 保存为新格式文件
# np.save('/home/zly/hjy_code/RGC/dataset/cora/cora_feat.npy', feat_data)
# np.save('/home/zly/hjy_code/RGC/dataset/cora/cora_label.npy', label_data)
# np.save('/home/zly/hjy_code/RGC/dataset/cora/cora_adj.npy', adj_data)
#
# print("转换成功！")

#-----------------------------读取npy文件数据--------------------------------------

import numpy as np
def load_graph_data(dataset_name, show_details=True):
    """
    - Param
    dataset_name: the name of the dataset
    show_details: if show the details of dataset
    - Return:
    the features, labels and adj
    """
    load_path = "dataset/" + dataset_name + "/" + dataset_name
    feat = np.load(load_path + "_feat.npy", allow_pickle=True)
    # label = np.load(load_path + "_label.npy", allow_pickle=True)
    adj = np.load(load_path + "_adj.npy", allow_pickle=True)
    if show_details:
        print("dataset name:   ", dataset_name)
        print("feature shape:  ", feat.shape)
        # print("label shape:    ", label.shape)
        print("adj shape:      ", adj.shape)
        print("undirected edge num:   ", int(np.nonzero(adj)[0].shape[0] / 2))
        # print("category num:          ", max(label) - min(label) + 1)
        # print("category distribution: ")
        # for i in range(max(label) + 1):
        #     print("label", i, end=":")
        #     print(len(label[np.where(label == i)]))

    featur_dim = feat.shape[1]

    return feat, adj
load_graph_data('cora')

