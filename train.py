import os
import argparse
import warnings
from utils import *
from tqdm import tqdm
from torch import optim
import torch.nn.functional as F
from torch_scatter import scatter
from model import my_model, my_Q_net
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from meta_model import meta_model
from data_utils import adarmic_adar

matplotlib.use('Agg')
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()

# --------------建图-------------------------
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
# # 读取CSV文件
# file_path = "/home/zly/hjy_code/RGC/dataset/Bai260/Bai260.csv"
# data = pd.read_csv(file_path).iloc[:, 1:]
#
# # 创建无向图
# G = nx.Graph()
#
# # 获取所有列名作为属性名
# attribute_names = list(data.columns)
#
# # 添加节点和节点特征
# for index, row in data.iterrows():
#     attributes = row.to_dict()
#     G.add_node(index, **attributes)
#
# # 添加边（按顺序相邻节点相连）
# for index in range(len(data) - 1):
#     G.add_edge(index, index + 1)
#
# # 检查并补全所有节点的属性
# for node in G.nodes:
#     for attr in attribute_names:
#         if attr not in G.nodes[node]:
#             G.nodes[node][attr] = 0.0  # 为缺少的属性设置默认值
#
# # 提取节点特征
# node_features = np.array([[G.nodes[node][attr] for attr in attribute_names] for node in G.nodes])
#
# # 使用KNN找到最近的8个邻居
# k = 8
# tree = KDTree(node_features)
# distances, indices = tree.query(node_features, k=k + 1)  # k+1是因为第一个最近邻是节点自身
#
# # 为每个节点添加边，连接到其最近的8个邻居
# for i, neighbors in enumerate(indices):
#     for neighbor in neighbors[1:]:  # 跳过第一个节点自身
#         G.add_edge(i, neighbor)
#
# # 导出边列表
# edge_index = np.array(G.edges).T
#
# # 转换为PyTorch Tensor
# x = torch.tensor(node_features, dtype=torch.float)
# edge_index = torch.tensor(edge_index, dtype=torch.long)
#
# # 创建PyTorch Geometric数据对象
# data = Data(x=x, edge_index=edge_index)
#
# # 查看数据对象
# print(data)
#
# # 打印节点特征和边列表
# print(node_features.shape)
# print(edge_index.shape)
# print(node_features[:5])
# print(edge_index[:, :5])
#
# # 保存邻接矩阵 (adj.npy)
# adj_matrix = nx.to_numpy_array(G, nodelist=sorted(G.nodes))
# noisy_adj_th = torch.tensor(adj_matrix, dtype=torch.float).detach()
#
# d = noisy_adj_th.sum(dim=0, keepdim=True).T
# num_edges = noisy_adj_th.sum() / 2
# mod_null = (d @ d.T) / (2 * num_edges)

# data
parser.add_argument('--dataset', type=str, default='cora', help='type of dataset.')
parser.add_argument('--cluster_num', type=int, default=7, help='type of dataset.')
parser.add_argument('--gnnlayers', type=int, default=3, help="Number of gnn layers")

# E net
parser.add_argument('--E_epochs', type=int, default=400, help='Number of epochs to train E.')
parser.add_argument('--n_input', type=int, default=1000, help='Number of units in hidden layer 1.')
parser.add_argument('--dims', type=int, default=[500], help='Number of units in hidden layer 1.')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')

# Q net
parser.add_argument('--Q_epochs', type=int, default=30, help='Number of epochs to train Q.')
parser.add_argument('--epsilon', type=float, default=0.5, help='Greedy rate.')
parser.add_argument('--replay_buffer_size', type=float, default=50, help='Replay buffer size')
parser.add_argument('--Q_lr', type=float, default=1e-3, help='Initial learning rate.')

args = parser.parse_args()

device = "cuda:0"
file_name = "result.csv"
for args.dataset in ["cora","bat"]:
    # "amap",
    file = open(file_name, "a+")
    print(args.dataset, file=file)
    file.close()

    if args.dataset == 'cora':
        args.cluster_num = 7
        args.gnnlayers = 2
        args.lr = 1e-3
        args.n_input = 500
        args.dims = [1500]
        args.epsilon = 0.5
        args.replay_buffer_size = 40

    elif args.dataset == 'citeseer':
        args.cluster_num = 6
        args.gnnlayers = 2
        args.lr = 1e-3
        args.n_input = 500
        args.dims = [1500]
        args.epsilon = 0.7
        args.replay_buffer_size = 50

    elif args.dataset == 'amap':
        args.cluster_num = 8
        args.gnnlayers = 3
        args.lr = 1e-5
        args.n_input = -1
        args.dims = [500]
        args.epsilon = 0.7
        args.replay_buffer_size = 50

    elif args.dataset == 'bat':
        args.cluster_num = 4
        args.gnnlayers = 6
        args.lr = 1e-3
        args.n_input = -1
        args.dims = [1500]
        args.epsilon = 0.3
        args.replay_buffer_size = 30

    elif args.dataset == 'eat':
        args.cluster_num = 4
        args.gnnlayers = 6
        args.lr = 1e-4
        args.n_input = -1
        args.dims = [1500]
        args.epsilon = 0.7
        args.replay_buffer_size = 40
    elif args.dataset == 'Huang109':
        args.cluster_num = 4
        args.gnnlayers = 2
        args.lr = 1e-4
        args.n_input = -1
        args.dims = [10]
        args.epsilon = 0.3
        args.replay_buffer_size = 40

    predict_labels_list = []
    sc_list = []
    k_list = []
    # init
    for seed in range(10):
        setup_seed(seed)
        X, A = load_graph_data(args.dataset, show_details=False)
        features = X
        # true_labels = y
        adj = sp.csr_matrix(A)

        if args.n_input != -1:
            pca = PCA(n_components=args.n_input)
            features = pca.fit_transform(features)

        A = torch.tensor(adj.todense()).float().to(device)

        adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
        adj.eliminate_zeros()
        print('Laplacian Smoothing...')
        adj_norm_s = preprocess_graph(adj, args.gnnlayers, norm='sym', renorm=True)
        sm_fea_s = sp.csr_matrix(features).toarray()

        path = "dataset/{}/{}_feat_sm_{}.npy".format(args.dataset, args.dataset, args.gnnlayers)
        if os.path.exists(path):
            sm_fea_s = np.load(path, allow_pickle=True)
        else:
            for a in adj_norm_s:
                print(f'Shape of a: {a.shape}')
                print(f'Shape of sm_fea_s before dot: {sm_fea_s.shape}')
                sm_fea_s = a.dot(sm_fea_s)
                print(f'Shape of sm_fea_s after dot: {sm_fea_s.shape}')
            np.save(path, sm_fea_s, allow_pickle=True)

        # X
        sm_fea_s = torch.FloatTensor(sm_fea_s)
        adj_1st = (adj + sp.eye(adj.shape[0])).toarray()

        # test
        # best_nmi = 0
        # best_ari = 0
        args.cluster_num = np.random.randint(0, 9) + 2

        # init clustering
        _,_,_, predict_labels, _, _ = clustering(sm_fea_s.detach(), args.cluster_num, device=device)

        best_sc = 0
        best_db=0
        best_ch=0
        # MLP
        if args.dataset == "citeseer":
            model = my_model([sm_fea_s.shape[1]] + args.dims, A, act="sigmoid")
            # model = my_model([sm_fea_s.shape[1]] + args.dims, act="sigmoid")
        else:
            model = my_model([sm_fea_s.shape[1]] + args.dims, A)
        Q_net = my_Q_net(args.dims + [256, 9]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        optimizer_Q = optim.Adam(Q_net.parameters(), lr=args.Q_lr)

        model.to(device)
        inx = sm_fea_s.to(device)
        inx_origin = torch.FloatTensor(features).to(device)

        A_label = torch.FloatTensor(adj_1st).to(device)

        target = A_label
        mask = torch.ones([target.shape[0] * 2, target.shape[0] * 2]).to(device)
        mask -= torch.diag_embed(torch.diag(mask))

        tmp_epsilon = args.epsilon
        epsilon_step = (1 - tmp_epsilon) / args.E_epochs
        replay_buffer = []

        print('Start Training...')
        for epoch in tqdm(range(args.E_epochs)):
            model.train()
            Q_net.eval()
            optimizer.zero_grad()
            z1, z2 = model(inx)


            # -----------------------------样本级相关性减少 (SCR)-----------------------------
            def calculate_SCR(z1, z2):
                # 计算相似性矩阵 SN
                S = z1 @ z2.T  # 使用余弦相似度
                SN = S / (torch.norm(z1, dim=1, keepdim=True) @ torch.norm(z2, dim=1, keepdim=True).T)

                # 对角项和非对角项
                diag_elements = torch.diag(SN)
                off_diag_elements = SN - torch.diag_embed(diag_elements)

                # 样本级相关性减少损失
                L_N = (torch.mean((diag_elements - 1) ** 2) +
                       torch.mean(off_diag_elements ** 2) / (SN.shape[0] * (SN.shape[0] - 1)))
                return L_N


            # -----------------------------特征级相关性减少 (FCR)-----------------------------
            def calculate_FCR(z1, z2):
                # 转置嵌入矩阵，计算特征级别的相似性
                z1_t = z1.T
                z2_t = z2.T

                # 计算相似性矩阵 SF
                SF = z1_t @ z2_t.T  # 使用余弦相似度
                SF = SF / (torch.norm(z1_t, dim=1, keepdim=True) @ torch.norm(z2_t, dim=1, keepdim=True).T)

                # 对角项和非对角项
                diag_elements = torch.diag(SF)
                off_diag_elements = SF - torch.diag_embed(diag_elements)

                # 特征级相关性减少损失
                L_F = (torch.mean((diag_elements - 1) ** 2) +
                       torch.mean(off_diag_elements ** 2) / (SF.shape[0] * (SF.shape[0] - 1)))
                return L_F


            # -------------------------------------计算SCR和FCR损失--------------
            L_N = calculate_SCR(z1, z2)
            L_F = calculate_FCR(z1, z2)
            # 总损失
            # loss = L_N + L_F
            # -----------------------------------------------------------------

            z1_z2 = torch.cat([z1, z2], dim=0)
            S = z1_z2 @ z1_z2.T

            # pos neg weight
            pos_neg = mask * torch.exp(S)

            pos = torch.cat([torch.diag(S, target.shape[0]), torch.diag(S, -target.shape[0])], dim=0)
            # pos weight
            pos = torch  .exp(pos)

            neg = (torch.sum(pos_neg, dim=1) - pos)

            infoNEC = (-torch.log(pos / (pos + neg))).sum() / (2 * target.shape[0])
            # # contrastive_loss = neighbor_oriented_contrastive_loss(z1, z2, A)
            # loss = infoNEC + calculate_FCR(z1, z2)
            loss = infoNEC

            state = (z1 + z2) / 2

            cluster_state = scatter(state, torch.tensor(predict_labels).to(device), dim=0, reduce="mean")

            rand = False

            # do action by random choose
            if random.random() > args.epsilon:
                action = np.random.randint(0, 9)
                rand = True
            # do action by Q-net
            else:
                action = int(Q_net(state, cluster_state).mean(0).argmax())

            args.cluster_num = action + 2
            SC, DB, CH, predict_labels, centers, dis = clustering(state.detach(), args.cluster_num, device=device)
            dis = (state.unsqueeze(1) - centers.unsqueeze(0)).pow(2).sum(-1) + 1

            q = dis / (dis.sum(-1).reshape(-1, 1))
            p = q.pow(2) / q.sum(0).reshape(1, -1)
            p = p / p.sum(-1).reshape(-1, 1)
            pq_loss = F.kl_div(q.log(), p)
            loss += 10 * pq_loss
            if SC >= best_sc and rand == False:
                best_sc = SC
                best_cluster = args.cluster_num
                best_predict_labels = predict_labels
            if DB >= best_db and rand == False:
                best_db = DB
            if CH >= best_ch and rand == False:
                best_ch = CH

            loss.backward()
            optimizer.step()

            # next state
            model.eval()
            z1, z2 = model(inx)
            next_state = (z1 + z2) / 2

            next_cluster_state = scatter(next_state, torch.tensor(predict_labels).to(device), dim=0, reduce="mean")

            center_dis = (centers.unsqueeze(1) - centers.unsqueeze(0)).pow(2).sum(-1).mean()
            reward = center_dis.detach() - torch.min(dis, dim=1).values.mean().detach()

            replay_buffer.append([[state.detach(), cluster_state.detach()], action,
                                  [next_state.detach(), next_cluster_state.detach()], reward])

            tmp_epsilon += epsilon_step

            # replay_buffer full: train Q network
            if len(replay_buffer) >= args.replay_buffer_size:
                for it in range(args.Q_epochs):
                    model.eval()
                    Q_net.train()
                    optimizer_Q.zero_grad()
                    idx = list(range(args.replay_buffer_size))
                    np.random.shuffle(idx)
                    loss_Q = 0
                    for i in idx:
                        s = replay_buffer[i][0][0]
                        s_c = replay_buffer[i][0][1]
                        a = replay_buffer[i][1]
                        s_new = replay_buffer[i][2][0]
                        s_new_c = replay_buffer[i][2][1]
                        r = replay_buffer[i][3]
                        Q_value = Q_net(s, s_c).mean(0)[a]
                        y = r + 0.1 * Q_net(s_new, s_new_c).mean(0).max()
                        loss_Q += (y - Q_value) ** 2
                    loss_Q /= len(idx)
                    # loss_Q = loss_Q ** 0.5
                    # MSE loss
                    loss_Q.backward()
                    optimizer_Q.step()
                # cleaning up
                replay_buffer = []
        predict_labels_list.append(best_predict_labels)

        df1 = pd.DataFrame(features)
        df1.to_csv(f'/home/zly/hjy_code/RGC/dataset/Bai257/ourfeature_seed_{seed}.csv', index=False)



        file = open(file_name, "a+")
        print(best_cluster, best_sc,best_db,best_ch, file=file)
        file.close()
        sc_list.append(best_sc)
        k_list.append(best_cluster)
        # Save best labels for the current seed
        labels_df = pd.DataFrame(predict_labels, columns=[f'Seed_{seed}'])
        labels_df.to_csv(f'/home/zly/hjy_code/RGC/dataset/Bai257/ourslabel_seed_{seed}.csv', index=False)

        tqdm.write("Optimization Finished!")
        tqdm.write('best_sc: {},best_db:{},best_ch:{}'.format(best_sc,best_db,best_ch))

    sc_list = np.array(sc_list)
    k_list = np.array(k_list)

    file = open(file_name, "a+")
    print(args.gnnlayers, args.lr, file=file)
    print(k_list.mean(), k_list.std(), file=file)
    print(sc_list.mean(), sc_list.std(), file=file)
    file.close()
