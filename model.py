import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import *
from GAT.gat import GAT


class my_model(nn.Module):
    def __init__(self, dims, adj, act="ident"):
        super(my_model, self).__init__()

        self.adj = adj

        # self.gat1 = GAT(nfeat=dims[0], nhid=16, nclass=dims[0], dropout=0.3, alpha=0.2, nheads=1)
        # self.gat2 = GAT(nfeat=dims[0], nhid=16, nclass=dims[0], dropout=0.3, alpha=0.2, nheads=1)

        self.lin1 = nn.Linear(dims[0], dims[1])
        self.lin2 = nn.Linear(dims[0], dims[1])
        self.bn = nn.BatchNorm1d(dims[1])
        self.reset_parameters()

        if act == "ident":
            self.activate = lambda x: x
        if act == "sigmoid":
            self.activate = nn.Sigmoid()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x):
        # self.adj = self.adj.to(x.device)
        # gatx1 = self.gat1(x, self.adj)
        # gatx2 = self.gat2(x, self.adj)

        out1 = self.activate(self.lin1(x))
        out2 = self.activate(self.lin2(x))

        out1 = F.normalize(out1, dim=1, p=2)
        out2 = F.normalize(out2, dim=1, p=2)

        return out1, out2

class SplineLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1, **kw) -> None:
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)


class RadialBasisFunction(nn.Module):
    def __init__(
            self,
            grid_min: float = -2.,
            grid_max: float = 2.,
            num_grids: int = 8,
            denominator: float = None,  # larger denominators lead to smoother basis
    ):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)


class FastKANLayer(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            grid_min: float = -2.,
            grid_max: float = 2.,
            num_grids: int = 8,
            use_base_update: bool = True,
            base_activation=F.silu,
            spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(input_dim)
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.spline_linear = SplineLinear(input_dim * num_grids, output_dim, spline_weight_init_scale)
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, time_benchmark=False):
        if not time_benchmark:
            spline_basis = self.rbf(self.layernorm(x))
        else:
            spline_basis = self.rbf(x)
        ret = self.spline_linear(spline_basis.view(*spline_basis.shape[:-2], -1))
        if self.use_base_update:
            base = self.base_linear(self.base_activation(x))
            ret = ret + base
        return ret


class FastKAN(nn.Module):
    def __init__(
            self,
            layers_hidden: List[int],
            grid_min: float = -2.,
            grid_max: float = 2.,
            num_grids: int = 8,
            use_base_update: bool = True,
            base_activation=F.silu,
            spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            FastKANLayer(
                in_dim, out_dim,
                grid_min=grid_min,
                grid_max=grid_max,
                num_grids=num_grids,
                use_base_update=use_base_update,
                base_activation=base_activation,
                spline_weight_init_scale=spline_weight_init_scale,
            ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# class my_model(nn.Module):
#     def __init__(self, dims, adj):
#         super(my_model, self).__init__()
#         self.kan_1 = FastKAN([dims[0], dims[1]])
#         self.kan_2 = FastKAN([dims[0], dims[1]])
#
#         self.adj = adj
#
#         self.gat1 = GAT(nfeat=dims[0], nhid=16, nclass=dims[0], dropout=0.1, alpha=0.2, nheads=1)
#         self.gat2 = GAT(nfeat=dims[0], nhid=16, nclass=dims[0], dropout=0.1, alpha=0.2, nheads=1)
#
#     def forward(self, x, is_train=True, sigma=0.01):
#
#         # self.adj = self.adj.to(x.device)
#         # gat1 = self.gat1(x, self.adj)
#         # gat2 = self.gat2(x, self.adj)
#
#         out1 = self.kan_1(x)
#         out2 = self.kan_2(x)
#
#         out1 = F.normalize(out1, dim=1, p=2)
#
#         if is_train:
#             out2 = F.normalize(out2, dim=1, p=2) + torch.normal(0, torch.ones_like(out2) * sigma).cuda()
#         else:
#             out2 = F.normalize(out2, dim=1, p=2)
#         return out1, out2


class my_model1(nn.Module):
    def __init__(self, dims, act="ident"):
        super(my_model1, self).__init__()
        self.gnn1 = GCNConv(dims[0], 20)  # 添加图卷积层
        self.gnn2 = GCNConv(dims[0], 20)  # 添加图卷积层
        self.lin1 = nn.Linear(20, dims[1])
        self.lin2 = nn.Linear(20, dims[1])
        self.bn = nn.BatchNorm1d(dims[1])
        self.reset_parameters()

        if act == "ident":
            self.activate = lambda x: x
        if act == "sigmoid":
            self.activate = nn.Sigmoid()

    def reset_parameters(self):
        self.gnn1.reset_parameters()  # 重置图卷积层的参数
        self.gnn2.reset_parameters()  # 重置图卷积层的参数
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index):
        # 第一个GCN和MLP
        gcn_out1 = self.activate(self.gnn1(x, edge_index))
        mlp_out1 = self.activate(self.lin1(gcn_out1))

        # 第二个GCN和MLP
        gcn_out2 = self.activate(self.gnn2(x, edge_index))
        mlp_out2 = self.activate(self.lin2(gcn_out2))

        # 归一化
        out1 = F.normalize(mlp_out1, dim=1, p=2)
        out2 = F.normalize(mlp_out2, dim=1, p=2)

        return out1, out2


class my_Q_net(nn.Module):
    def __init__(self, dims):
        super(my_Q_net, self).__init__()
        self.lin1 = nn.Linear(dims[0], dims[1])
        self.lin_cluster = nn.Linear(dims[0], dims[1])
        self.lin2 = nn.Linear(dims[1], dims[2])
        self.reset_parameters()
        self.act = torch.nn.ReLU()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, cluster):
        x = self.act(F.normalize(self.lin1(x), dim=1, p=2))
        cluster = self.act(F.normalize(self.lin_cluster(cluster), dim=1, p=2))
        # 根据x和cluster的特征共同确定输出层的概率（属于dim[1]每一个类别的概率）
        x = F.softmax(self.lin2(torch.cat([x, cluster], dim=0)), dim=-1)
        return x
