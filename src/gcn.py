import math
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch import nn

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            
    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
        
    def __repr__(self):
        return self.__class__.__name__ + "(" \
            + str(self.in_features) + "->" \
            + str(self.out_features) + ")"

def gen_adj_sim(f1, f2):
    B, N, C = f1.shape
    features1 = f1.view(B * N, C).contiguous()
    features2 = f2.view(B * N, C).contiguous()
    features2 = features2.t()
    edge = torch.matmul(features1, features2)
    D = torch.pow(edge.sum(1).float(), -0.5)
    D = torch.diag(D)
   
    adj = torch.matmul(torch.matmul(edge, D).t(), D)
    return adj


def normalize_adj(edge_index, num_nodes):
    """
    归一化邻接矩阵，加入自连接
    :param edge_index: 边的索引 (2, num_edges)
    :param num_nodes: 节点数量
    :return: 归一化邻接矩阵
    """
    adj = torch.zeros((num_nodes, num_nodes), device=edge_index.device)
    adj[edge_index[0], edge_index[1]] = 1  # 邻接矩阵填充边
    adj = adj + torch.eye(num_nodes, device=adj.device)
    
    # 归一化：D^(-1/2) A D^(-1/2)，A 为邻接矩阵，D 为度矩阵
    degree = adj.sum(dim=1)  # 计算每个节点的度
    degree_inv_sqrt = torch.pow(degree, -0.5)  # D^(-1/2)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0  # 防止除零
    
    adj_norm = degree_inv_sqrt.view(-1, 1) * adj * degree_inv_sqrt.view(1, -1)
    return adj_norm
    
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass):
        super().__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.threshold = 0.6
 
    
    def forward(self, x, adj):
        
        B, C, H, W = x.shape
        adj = adj.flatten(2).transpose(1, 2).contiguous().view(B*H*W, C)
        adj = F.normalize(adj,p=2,dim=1)
        adj = torch.mm(adj,adj.T)
        edge_index = ((adj > self.threshold).nonzero(as_tuple=False).T)
        edge_index = normalize_adj(edge_index, B*H*W)
        x = x.flatten(2).transpose(1, 2).contiguous().view(B*H*W, C)
        x = F.relu(self.gc1(x, edge_index))
        x = self.gc2(x, edge_index)
        x = x.view(B,H,W,C).permute(0,3,1,2).contiguous()
        return x