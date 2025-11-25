import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2, concat=True):
        super().__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  # [N, out_features]
        N = Wh.size(0)  # 节点数

        # 计算注意力分数
        a_input = torch.cat([Wh.repeat(1, N).view(N * N, -1), 
                             Wh.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))  # [N, N]

        # 只对邻接矩阵中的有效连接计算注意力
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)

        # 加权和
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)  # 多头时用 ELU
        else:
            return h_prime  # 最后输出层不需要 ELU
    
    def __repr__(self):
        return self.__class__.__name__+"("+str(self.in_features)+"->"+str(self.out_features)+")"
    
    
class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.1, alpha=0.2, nheads=8):
        super().__init__()
        self.dropout = dropout
        self.attentions = nn.ModuleList([GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)])
            
        self.out_att = GraphAttentionLayer(nhid*nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)