from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False, batch_norm=False):
        super(GCNLayer, self).__init__()
        self.weight = torch.Tensor(in_features, out_features)
        self.weight = nn.Parameter(nn.init.xavier_uniform_(self.weight))
        if bias:
            self.bias = torch.Tensor(out_features)
            self.bias = nn.Parameter(nn.init.xavier_uniform_(self.bias))
        else:
            self.register_parameter('bias', None)

        self.bn = nn.BatchNorm1d(out_features) if batch_norm else None

    def forward(self, input, adj, batch_norm=True):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)

        if self.bias is not None:
            output = output + self.bias

        if self.bn is not None and batch_norm:
            output = self.compute_bn(output)

        return output

    def compute_bn(self, x):
        if len(x.shape) == 2:
            return self.bn(x)
        else:
            return self.bn(x.view(-1, x.size(-1))).view(x.size())


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, graph_hops, dropout, batch_norm=False):
        super(GCN, self).__init__()
        self.dropout = dropout

        self.graph_encoders = nn.ModuleList()
        self.graph_encoders.append(GCNLayer(nfeat, nhid, batch_norm=batch_norm))

        for _ in range(graph_hops - 2):
            self.graph_encoders.append(GCNLayer(nhid, nhid, batch_norm=batch_norm))

        self.graph_encoders.append(GCNLayer(nhid, nclass, batch_norm=False))

    def forward(self, x, node_anchor_adj):
        for i, encoder in enumerate(self.graph_encoders[:-1]):
            x = F.relu(encoder(x, node_anchor_adj))
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.graph_encoders[-1](x, node_anchor_adj)
        return x


class MLP(nn.Module):
    def __init__(self, inp_dim, hidden_dim, num_layers, dropout=0):
        super(MLP, self).__init__()
        layer_list = OrderedDict()
        in_dim = inp_dim
        for l in range(num_layers):
            layer_list['fc{}'.format(l)] = nn.Linear(in_dim, hidden_dim)
            if l < num_layers - 1:
                layer_list['relu{}'.format(l)] = nn.LeakyReLU()
                if dropout > 0:
                    layer_list['drop{}'.format(l)] = nn.Dropout(p=dropout)
            in_dim = hidden_dim
        if num_layers > 0:
            self.network = nn.Sequential(layer_list)
        else:
            self.network = nn.Identity()

    def forward(self, emb):
        out = self.network(emb)
        return out


class GINLayer(nn.Module):
    def __init__(self, in_features, out_features, mlp_layer=2, dropout=0, init_eps=0, learn_eps=False):
        super(GINLayer, self).__init__()
        self.mlp = MLP(in_features, out_features, mlp_layer, dropout)

        if learn_eps:
            self.eps = torch.nn.Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer('eps', torch.FloatTensor([init_eps]))

    def forward(self, x, adj):
        x = (1 + self.eps) * x + torch.matmul(adj, x)
        return self.mlp(x)


class GIN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, graph_hops, dropout=0, init_eps=0, learn_eps=False):
        super(GIN, self).__init__()

        self.graph_encoders = nn.ModuleList()
        self.graph_encoders.append(GINLayer(nfeat, nhid, dropout=dropout, init_eps=init_eps, learn_eps=learn_eps))

        for _ in range(graph_hops - 2):
            self.graph_encoders.append(GINLayer(nhid, nhid, dropout=dropout, init_eps=init_eps, learn_eps=learn_eps))

        self.graph_encoders.append(GINLayer(nhid, nclass, dropout=dropout, init_eps=init_eps, learn_eps=learn_eps))

    def forward(self, x, adj):
        for i, encoder in enumerate(self.graph_encoders[:-1]):
            x = encoder(x, adj)

        x = self.graph_encoders[-1](x, adj)
        return x
