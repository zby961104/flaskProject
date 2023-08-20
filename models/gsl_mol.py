import torch
from torch import nn
from .gnn import GCN
from .graphlearn import GraphLearner
import torch.nn.functional as F
from utils.data_process import process_smiles


VERY_SMALL_NUMBER = 1e-12


class GslMolNet(nn.Module):
    def __init__(self, opt, criterion):
        super(GslMolNet, self).__init__()
        self.opt = opt
        self.device = torch.device(
            f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if torch.cuda.is_available() else 'cpu')

        self.criterion = criterion
        self.gnn_type = opt.args['gnn_type']
        self.graph_learn = opt.args['graph_learn']
        self.graph_metric_type = opt.args['graph_learn_type']
        self.graph_skip_conn = opt.args['graph_skip_conn']
        self.graph_include_self = opt.args['graph_include_self']
        self.dropout = opt.args['dropout']
        input_dim = opt.args['AtomFeatureSize']
        emb_dim = opt.args['emb_dim']

        if self.gnn_type == 'gcn':
            self.encoder = GCN(nfeat=input_dim, nhid=emb_dim, nclass=emb_dim,
                               graph_hops=opt.args['gnn_layer'], dropout=self.dropout,
                               batch_norm=opt.args['gnn_batch_norm'])

        if self.graph_learn:
            self.graph_learner = GraphLearner(input_size=input_dim,
                                              hidden_size=opt.args['graph_learn_hidden_size'],
                                              topk=opt.args['graph_learn_topk'],
                                              epsilon=opt.args['graph_learn_epsilon'],
                                              num_pers=opt.args['graph_learn_num_pers'],
                                              metric_type=opt.args['graph_learn_type'],
                                              device=self.device)

            self.graph_learner2 = GraphLearner(input_size=emb_dim,
                                              hidden_size=opt.args['graph_learn_hidden_size'],
                                              topk=opt.args['graph_learn_topk'],
                                              epsilon=opt.args['graph_learn_epsilon'],
                                              num_pers=opt.args['graph_learn_num_pers'],
                                              metric_type=opt.args['graph_learn_type'],
                                              device=self.device)

        else:
            self.graph_learner = None

        self.linear_out = nn.Linear(emb_dim, opt.args['ClassNum'], bias=False)

    def learn_graph(self, graph_learner, node_features, graph_skip_conn=None, node_mask=None, graph_include_self=False, init_adj=None):
        if self.graph_learn:
            raw_adj = graph_learner(node_features, node_mask)

            if self.graph_metric_type in ('kernel', 'weighted_cosine'):
                assert raw_adj.min().item() >= 0
                adj = raw_adj / torch.clamp(torch.sum(raw_adj, dim=-1, keepdim=True), min=VERY_SMALL_NUMBER)

            elif self.graph_metric_type == 'cosine':
                adj = (raw_adj > 0).float()
                adj = self.normalize_adj(adj)

            else:
                adj = torch.softmax(raw_adj, dim=-1)

            if graph_skip_conn in (0, None):
                if graph_include_self:
                    adj = adj + torch.eye(adj.size(0)).to(self.device)
            else:
                adj = graph_skip_conn * init_adj + (1 - graph_skip_conn) * adj

            return raw_adj, adj

        else:
            raw_adj = None
            adj = init_adj

            return raw_adj, adj

    def normalize_adj(self, mx):
        """Row-normalize matrix: symmetric normalized Laplacian"""
        rowsum = mx.sum(1)
        r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
        r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.
        r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
        return torch.mm(torch.mm(mx, r_mat_inv_sqrt).transpose(-1, -2), r_mat_inv_sqrt)

    def compute_output(self, node_vec, node_mask=None):
        if self.opt.args['graph_pooling'] == 'max':
            graph_pool = self.graph_maxpool
        elif self.opt.args['graph_pooling'] == 'mean':
            graph_pool = self.graph_meanpool
        elif self.opt.args['graph_pooling'] == 'add':
            graph_pool = self.graph_addpool
        else:
            raise ValueError("Invalid graph pooling type.")

        graph_vec = graph_pool(node_vec.transpose(-1, -2), node_mask=node_mask)
        output = self.linear_out(graph_vec)
        return output

    def graph_maxpool(self, node_vec, node_mask=None):
        # Maxpool
        # Shape: (batch_size, hidden_size, num_nodes)
        graph_embedding = F.max_pool1d(node_vec, kernel_size=node_vec.size(-1)).squeeze(-1)
        return graph_embedding

    def graph_meanpool(self, node_vec, node_mask=None):
        graph_embedding = F.avg_pool1d(node_vec, kernel_size=node_vec.size(-1)).squeeze(-1)
        return graph_embedding

    def graph_addpool(self, node_vec, node_mask=None):
        graph_embedding = torch.sum(node_vec, dim=-1)
        return graph_embedding

    def CalculateLoss(self, output, Label, criterion):
        loss = 0.0
        if self.opt.args['ClassNum'] != 1:
            for i in range(self.opt.args['TaskNum']):
                cur_task_output = output[:, i * self.opt.args['ClassNum']: (i + 1) * self.opt.args['ClassNum']]     # select the output of the current task i
                cur_task_label = Label[i]                               # [batch], the label of the current task i
                valid_index = (cur_task_label != -1)                    # Not all of the samples have labels of the current task i.
                valid_label = cur_task_label[valid_index]               # Only the samples that have labels of the current task i will participate in the loss calculation.
                if len(valid_label) == 0:
                    continue
                else:
                    valid_output = cur_task_output[valid_index]
                    loss += criterion[i](valid_output, valid_label)
        else:
            for i in range(self.opt.args['TaskNum']):
                cur_task_output = output[:, i * self.opt.args['ClassNum']: (i + 1) * self.opt.args['ClassNum']]
                cur_task_label = Label[i].unsqueeze(-1)
                loss += criterion[i](cur_task_output, cur_task_label)

        return loss

    # def batch_run(self, data, targets):
    #
    #     init_node_vec, init_bond_vec, init_adj, node_mask = data
    #
    #     cur_raw_adj, cur_adj = self.learn_graph(self.graph_learner, init_node_vec, self.graph_skip_conn,
    #                                                node_mask=node_mask, graph_include_self=self.graph_include_self,
    #                                                init_adj=init_adj)
    #     node_vec = torch.relu(self.encoder.graph_encoders[0](init_node_vec, cur_adj))
    #     node_vec = F.dropout(node_vec, self.dropout, training=self.training)
    #
    #     # Add mid GNN layers
    #     for encoder in self.encoder.graph_encoders[1:-1]:
    #         node_vec = torch.relu(encoder(node_vec, cur_adj))
    #         node_vec = F.dropout(node_vec, self.dropout, training=self.training)
    #
    #     # BP to update weights
    #     output = self.encoder.graph_encoders[-1](node_vec, cur_adj)
    #     output = self.compute_output(output, node_mask=node_mask)
    #     loss1 = self.CalculateLoss(output, targets, self.criterion)
    #
    #     first_raw_adj, first_adj = cur_raw_adj, cur_adj
    #
    #     loss = 0
    #     iter_ = 0
    #     max_iter_ = self.opt.args['max_iter']
    #
    #     while self.opt.args['graph_learn'] and iter_ < max_iter_:
    #         iter_ += 1
    #
    #         cur_raw_adj, cur_adj = self.learn_graph(self.graph_learner2, node_vec, self.graph_skip_conn,
    #                                                    node_mask=node_mask,
    #                                                    graph_include_self=self.graph_include_self, init_adj=init_adj)
    #
    #         update_adj_ratio = self.opt.get_args('update_adj_ratio', None)
    #         if update_adj_ratio is not None:
    #             cur_adj = update_adj_ratio * cur_adj + (1 - update_adj_ratio) * first_adj
    #
    #         node_vec = torch.relu(self.encoder.graph_encoders[0](init_node_vec, cur_adj))
    #         node_vec = F.dropout(node_vec, self.opt.get_args('gl_dropout', 0), training=self.training)
    #
    #         # Add mid GNN layers
    #         for encoder in self.encoder.graph_encoders[1:-1]:
    #             node_vec = torch.relu(encoder(node_vec, cur_adj))
    #             node_vec = F.dropout(node_vec, self.opt.get_args('gl_dropout', 0), training=self.training)
    #
    #         # BP to update weights
    #         output = self.encoder.graph_encoders[-1](node_vec, cur_adj)
    #         output = self.compute_output(output, node_mask=node_mask)
    #         loss += self.CalculateLoss(output, targets, self.criterion)
    #
    #     if iter_ > 0:
    #         loss = loss / iter_ + loss1
    #     else:
    #         loss = loss1
    #
    #     return output, loss

    def forward(self, smiles):

        print(smiles)
        init_node_vec, init_adj = process_smiles(smiles)
        print("init_node_vec:", init_node_vec.shape)
        print("init_adj:", init_adj.shape)

        cur_raw_adj, cur_adj = self.learn_graph(self.graph_learner, init_node_vec, self.graph_skip_conn,
                                                graph_include_self=self.graph_include_self,
                                                init_adj=init_adj)
        node_vec = torch.relu(self.encoder.graph_encoders[0](init_node_vec, cur_adj))
        node_vec = F.dropout(node_vec, self.dropout, training=self.training)

        # Add mid GNN layers
        for encoder in self.encoder.graph_encoders[1:-1]:
            node_vec = torch.relu(encoder(node_vec, cur_adj))
            node_vec = F.dropout(node_vec, self.dropout, training=self.training)

        # BP to update weights
        output = self.encoder.graph_encoders[-1](node_vec, cur_adj)
        output = self.compute_output(output)

        first_raw_adj, first_adj = cur_raw_adj, cur_adj

        iter_ = 0
        max_iter_ = self.opt.args['max_iter']

        while self.opt.args['graph_learn'] and iter_ < max_iter_:
            iter_ += 1

            cur_raw_adj, cur_adj = self.learn_graph(self.graph_learner2, node_vec, self.graph_skip_conn,
                                                    graph_include_self=self.graph_include_self, init_adj=init_adj)

            update_adj_ratio = self.opt.get_args('update_adj_ratio', None)
            if update_adj_ratio is not None:
                cur_adj = update_adj_ratio * cur_adj + (1 - update_adj_ratio) * first_adj

            node_vec = torch.relu(self.encoder.graph_encoders[0](init_node_vec, cur_adj))
            node_vec = F.dropout(node_vec, self.opt.get_args('gl_dropout', 0), training=self.training)

            # Add mid GNN layers
            for encoder in self.encoder.graph_encoders[1:-1]:
                node_vec = torch.relu(encoder(node_vec, cur_adj))
                node_vec = F.dropout(node_vec, self.opt.get_args('gl_dropout', 0), training=self.training)

            # BP to update weights
            output = self.encoder.graph_encoders[-1](node_vec, cur_adj)
            output = self.compute_output(output)

        return output




