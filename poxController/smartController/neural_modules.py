import torch
import torch.nn as nn
import torch.nn.functional as F


def get_collaborative_kernel(
    n_elements, 
    one_hot_masked_labels, 
    device):

    # kernel init
    kernel = torch.ones(
        [n_elements, n_elements],
        device=device)

    # unknown mask
    unknown_mask = one_hot_masked_labels.sum(1) == 0
    # known mask:
    known_mask = ~unknown_mask
    # we need tensors indicating the indexes to fill the kernel
    known_indexes_tensor = known_mask\
        .unsqueeze(0)\
        .repeat(n_elements, 1)
    # mask the unknown rows to False
    known_indexes_tensor[unknown_mask] = False

    # build semantic kernel:
    semantic_kernel = one_hot_masked_labels @ one_hot_masked_labels.T
    # fill the known region of the kernel with the semantic kernel:
    kernel = torch.where(
        condition=known_indexes_tensor,
        input=semantic_kernel,
        other=kernel)
    return kernel


class RecurrentModel(nn.Module):

    def __init__(self, input_size, hidden_size, device='cpu'):
        super(RecurrentModel, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        
        # Forward pass through GRU layer
        out, _ = self.gru(x, h0)
        
        return F.relu(out[:, -1, :])


class BinaryMLPClassifier(nn.Module):

    def __init__(self, input_size, hidden_size, dropout_prob=0.3, device='cpu'):
        super(BinaryMLPClassifier, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)  # Dropout layer with dropout probability
        self.fc2 = nn.Linear(hidden_size, 1)  # Output size is 1 for binary classification

    def forward(self, x):
        # Apply first fully connected layer with ReLU activation
        x = F.relu(self.fc1(x))
        # Apply dropout layer
        x = self.dropout(x)
        # Apply second fully connected layer without activation
        hiddens = self.fc2(x)
        # Apply sigmoid activation to obtain probabilities
        preds = torch.sigmoid(hiddens)
        return preds, hiddens


class MulticlassPrototypicalClassifier(nn.Module):

    def __init__(self, in_features, out_features, dropout, device='cpu'):
        super(MulticlassPrototypicalClassifier, self).__init__()

        self.kernel_regressor = GraphAttentionV2Layer(
            in_features=in_features,
            out_features=out_features,
            n_heads=1,
            is_concat=True,
            dropout=dropout,
            leaky_relu_negative_slope=0.2,
            share_weights=True,
            colaborative=True,
            )
        
        self.device = device


    def get_oh_labels(
        self,
        decimal_labels,
        n_way):

        # create placeholder for one_hot encoding:
        labels_onehot = torch.zeros(
            [decimal_labels.size()[0],
            n_way], device=self.device)
        # transform to one_hot encoding:
        labels_onehot = labels_onehot.scatter(
            1,
            decimal_labels,
            1)
        return labels_onehot


    def get_centroids(
            self,
            hidden_vectors,
            onehot_labels):

        cluster_agg = onehot_labels.T @ hidden_vectors
        samples_per_cluster = onehot_labels.sum(0)
        centroids = torch.zeros_like(cluster_agg, device=self.device)
        missing_clusters = samples_per_cluster == 0
        existent_centroids = cluster_agg[~missing_clusters] / \
            samples_per_cluster[~missing_clusters].unsqueeze(-1)
        centroids[~missing_clusters] = existent_centroids
        assert torch.all(centroids[~missing_clusters] == existent_centroids)
        return centroids, missing_clusters
    


    def get_one_hot_masked_labels(
            self,
            one_hot_labels,
            unknown_mask):
        # compute one_hot_masked_cluster_labels:
        ohm_cluster_labels = \
            torch.zeros_like(one_hot_labels, device=self.device)

        ohm_cluster_labels[~unknown_mask.squeeze(-1)] = \
            one_hot_labels[~unknown_mask.squeeze(-1)]
        return ohm_cluster_labels


    def forward(self, hidden_vectors, labels, known_attacks_count, query_mask, zda_mask):
        """
        known_attacks_count is the current number of known attacks. 
        """
        # get one_hot_labels of current batch:
        oh_labels = self.get_oh_labels(
            decimal_labels=labels.long(),
            n_way=known_attacks_count)
        
        one_hot_masked_labels = self.get_one_hot_masked_labels(
            one_hot_labels=oh_labels,
            unknown_mask=zda_mask.bool())
        
        collaborative_kernel = get_collaborative_kernel(
            n_elements=oh_labels.shape[0],
            one_hot_masked_labels=one_hot_masked_labels,
            device=self.device)
        
        # This could be a gat:
        hidden_vectors, predicted_kernel = self.kernel_regressor(
            hidden_vectors, 
            collaborative_kernel,
            one_hot_masked_labels)
        

        # get latent centroids:
        centroids, _ = self.get_centroids(
            hidden_vectors[~query_mask],
            oh_labels[~query_mask])

        # compute scores:
        scores = 1 / (torch.cdist(hidden_vectors[query_mask], centroids) + 1e-10)

        return scores, hidden_vectors, predicted_kernel


class BinaryFlowClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_prob=0.2, device='cpu'):
        super(BinaryFlowClassifier, self).__init__()
        self.device = device
        self.normalizer = nn.BatchNorm1d(input_size)
        self.rnn = RecurrentModel(input_size, hidden_size, device=self.device)
        self.classifier = BinaryMLPClassifier(hidden_size, hidden_size, dropout_prob, device=self.device)

    def forward(self, x):
        # nn.BatchNorm1d ingests (N,C,L), where N is the batch size, 
        # C is the number of features or channels, and L is the sequence length
        x = self.normalizer(x.permute((0,2,1))).permute((0,2,1))
        x = self.rnn(x)
        preds, hiddens =  self.classifier(x)
        return preds, hiddens, None
    

class MultiClassFlowClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_prob=0.2, device='cpu'):
        super(MultiClassFlowClassifier, self).__init__()
        self.device=device
        self.normalizer = nn.BatchNorm1d(input_size)
        self.rnn = RecurrentModel(input_size, hidden_size, device=self.device)
        self.classifier = MulticlassPrototypicalClassifier(
            in_features=hidden_size,
            out_features=2,
            dropout=dropout_prob,
            device=self.device)

    def forward(self, x, labels, curr_known_attack_count, query_mask, zda_mask):
        # nn.BatchNorm1d ingests (N,C,L), where N is the batch size, 
        # C is the number of features or channels, and L is the sequence length
        x = self.normalizer(x.permute((0,2,1))).permute((0,2,1))
        hiddens = self.rnn(x)
        scores, hiddens, predicted_kernel = self.classifier(hiddens, labels, curr_known_attack_count, query_mask, zda_mask)
        return scores, hiddens.detach(), predicted_kernel


class TwoStreamBinaryFlowClassifier(nn.Module):
    def __init__(self, flow_input_size, packet_input_size, hidden_size, dropout_prob=0.2, device='cpu'):
        super(TwoStreamBinaryFlowClassifier, self).__init__()
        self.device = device
        self.flow_normalizer = nn.BatchNorm1d(flow_input_size, device=self.device)
        self.flow_rnn = RecurrentModel(flow_input_size, hidden_size, device=self.device)
        self.packet_normalizer = nn.BatchNorm1d(packet_input_size)
        self.packet_rnn = RecurrentModel(packet_input_size, hidden_size, device=self.device)
        self.classifier = BinaryMLPClassifier(hidden_size*2, hidden_size, dropout_prob, device=self.device)

    def forward(self, flows, packets):
        
        flows = self.flow_normalizer(flows.permute((0,2,1))).permute((0,2,1))
        packets = self.packet_normalizer(packets.permute((0,2,1))).permute((0,2,1))

        flows = self.flow_rnn(flows)
        packets = self.packet_rnn(packets)

        inferences, hiddens = self.classifier(torch.cat([flows, packets], dim=1))

        return inferences, hiddens, None
    

class TwoStreamMulticlassFlowClassifier(nn.Module):
    def __init__(self, flow_input_size, packet_input_size, hidden_size, dropout_prob=0.2, device='cpu'):
        super(TwoStreamMulticlassFlowClassifier, self).__init__()
        self.device = device
        self.flow_normalizer = nn.BatchNorm1d(flow_input_size)
        self.flow_rnn = RecurrentModel(flow_input_size, hidden_size, device=self.device)
        self.packet_normalizer = nn.BatchNorm1d(packet_input_size)
        self.packet_rnn = RecurrentModel(packet_input_size, hidden_size, device=self.device)
        self.classifier = MulticlassPrototypicalClassifier(
            in_features=hidden_size*2,
            out_features=2,
            dropout=dropout_prob,
            device=self.device)

    def forward(self, flows, packets, labels, curr_known_attack_count, query_mask, zda_mask):
        
        flows = self.flow_normalizer(flows.permute((0,2,1))).permute((0,2,1))
        packets = self.packet_normalizer(packets.permute((0,2,1))).permute((0,2,1))

        flows = self.flow_rnn(flows)
        packets = self.packet_rnn(packets)

        hiddens = torch.cat([flows, packets], dim=1)

        preds, hiddens, predicted_kernel = self.classifier(hiddens, labels, curr_known_attack_count, query_mask, zda_mask)

        return preds, hiddens.detach(), predicted_kernel

    

class ConfidenceDecoder(nn.Module):

    def __init__(
            self,
            device):

        super(ConfidenceDecoder, self).__init__()
        self.rnn = RecurrentModel(input_size=1,hidden_size=1,device=device)
        self.device = device


    def forward(
            self,
            scores):

        scores = self.rnn(scores.unsqueeze(-1))

        unknown_indicators = torch.sigmoid(scores)
        return unknown_indicators
    

class GraphAttentionV2Layer(nn.Module):

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 n_heads: int,
                 is_concat: bool = False,
                 dropout: float = 0.6,
                 leaky_relu_negative_slope: float = 0.2,
                 share_weights: bool = True,
                 colaborative: bool = False):

        super().__init__()

        self.is_concat = is_concat
        self.n_heads = n_heads
        self.share_weights = share_weights
        self.colaborative = colaborative

        # Calculate the number of dimensions per head
        if is_concat:
            assert out_features % n_heads == 0
            self.n_hidden = out_features // n_heads
        else:
            self.n_hidden = out_features

        self.linear_l = nn.Linear(
            in_features,
            self.n_hidden * n_heads,
            bias=False)

        if share_weights:
            self.linear_r = self.linear_l
        else:
            self.linear_r = nn.Linear(
                in_features,
                self.n_hidden * n_heads,
                bias=False)

        # Linear layer to compute attention score $e_{ij}$
        self.attn = nn.Linear(
            self.n_hidden,
            1,
            bias=False)

        # The activation for attention score $e_{ij}$
        self.activation = nn.LeakyReLU(
            negative_slope=leaky_relu_negative_slope)

        # Softmax to compute attention $\alpha_{ij}$
        self.softmax = nn.Softmax(dim=1)

        # Dropout layer to be applied for attention
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                h: torch.Tensor,
                adj_mat: torch.Tensor,
                masked_oh_labels: torch.Tensor):

        # Number of nodes
        n_nodes = h.shape[0]
        # unknown mask
        unknown_mask = masked_oh_labels.sum(1) == 0
        # known mask:
        known_mask = ~unknown_mask

        # The initial GAT transformations,
        # We do two linear transformations and then split it up for each head.
        g_l = self.linear_l(h).view(
            n_nodes,
            self.n_heads,
            self.n_hidden)

        g_r = self.linear_r(h).view(
            n_nodes,
            self.n_heads,
            self.n_hidden)

        # #### Calculate attention score
        g_l_repeat = g_l.repeat(
            n_nodes,
            1,
            1)

        g_r_repeat_interleave = g_r.repeat_interleave(
            n_nodes,
            dim=0)

        g_sum = g_l_repeat + g_r_repeat_interleave

        g_sum = g_sum.view(
            n_nodes,
            n_nodes,
            self.n_heads,
            self.n_hidden)

        # get energies
        e = self.attn(self.activation(g_sum))
        e = e.squeeze(-1)

        # Introduce the adj_mat
        assert adj_mat.shape[0] == n_nodes
        assert adj_mat.shape[1] == n_nodes
        adj_mat = adj_mat.unsqueeze(-1)
        adj_mat = adj_mat.repeat(1, 1, self.n_heads)

        if self.colaborative:
            e = e.masked_fill(adj_mat == 0, float('-inf'))
        else:
            # mask homogeneous regions
            e[known_mask][:, unknown_mask] = torch.zeros_like(
                e[known_mask][:, unknown_mask])
            e[unknown_mask][:, known_mask] = torch.zeros_like(
                e[unknown_mask][:, known_mask])

            # scale energies between 0 and 0.5:
            e = e + e.min().abs()
            e = e / (2 * e.max())

            # fill homogeneous regions
            e[known_mask][:, unknown_mask] = \
                adj_mat[known_mask][:, unknown_mask]
            e[unknown_mask][:, known_mask] = \
                adj_mat[unknown_mask][:, known_mask]

        # Normalization
        a = self.softmax(e)
        a = self.dropout(a)

        # Calculate final output for each head
        attn_res = torch.einsum('ijh,jhf->ihf', a, g_r)

        # Concatenate the heads
        if self.is_concat:
            return attn_res.reshape(n_nodes, self.n_heads * self.n_hidden), a.mean(dim=2)
        # Take the mean of the heads
        else:
            return attn_res.mean(dim=1), a.mean(dim=2)

