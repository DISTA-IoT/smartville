import torch
import torch.nn as nn
import torch.nn.functional as F


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
        x = self.fc2(x)
        # Apply sigmoid activation to obtain probabilities
        x = torch.sigmoid(x)
        return x


class MulticlassPrototypicalClassifier(nn.Module):

    def __init__(self, device='cpu'):
        super(MulticlassPrototypicalClassifier, self).__init__()
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
    

    def forward(self, hidden_vectors, labels, known_attacks_count, query_mask):
        """
        known_attacks_count is the current number of known attacks. 
        """
        # get one_hot_labels of current batch:
        oh_labels = self.get_oh_labels(
            decimal_labels=labels.long(),
            n_way=known_attacks_count)

        # get latent centroids:
        centroids, _ = self.get_centroids(
            hidden_vectors[~query_mask],
            oh_labels[~query_mask])

        # compute scores:
        scores = 1 / (torch.cdist(hidden_vectors[query_mask], centroids) + 1e-10)

        if scores.shape[0] == 0:
            print('hello')
        return scores


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
        x = self.classifier(x)
        return x
    

class MultiClassFlowClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_prob=0.2, device='cpu'):
        super(MultiClassFlowClassifier, self).__init__()
        self.device=device
        self.normalizer = nn.BatchNorm1d(input_size)
        self.rnn = RecurrentModel(input_size, hidden_size, device=self.device)
        self.classifier = MulticlassPrototypicalClassifier(device=self.device)

    def forward(self, x, labels, curr_known_attack_count, query_mask):
        # nn.BatchNorm1d ingests (N,C,L), where N is the batch size, 
        # C is the number of features or channels, and L is the sequence length
        x = self.normalizer(x.permute((0,2,1))).permute((0,2,1))
        x = self.rnn(x)
        x = self.classifier(x, labels, curr_known_attack_count, query_mask)
        return x


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

        inferences = self.classifier(torch.cat([flows, packets], dim=1))

        return inferences
    

class TwoStreamMulticlassFlowClassifier(nn.Module):
    def __init__(self, flow_input_size, packet_input_size, hidden_size, dropout_prob=0.2, device='cpu'):
        super(TwoStreamMulticlassFlowClassifier, self).__init__()
        self.device = device
        self.flow_normalizer = nn.BatchNorm1d(flow_input_size)
        self.flow_rnn = RecurrentModel(flow_input_size, hidden_size, device=self.device)
        self.packet_normalizer = nn.BatchNorm1d(packet_input_size)
        self.packet_rnn = RecurrentModel(packet_input_size, hidden_size, device=self.device)
        self.classifier = MulticlassPrototypicalClassifier(device=self.device)

    def forward(self, flows, packets, labels, curr_known_attack_count, query_mask):
        
        flows = self.flow_normalizer(flows.permute((0,2,1))).permute((0,2,1))
        packets = self.packet_normalizer(packets.permute((0,2,1))).permute((0,2,1))

        flows = self.flow_rnn(flows)
        packets = self.packet_rnn(packets)

        inferences = self.classifier(torch.cat([flows, packets], dim=1), labels, curr_known_attack_count, query_mask)

        return inferences