import torch
import torch.nn as nn
import torch.nn.functional as F


class RecurrentModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RecurrentModel, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        
        # Forward pass through GRU layer
        out, _ = self.gru(x, h0)
        
        return F.relu(out[:, -1, :])


class BinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_prob=0.3):
        super(BinaryClassifier, self).__init__()
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


class BinaryFlowClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_prob=0.2):
        super(BinaryFlowClassifier, self).__init__()
        self.normalizer = nn.BatchNorm1d(input_size)
        self.rnn = RecurrentModel(input_size, hidden_size)
        self.classifier = BinaryClassifier(hidden_size, hidden_size, dropout_prob)

    def forward(self, x):
        

        # nn.BatchNorm1d ingests (N,C,L), where N is the batch size, 
        # C is the number of features or channels, and L is the sequence length
        x = self.normalizer(x.permute((0,2,1))).permute((0,2,1))
        
        x = self.rnn(x)
        x = self.classifier(x)
        return x