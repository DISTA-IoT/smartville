import torch
import random
from collections import deque

class ReplayBuffer():
    def __init__(self, capacity, batch_size, seed):
        self.batch_size = batch_size
        self.buffer = deque(maxlen=capacity)
        random.seed(seed)

    def push(self, flow_state, packet_state, label):
        self.buffer.append((flow_state, packet_state, label))

    def sample(self):
        if len(self.buffer) < self.batch_size:
            batch = random.sample(self.buffer, len(self.buffer))
        else:
            batch = random.sample(self.buffer, self.batch_size)
        
        flow_state_batch, packet_state_batch, label_batch = zip(*batch)

        return torch.vstack(flow_state_batch), torch.vstack(packet_state_batch), torch.vstack(label_batch)

    def __len__(self):
        return len(self.buffer)