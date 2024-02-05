import torch
import random
from collections import deque

class ReplayBuffer():
    def __init__(self, capacity, batch_size, seed):
        self.batch_size = batch_size
        self.buffer = deque(maxlen=capacity)
        random.seed(seed)

    def push(self, state, label):
        self.buffer.append((state, label))

    def sample(self):
        if len(self.buffer) < self.batch_size:
            batch = random.sample(self.buffer, len(self.buffer))
        else:
            batch = random.sample(self.buffer, self.batch_size)
        state_batch, label_batch = zip(*batch)
        return torch.vstack(state_batch), torch.vstack(label_batch)

    def __len__(self):
        return len(self.buffer)