import torch
import random
from collections import deque



class ReplayBuffer():
    
    def __init__(self, capacity, batch_size, seed):
        self.batch_size = batch_size
        self.buffer = deque(maxlen=capacity)
        random.seed(seed)


    def push(self, flow_state, packet_state, label, zda_label, test_zda_label):
        self.buffer.append((flow_state, packet_state, label, zda_label, test_zda_label))


    def sample(self, num_of_samples):
        
        batch = random.sample(self.buffer, num_of_samples)

        flow_state_batch, packet_state_batch, label_batch, zda_label_batch, test_zda_label_batch = zip(*batch)

        if packet_state_batch[0] is None:
            return torch.vstack(flow_state_batch), \
                None, \
                    torch.vstack(label_batch), \
                        torch.vstack(zda_label_batch), \
                            torch.vstack(test_zda_label_batch)  

        return torch.vstack(flow_state_batch), \
            torch.vstack(packet_state_batch), \
                torch.vstack(label_batch), \
                    torch.vstack(zda_label_batch), \
                        torch.vstack(test_zda_label_batch)
        

    def __len__(self):
        return len(self.buffer)