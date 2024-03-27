# This file is part of the "Smartville" project.
# Copyright (c) 2024 University of Insubria
# Licensed under the Apache License 2.0.
# SPDX-License-Identifier: Apache-2.0
# For the full text of the license, visit:
# https://www.apache.org/licenses/LICENSE-2.0

# Smartville is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# Apache License 2.0 for more details.

# You should have received a copy of the Apache License 2.0
# along with Smartville. If not, see <https://www.apache.org/licenses/LICENSE-2.0>.

# Additional licensing information for third-party dependencies
# used in this file can be found in the accompanying `NOTICE` file.
import torch
import random
from collections import deque



class ReplayBuffer():
    
    def __init__(self, capacity, batch_size, seed):
        self.batch_size = batch_size
        self.buffer = deque(maxlen=capacity)
        random.seed(seed)


    def push(self, flow_state, packet_state, node_state, label, zda_label, test_zda_label):
        self.buffer.append((flow_state, packet_state, node_state, label, zda_label, test_zda_label))


    def sample(self, num_of_samples):
        
        batch = random.sample(self.buffer, num_of_samples)

        flow_state_batch, packet_state_batch, node_state_batch, label_batch, zda_label_batch, test_zda_label_batch = zip(*batch)

        if packet_state_batch[0] is None:
            if node_state_batch[0] is None:
                return torch.vstack(flow_state_batch), \
                    None, \
                    None, \
                        torch.vstack(label_batch), \
                            torch.vstack(zda_label_batch), \
                                torch.vstack(test_zda_label_batch)
            else:
                return torch.vstack(flow_state_batch), \
                    None, \
                    torch.vstack(node_state_batch), \
                        torch.vstack(label_batch), \
                            torch.vstack(zda_label_batch), \
                                torch.vstack(test_zda_label_batch)             
        else:
            if node_state_batch[0] is None:
                return torch.vstack(flow_state_batch), \
                    torch.vstack(packet_state_batch), \
                    None, \
                        torch.vstack(label_batch), \
                            torch.vstack(zda_label_batch), \
                                torch.vstack(test_zda_label_batch)
            else:
                return torch.vstack(flow_state_batch), \
                    torch.vstack(packet_state_batch), \
                    torch.vstack(node_state_batch), \
                        torch.vstack(label_batch), \
                            torch.vstack(zda_label_batch), \
                                torch.vstack(test_zda_label_batch)      

    def __len__(self):
        return len(self.buffer)