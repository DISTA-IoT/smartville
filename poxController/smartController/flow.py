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


class CircularBuffer:
    def __init__(self, buffer_size=10, feature_size=4):
        self.buffer_size = buffer_size
        self.feature_size = feature_size
        self.buffer = torch.zeros(buffer_size, feature_size)
        self.is_full = False
        self.calls_to_add = 0


    def add(self, new_tensor):
        # Roll the buffer up by 1 along the first dimension
        self.buffer = torch.roll(self.buffer, shifts=-1, dims=0)
        # Add new tensor to the last row
        self.buffer[-1] = new_tensor
        self.calls_to_add += 1
        if self.calls_to_add >= self.buffer_size:
            self.is_full = True

    def get_buffer(self):
        return self.buffer
    

class Flow():
    def __init__(
        self, 
        source_ip, 
        dest_ip, 
        switch_output_port,
        flow_feat_dim=4,
        flow_buff_len=10):
        
        self.source_ip = source_ip
        self.dest_ip = dest_ip
        self.switch_output_port = switch_output_port
        self.flow_id = self.source_ip + "_" + self.dest_ip + "_" + str(self.switch_output_port)
        self.switch_input_port = None
        self.__feat_tensor = CircularBuffer(
                            buffer_size=flow_buff_len, 
                            feature_size=flow_feat_dim)
        self.packets_tensor = None
        self.element_class = "Bening"
        self.zda = False
        self.test_zda = False

    def get_feat_tensor(self):
        return self.__feat_tensor.get_buffer()
    
    def enrich_flow_features(self, feat_slice: torch.Tensor):
            self.__feat_tensor.add(feat_slice)