import torch

class CircularBuffer:
    def __init__(self, buffer_size=10, feature_size=4):
        self.buffer_size = buffer_size
        self.feature_size = feature_size
        self.buffer = torch.zeros(buffer_size, feature_size)

    def add(self, new_tensor):
        # Roll the buffer up by 1 along the first dimension
        self.buffer = torch.roll(self.buffer, shifts=-1, dims=0)
        # Add new tensor to the last row
        self.buffer[-1] = new_tensor

    def get_buffer(self):
        return self.buffer
    

class Flow():
    def __init__(
        self, 
        source_ip, 
        dest_ip, 
        switch_output_port,
        FLOW_FEAT_DIM=4,
        MAX_FLOW_TIMESTEPS=10):
        
        self.source_ip = source_ip
        self.dest_ip = dest_ip
        self.switch_output_port = switch_output_port
        self.flow_id = self.source_ip + "_" + self.dest_ip + "_" + str(self.switch_output_port)
        self.switch_input_port = None
        self.__feat_tensor = CircularBuffer(
                            buffer_size=MAX_FLOW_TIMESTEPS, 
                            feature_size=FLOW_FEAT_DIM)
        
        self.infected = False

    def get_feat_tensor(self):
        return self.__feat_tensor.get_buffer()
    
    def enrich_features(self, feat_slice: torch.Tensor):
            self.__feat_tensor.add(feat_slice)
