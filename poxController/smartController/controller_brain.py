from smartController.neural_modules import BinaryFlowClassifier, TwoStreamBinaryFlowClassifier
from smartController.replay_buffer import ReplayBuffer
import os
import torch
import torch.optim as optim
import torch.nn as nn
from smartController.wandb_tracker import WandBTracker

"""
######## PORT STATS: ###################
00: 'collisions'
01: 'port_no'
02: 'rx_bytes'
03: 'rx_crc_err'
04: 'rx_dropped'
05: 'rx_errors'
06: 'rx_frame_err'
07: 'rx_over_err'
08: 'rx_packets'
09: 'tx_bytes'
10: 'tx_dropped'
11: 'tx_errors'
12: 'tx_packets'
#########################################


#############FLOW FEATURES###############
'byte_count', 
'duration_nsec' / 10e9,
'duration_sec',
'packet_count'
#########################################
"""

SAVING_MODULES_FREQ = 50


PRETRAINED_MODEL_PATH = 'models/BinaryFlowClassifier.pt'

MAX_FLOW_TIMESTEPS=10

REPLAY_BUFFER_MAX_CAPACITY=1000

REPLAY_BUFFER_BATCH_SIZE=50

LEARNING_RATE=1e-3

DEVICE='cpu'

# Constants for wandb monitoring:
INFERENCE = 'Inference'
TRAINING = 'Training'
ACC = 'Acc'
LOSS = 'Loss'
STEP_LABEL = 'step'


class ControllerBrain():

    def __init__(self,
                 use_packet_feats,
                 flow_feat_dim,
                 packet_feat_dim,
                 logger_instance,
                 seed=777,
                 debug=False,
                 wb_track=False,
                 wb_project_name='',
                 wb_run_name='',
                 **wb_config_dict):
                
        self.use_packet_feats = use_packet_feats
        self.flow_feat_dim = flow_feat_dim
        self.packet_feat_dim = packet_feat_dim
        self.logger_instance = logger_instance
        self.AI_DEBUG = debug
        self.MAX_FLOW_TIMESTEPS=MAX_FLOW_TIMESTEPS

        self.replay_buffer = ReplayBuffer(
            capacity=REPLAY_BUFFER_MAX_CAPACITY,
            batch_size=REPLAY_BUFFER_BATCH_SIZE,
            seed=seed)
        
        self.batch_training_counts = 0
        self.best_accuracy = 0
        self.inference_counter = 0
        self.wbt = wb_track

        if self.wbt:

            wb_config_dict['SAVING_MODULES_FREQ'] = SAVING_MODULES_FREQ
            wb_config_dict['PRETRAINED_MODEL_PATH'] = PRETRAINED_MODEL_PATH
            wb_config_dict['MAX_FLOW_TIMESTEPS'] = MAX_FLOW_TIMESTEPS
            wb_config_dict['REPLAY_BUFFER_MAX_CAPACITY'] = REPLAY_BUFFER_MAX_CAPACITY
            wb_config_dict['REPLAY_BUFFER_BATCH_SIZE'] = REPLAY_BUFFER_BATCH_SIZE
            wb_config_dict['LEARNING_RATE'] = LEARNING_RATE
            wb_config_dict['DEVICE'] = DEVICE

            
            self.wbl = WandBTracker(
                wanb_project_name=wb_project_name,
                run_name=wb_run_name,
                config_dict=wb_config_dict).wb_logger

        self.initialize_binary_classifier(DEVICE, LEARNING_RATE, seed)


    def initialize_binary_classifier(self, device, lr, seed):
        
        torch.manual_seed(seed)
        if self.use_packet_feats:
            self.flow_classifier = TwoStreamBinaryFlowClassifier(
                flow_input_size=self.flow_feat_dim, 
                packet_input_size=self.packet_feat_dim,
                hidden_size=40,
                dropout_prob=0.1)
        else:
            self.flow_classifier = BinaryFlowClassifier(
                input_size=self.flow_feat_dim, 
                hidden_size=40,
                dropout_prob=0.1)
        self.check_pretrained()
        self.flow_classifier.to(device)
        self.fc_optimizer = optim.Adam(
            self.flow_classifier.parameters(), 
            lr=lr)
        self.binary_criterion = nn.BCELoss().to(device)


    def check_pretrained(self):
        # Check if the file exists
        if os.path.exists(PRETRAINED_MODEL_PATH):
            # Load the pre-trained weights
            self.flow_classifier.load_state_dict(torch.load(PRETRAINED_MODEL_PATH))
            if self.AI_DEBUG:
                self.logger_instance.info(f"Pre-trained weights loaded successfully from {PRETRAINED_MODEL_PATH}.")
        elif self.AI_DEBUG:
            self.logger_instance.info(f"Pre-trained weights not found at {PRETRAINED_MODEL_PATH}.")


    def classify_duet(self, flows):
        """
        makes inferences about a duet flow (source ip, dest ip)
        """
        if len(flows) == 0:
            return None
        else:
            self.inference_counter += 1
            flow_input_batch, packet_input_batch = self.assembly_input_tensor(flows)
            labels = self.get_labels(flows)
            self.replay_buffer.push(flow_input_batch, packet_input_batch, labels)

            if self.use_packet_feats:
                predictions = self.flow_classifier(flow_input_batch, packet_input_batch)
            else:
                predictions = self.flow_classifier(flow_input_batch)

            accuracy = self.learn_binary_classification(labels, predictions, INFERENCE)            
            if self.AI_DEBUG: 
                self.logger_instance.info(f'inference accuracy: {accuracy}')

            accuracy = self.experience_learning()
            return accuracy
    
    
    def experience_learning(self):
        flow_batch, packet_batch, batch_labels = self.replay_buffer.sample()
        if self.use_packet_feats:
            more_predictions = self.flow_classifier(flow_batch, packet_batch)
        else:
            more_predictions = self.flow_classifier(flow_batch)

        accuracy = self.learn_binary_classification(batch_labels, more_predictions, TRAINING)
        self.batch_training_counts += 1
        self.check_progress(curr_acc=accuracy)
        if self.AI_DEBUG: 
            self.logger_instance.info(f'batch labels mean: {batch_labels.mean().item()} batch prediction mean: {more_predictions.mean().item()}')
            self.logger_instance.info(f'mean training accuracy: {accuracy}')

    def check_progress(self, curr_acc):
        if (self.batch_training_counts > 0) and\
              (self.batch_training_counts % SAVING_MODULES_FREQ == 0) and\
                  (self.best_accuracy < curr_acc):
            self.best_accuracy = curr_acc
            self.save_models()

    def save_models(self):
        torch.save(
            self.flow_classifier.state_dict(), 
            'BinaryFlowClassifier.pt')
        if self.AI_DEBUG: 
            self.logger_instance.info(f'New model version saved')

    def learn_binary_classification(self, labels, predictions, mode):
        loss = self.binary_criterion(input=predictions,
                                      target=labels)
        # backward pass
        self.fc_optimizer.zero_grad()
        loss.backward()
        # update weights
        self.fc_optimizer.step()
        # print progress
        acc = (predictions.round() == labels).float().mean()
        if self.wbt:
            self.wbl.log({mode+'_'+ACC: acc.item(), STEP_LABEL:self.inference_counter})
            self.wbl.log({mode+'_'+LOSS: loss.item(), STEP_LABEL:self.inference_counter})

        return acc
    

    def get_labels(self, flows):
        labels = torch.Tensor([flows[0].infected]).unsqueeze(0).to(torch.float32)
        for flow in flows[1:]:
            labels = torch.cat([
                labels,
                torch.Tensor([flow.infected]).unsqueeze(0).to(torch.float32)
            ])
        return labels
    

    def assembly_input_tensor(
            self,
            flows):
        """
        A batch is composed of a set of flows. 
        Each Flow has a bidimensional feature tensor. 
        (self.MAX_FLOW_TIMESTEPS x 4 features)
        """
        flow_input_batch = flows[0].get_feat_tensor().unsqueeze(0)
        packet_input_batch = None

        if self.use_packet_feats:
            packet_input_batch = flows[0].packets_tensor.buffer.unsqueeze(0)

        for flow in flows[1:]:
            flow_input_batch = torch.cat( 
                [flow_input_batch,
                 flow.get_feat_tensor().unsqueeze(0)],
                 dim=0)
            if self.use_packet_feats:
                packet_input_batch = torch.cat( 
                    [packet_input_batch,
                    flow.packets_tensor.buffer.unsqueeze(0)],
                    dim=0)
                
        return flow_input_batch, packet_input_batch