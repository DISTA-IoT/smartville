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
from smartController.neural_modules import  MultiClassFlowClassifier, ThreeStreamMulticlassFlowClassifier, \
        TwoStreamMulticlassFlowClassifier, KernelRegressionLoss, ConfidenceDecoder
from smartController.replay_buffer import ReplayBuffer
import os
import torch
import torch.optim as optim
import torch.nn as nn
from smartController.wandb_tracker import WandBTracker
from smartController.flow import CircularBuffer
from pox.core import core  # only for logging.
import seaborn as sns
import matplotlib.pyplot as plt
import threading
from wandb import Image as wandbImage
import itertools
from sklearn.decomposition import PCA
import random
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# List of colors
colors = [
    'red', 'blue', 'green', 'purple', 'orange', 'pink', 'cyan',  'brown', 'yellow',
    'olive', 'lime', 'teal', 'maroon', 'navy', 'fuchsia', 'aqua', 'silver', 'sienna', 'gold',
    'indigo', 'violet', 'turquoise', 'tomato', 'orchid', 'slategray', 'peru', 'magenta', 'limegreen',
    'royalblue', 'coral', 'darkorange', 'darkviolet', 'darkslateblue', 'dodgerblue', 'firebrick',
    'lightseagreen', 'mediumorchid', 'orangered', 'powderblue', 'seagreen', 'springgreen', 'tan', 'wheat',
    'burlywood', 'chartreuse', 'crimson', 'darkgoldenrod', 'darkolivegreen', 'darkseagreen', 'indianred',
    'lavender', 'lightcoral', 'lightpink', 'lightsalmon', 'limegreen', 'mediumseagreen', 'mediumpurple',
    'midnightblue', 'palegreen', 'rosybrown', 'saddlebrown', 'salmon', 'slateblue', 'steelblue',
]

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

RAM = 'RAM'
CPU = 'CPU'
IN_TRAFFIC = 'IN_TRAFFIC'
OUT_TRAFFIC = 'OUT_TRAFFIC'
DELAY = 'DELAY'



PRETRAINED_MODELS_DIR = '/pox/pox/smartController/models/'

REPLAY_BUFFER_MAX_CAPACITY=1000

LEARNING_RATE=1e-3

REPORT_STEP_FREQUENCY = 50

REPULSIVE_WEIGHT = 1

ATTRACTIVE_WEIGHT = 1

KERNEL_REGRESSOR_HEADS = 2

EVALUATION_ROUNDS = 50

REGULARIZATION = True

# Constants for wandb monitoring:
INFERENCE = 'Inference'
TRAINING = 'Training'
CS_ACC = 'Acc'
CS_LOSS = 'Loss'
OS_ACC = 'AD Acc'
OS_LOSS = 'AD Loss'
KR_LOSS = 'KR_LOSS'
KR_ARI = 'KR_ARI'
KR_NMI = 'KR_NMI'
STEP_LABEL = 'step'
ANOMALY_BALANCE = 'ANOMALY_BALANCE'
CLOSED_SET = 'CS'
ANOMALY_DETECTION = 'AD'

# Create a lock object
lock = threading.Lock()


def efficient_cm(preds, targets_onehot):

    predictions_decimal = preds.argmax(dim=1).to(torch.int64)
    predictions_onehot = torch.zeros_like(
        preds,
        device=preds.device)
    predictions_onehot.scatter_(1, predictions_decimal.view(-1, 1), 1)

    return targets_onehot.T @ predictions_onehot


def efficient_os_cm(preds, targets_onehot):

    predictions_onehot = torch.zeros(
        [preds.size(0), 2],
        device=preds.device)
    predictions_onehot.scatter_(1, preds.view(-1, 1), 1)

    return targets_onehot.T @ predictions_onehot.long()


def get_balanced_accuracy(os_cm, negative_weight):
        
    N = os_cm[0][0] + os_cm[0][1]
    TN = os_cm[0][0]
    TNR = TN / (N + 1e-10)
        

    P = os_cm[1][1] + os_cm[1][0]
    TP = os_cm[1][1]
    TPR = TP / (P + 1e-10)
    
    return (negative_weight * TNR) + ((1-negative_weight) * TPR)


def get_clusters(predicted_kernel):
        
        discrete_predicted_kernel = (predicted_kernel > 0.5).long()
        
        assigned_mask = torch.zeros_like(discrete_predicted_kernel.diag())
        clusters = torch.zeros_like(discrete_predicted_kernel.diag())
        curr_cluster = 1

        for idx in range(discrete_predicted_kernel.shape[0]):
            if assigned_mask[idx] > 0:
                continue
            new_cluster_mask = discrete_predicted_kernel[idx]
            new_cluster_mask = torch.relu(new_cluster_mask - assigned_mask)
            assigned_mask += new_cluster_mask
            clusters += new_cluster_mask*curr_cluster
            if new_cluster_mask.sum() > 0:
                curr_cluster += 1

        return clusters -1 
    

class DynamicLabelEncoder:

    def __init__(self):
        self.label_to_int = {}
        self.int_to_label = {}
        self.current_code = 0


    def fit(self, labels):
        """
        returns the number of new classes found!
        """
        new_labels = set(labels) - set(self.label_to_int.keys())

        for label in new_labels:
            self.label_to_int[label] = self.current_code
            self.int_to_label[self.current_code] = label
            self.current_code += 1

        return new_labels


    def transform(self, labels):
        encoded_labels = [self.label_to_int[label] for label in labels]
        return torch.tensor(encoded_labels)


    def inverse_transform(self, encoded_labels):
        decoded_labels = [self.int_to_label[code.item()] for code in encoded_labels]
        return decoded_labels


    def get_mapping(self):
        return self.label_to_int


    def get_labels(self):
        return list(self.label_to_int.keys())


class ControllerBrain():

    def __init__(self,
                 eval,
                 use_packet_feats,
                 use_node_feats,
                 flow_feat_dim,
                 packet_feat_dim,
                 h_dim,
                 dropout,
                 multi_class,
                 k_shot,
                 replay_buffer_batch_size,
                 kernel_regression,
                 device='cpu',
                 seed=777,
                 debug=False,
                 wb_track=False,
                 wb_project_name='',
                 wb_run_name='',
                 **wb_config_dict):
        
        self.eval = eval
        self.use_packet_feats = use_packet_feats
        self.use_node_feats = use_node_feats
        self.flow_feat_dim = flow_feat_dim
        self.packet_feat_dim = packet_feat_dim
        self.h_dim = h_dim
        self.dropout = dropout
        self.multi_class = multi_class
        self.AI_DEBUG = debug
        self.best_cs_accuracy = 0
        self.best_AD_accuracy = 0
        self.best_KR_accuracy = 0
        self.backprop_counter = 0
        self.wbt = wb_track
        self.wbl = None
        self.kernel_regression = kernel_regression
        self.logger_instance = core.getLogger()
        self.device=device
        self.seed = seed
        random.seed(seed)
        self.current_known_classes_count = 0
        self.current_training_known_classes_count = 0
        self.current_test_known_classes_count = 0
        self.reset_train_cms()
        self.reset_test_cms()
        self.k_shot = k_shot
        self.replay_buff_batch_size = replay_buffer_batch_size
        self.encoder = DynamicLabelEncoder()
        self.replay_buffers = {}
        self.test_replay_buffers = {}
        self.init_neural_modules(LEARNING_RATE, seed)
        
        if self.wbt:

            wb_config_dict['PRETRAINED_MODEL_PATH'] = PRETRAINED_MODELS_DIR
            wb_config_dict['REPLAY_BUFFER_MAX_CAPACITY'] = REPLAY_BUFFER_MAX_CAPACITY
            wb_config_dict['LEARNING_RATE'] = LEARNING_RATE
            wb_config_dict['REPORT_STEP_FREQUENCY'] = REPORT_STEP_FREQUENCY
            wb_config_dict['KERNEL_REGRESSOR_HEADS'] = KERNEL_REGRESSOR_HEADS
            wb_config_dict['REPULSIVE_WEIGHT']  = REPULSIVE_WEIGHT
            wb_config_dict['ATTRACTIVE_WEIGHT']  = ATTRACTIVE_WEIGHT
            wb_config_dict['REGULARIZATION']  = REGULARIZATION

            self.wbl = WandBTracker(
                wanb_project_name=wb_project_name,
                run_name=wb_run_name,
                config_dict=wb_config_dict).wb_logger        


    def add_replay_buffer(self, class_name):
        self.inference_allowed = False
        self.experience_learning_allowed = False
        self.eval_allowed = False
        if self.AI_DEBUG:
            self.logger_instance.info(f'Adding a replay buffer with code {self.current_known_classes_count-1}')
            self.logger_instance.info(f'Encoder state mapping: {self.encoder.get_mapping()}')
        
        if not 'G2' in class_name:
            self.replay_buffers[self.current_known_classes_count-1] = ReplayBuffer(
                capacity=REPLAY_BUFFER_MAX_CAPACITY,
                batch_size=self.replay_buff_batch_size,
                seed=self.seed)
        if not 'G1' in class_name:
            self.test_replay_buffers[self.current_known_classes_count-1] = ReplayBuffer(
                        capacity=REPLAY_BUFFER_MAX_CAPACITY,
                        batch_size=self.replay_buff_batch_size,
                        seed=self.seed)


    def add_class_to_knowledge_base(self, new_class):
        if self.AI_DEBUG:
            self.logger_instance.info(f'New class found: {new_class}')
        self.current_known_classes_count += 1
        if not 'G2' in new_class:
            self.current_training_known_classes_count += 1 
        if not 'G1' in new_class:
            self.current_test_known_classes_count += 1
        self.add_replay_buffer(new_class)
        self.reset_train_cms()
        self.reset_test_cms()


    def reset_train_cms(self):
        self.training_cs_cm = torch.zeros(
            [self.current_known_classes_count, self.current_known_classes_count],
            device=self.device)
        self.training_os_cm = torch.zeros(
            size=(2, 2),
            device=self.device)
        
    
    def reset_test_cms(self):
        self.eval_cs_cm = torch.zeros(
            [self.current_known_classes_count, self.current_known_classes_count],
            device=self.device)
        self.eval_os_cm = torch.zeros(
            size=(2, 2),
            device=self.device)
        

    def init_neural_modules(self, lr, seed):
        torch.manual_seed(seed)
        self.confidence_decoder = ConfidenceDecoder(device=self.device)
        self.os_criterion = nn.BCEWithLogitsLoss().to(self.device)
        self.cs_criterion = nn.CrossEntropyLoss().to(self.device)
        self.kr_criterion = KernelRegressionLoss(repulsive_weigth=REPULSIVE_WEIGHT, 
            attractive_weigth=ATTRACTIVE_WEIGHT).to(self.device)
        
        if self.use_packet_feats:
            
            if self.use_node_feats:

                self.classifier = ThreeStreamMulticlassFlowClassifier(
                    flow_input_size=self.flow_feat_dim, 
                    second_stream_input_size=self.packet_feat_dim,
                    third_stream_input_size=5,
                    hidden_size=self.h_dim,
                    kr_heads=KERNEL_REGRESSOR_HEADS,
                    dropout_prob=self.dropout,
                    device=self.device)
            
            else: 

                self.classifier = TwoStreamMulticlassFlowClassifier(
                    flow_input_size=self.flow_feat_dim, 
                    second_stream_input_size=self.packet_feat_dim,
                    hidden_size=self.h_dim,
                    kr_heads=KERNEL_REGRESSOR_HEADS,
                    dropout_prob=self.dropout,
                    device=self.device)

        else:
            
            if self.use_node_feats:
                
                self.classifier = TwoStreamMulticlassFlowClassifier(
                    flow_input_size=self.flow_feat_dim, 
                    second_stream_input_size=5,
                    hidden_size=self.h_dim,
                    kr_heads=KERNEL_REGRESSOR_HEADS,
                    dropout_prob=self.dropout,
                    device=self.device)
            else:

                self.classifier = MultiClassFlowClassifier(
                    input_size=self.flow_feat_dim, 
                    hidden_size=self.h_dim,
                    dropout_prob=self.dropout,
                    kr_heads=KERNEL_REGRESSOR_HEADS,
                    device=self.device)
            


        self.check_pretrained()

        params_for_optimizer = \
            list(self.confidence_decoder.parameters()) + \
                list(self.classifier.parameters())

        self.classifier.to(self.device)
        self.cs_optimizer = optim.Adam(
            params_for_optimizer, 
            lr=lr)

        if self.eval:
            self.classifier.eval()
            self.confidence_decoder.eval()
            self.logger_instance.info(f"Using MODULES in EVAL mode!")                


    def check_pretrained(self):

        if self.use_packet_feats:
            if self.use_node_feats:
                self.classifier_path = PRETRAINED_MODELS_DIR+'multiclass_flow_packet_node_classifier_pretrained'
                self.confidence_decoder_path = PRETRAINED_MODELS_DIR+'flow_packet_node_confidence_decoder_pretrained'
            else:
                self.classifier_path = PRETRAINED_MODELS_DIR+'multiclass_flow_packet_classifier_pretrained'
                self.confidence_decoder_path = PRETRAINED_MODELS_DIR+'flow_packet_confidence_decoder_pretrained'
        else:
            if self.use_node_feats:
                self.classifier_path = PRETRAINED_MODELS_DIR+'multiclass_flow_node_classifier_pretrained'
                self.confidence_decoder_path = PRETRAINED_MODELS_DIR+'flow_node_confidence_decoder_pretrained'
            else:    
                self.classifier_path = PRETRAINED_MODELS_DIR+'multiclass_flow_classifier_pretrained'
                self.confidence_decoder_path = PRETRAINED_MODELS_DIR+'flow_confidence_decoder_pretrained'

        # Check if the file exists
        if os.path.exists(PRETRAINED_MODELS_DIR):

            if os.path.exists(self.classifier_path+'.pt'):
                # Load the pre-trained weights
                self.classifier.load_state_dict(torch.load(self.classifier_path+'.pt'))
                self.logger_instance.info(f"Pre-trained weights loaded successfully from {self.classifier_path}.pt")
            else:
                self.logger_instance.info(f"Pre-trained weights not found at {self.classifier_path}.pt")
                
            if self.multi_class:
                if os.path.exists(self.confidence_decoder_path+'.pt'):
                    self.confidence_decoder.load_state_dict(torch.load(self.confidence_decoder_path+'.pt'))
                    self.logger_instance.info(f"Pre-trained weights loaded successfully from {self.confidence_decoder_path}.pt")
                else:
                    self.logger_instance.info(f"Pre-trained weights not found at {self.confidence_decoder_path}.pt")             
             

        elif self.AI_DEBUG:
            self.logger_instance.info(f"Pre-trained folder not found at {PRETRAINED_MODELS_DIR}.")


    def infer(
            self,
            flow_input_batch,
            packet_input_batch,
            node_feat_input_batch,
            batch_labels,
            query_mask):

        if self.use_packet_feats:
            if self.use_node_feats:
                logits, hiddens, predicted_kernel = self.classifier(
                    flow_input_batch, 
                    packet_input_batch, 
                    node_feat_input_batch,
                    batch_labels, 
                    self.current_known_classes_count,
                    query_mask)
            else:
                logits, hiddens, predicted_kernel = self.classifier(
                    flow_input_batch, 
                    packet_input_batch, 
                    batch_labels, 
                    self.current_known_classes_count,
                    query_mask)
        else:
            if self.use_node_feats:
                logits, hiddens, predicted_kernel = self.classifier(
                    flow_input_batch, 
                    node_feat_input_batch, 
                    batch_labels, 
                    self.current_known_classes_count,
                    query_mask)
            else:
                logits, hiddens, predicted_kernel = self.classifier(
                    flow_input_batch, 
                    batch_labels, 
                    self.current_known_classes_count,
                    query_mask)


        return logits, hiddens, predicted_kernel


    def push_to_replay_buffers(
            self,
            flow_input_batch, 
            packet_input_batch,
            node_feat_input_batch,
            batch_labels,
            zda_batch_labels,
            test_zda_batch_labels):
        """
        Don't know why, but you can have more than one sample
        per class in inference time. 
        (More than one flowstats object for a single Flow!)
        So we need to take care of carefully populating our buffers...
        Otherwise we will have bad surprises when sampling from them!!!
        (i.e. sampling more elements than those requested!)
        """
        unique_labels = torch.unique(batch_labels)

        for label in unique_labels:
            mask = batch_labels == label

            if self.use_packet_feats:
                if self.use_node_feats:
                    for sample_idx in range(flow_input_batch[mask].shape[0]):
                        self.replay_buffers[label.item()].push(
                            flow_input_batch[mask][sample_idx].unsqueeze(0), 
                            packet_input_batch[mask][sample_idx].unsqueeze(0),
                            node_feat_input_batch[mask][sample_idx].unsqueeze(0),
                            label=batch_labels[mask][sample_idx].unsqueeze(0),
                            zda_label=zda_batch_labels[mask][sample_idx].unsqueeze(0),
                            test_zda_label=test_zda_batch_labels[mask][sample_idx].unsqueeze(0))
                else:
                    for sample_idx in range(flow_input_batch[mask].shape[0]):
                        self.replay_buffers[label.item()].push(
                            flow_input_batch[mask][sample_idx].unsqueeze(0), 
                            packet_input_batch[mask][sample_idx].unsqueeze(0),
                            None,
                            label=batch_labels[mask][sample_idx].unsqueeze(0),
                            zda_label=zda_batch_labels[mask][sample_idx].unsqueeze(0),
                            test_zda_label=test_zda_batch_labels[mask][sample_idx].unsqueeze(0))
            else:
                if self.use_node_feats:
                    for sample_idx in range(flow_input_batch[mask].shape[0]):
                        self.replay_buffers[label.item()].push(
                            flow_input_batch[mask][sample_idx].unsqueeze(0), 
                            None,
                            node_feat_input_batch[mask][sample_idx].unsqueeze(0),
                            label=batch_labels[mask][sample_idx].unsqueeze(0),
                            zda_label=zda_batch_labels[mask][sample_idx].unsqueeze(0),
                            test_zda_label=test_zda_batch_labels[mask][sample_idx].unsqueeze(0))                   
                else:
                    for sample_idx in range(flow_input_batch[mask].shape[0]):
                        self.replay_buffers[label.item()].push(
                            flow_input_batch[mask][sample_idx].unsqueeze(0), 
                            None,
                            None, 
                            label=batch_labels[mask][sample_idx].unsqueeze(0),
                            zda_label=zda_batch_labels[mask][sample_idx].unsqueeze(0),
                            test_zda_label=test_zda_batch_labels[mask][sample_idx].unsqueeze(0))
        
        if not self.inference_allowed or not self.experience_learning_allowed or not self.eval_allowed:
            buff_lengths = [len(replay_buff) for replay_buff in self.replay_buffers.values()]
            test_buff_lengths = [len(replay_buff) for replay_buff in self.test_replay_buffers.values()]

            if self.AI_DEBUG:
                self.logger_instance.info(f'Buffer lengths: {buff_lengths}')
                self.logger_instance.info(f'Test Buffer lengths: {test_buff_lengths}')

            self.inference_allowed = torch.all(
                torch.Tensor([buff_len  > self.k_shot for buff_len in buff_lengths]))
            self.experience_learning_allowed = torch.all(
                torch.Tensor([buff_len  > self.replay_buff_batch_size for buff_len in buff_lengths]))
            self.eval_allowed = torch.all(
                torch.Tensor([buff_len  > self.replay_buff_batch_size for buff_len in test_buff_lengths]))


    def push_to_test_replay_buffers(
            self,
            flow_input_batch, 
            packet_input_batch,
            node_feat_input_batch, 
            batch_labels,
            zda_batch_labels,
            test_zda_batch_labels):

        unique_labels = torch.unique(batch_labels)

        for label in unique_labels:
            mask = batch_labels == label

            if self.use_packet_feats:
                if self.use_node_feats:
                    for sample_idx in range(flow_input_batch[mask].shape[0]):
                        self.test_replay_buffers[label.item()].push(
                            flow_input_batch[mask][sample_idx].unsqueeze(0), 
                            packet_input_batch[mask][sample_idx].unsqueeze(0),
                            node_feat_input_batch[mask][sample_idx].unsqueeze(0),
                            label=batch_labels[mask][sample_idx].unsqueeze(0),
                            zda_label=zda_batch_labels[mask][sample_idx].unsqueeze(0),
                            test_zda_label=test_zda_batch_labels[mask][sample_idx].unsqueeze(0))                 
                else:
                    for sample_idx in range(flow_input_batch[mask].shape[0]):
                        self.test_replay_buffers[label.item()].push(
                            flow_input_batch[mask][sample_idx].unsqueeze(0), 
                            packet_input_batch[mask][sample_idx].unsqueeze(0),
                            None,
                            label=batch_labels[mask][sample_idx].unsqueeze(0),
                            zda_label=zda_batch_labels[mask][sample_idx].unsqueeze(0),
                            test_zda_label=test_zda_batch_labels[mask][sample_idx].unsqueeze(0))
            else:
                if self.use_node_feats:
                    for sample_idx in range(flow_input_batch[mask].shape[0]):
                        self.test_replay_buffers[label.item()].push(
                            flow_input_batch[mask][sample_idx].unsqueeze(0), 
                            None,
                            node_feat_input_batch[mask][sample_idx].unsqueeze(0),
                            label=batch_labels[mask][sample_idx].unsqueeze(0),
                            zda_label=zda_batch_labels[mask][sample_idx].unsqueeze(0),
                            test_zda_label=test_zda_batch_labels[mask][sample_idx].unsqueeze(0))
                else:
                    for sample_idx in range(flow_input_batch[mask].shape[0]):
                        self.test_replay_buffers[label.item()].push(
                            flow_input_batch[mask][sample_idx].unsqueeze(0), 
                            None,
                            None,
                            label=batch_labels[mask][sample_idx].unsqueeze(0),
                            zda_label=zda_batch_labels[mask][sample_idx].unsqueeze(0),
                            test_zda_label=test_zda_batch_labels[mask][sample_idx].unsqueeze(0))
            

    def classify_duet(self, flows, node_feats: dict = None):
        """
        makes inferences about a duet flow (source ip, dest ip)
        """
        node_feat_arg = None
        packet_feat_arg = None
        
        with lock:

            if len(flows) == 0:
                return None
            else:
                flow_input_batch, packet_input_batch, node_feat_input_batch = self.assembly_input_tensor(
                    flows,
                    node_feats)
                batch_labels, zda_labels, test_zda_labels = self.get_labels(flows)

                if random.random() > 0.3:
                    to_push_mask = ~test_zda_labels.bool()

                    if self.use_node_feats:
                        node_feat_arg = node_feat_input_batch[to_push_mask]
                    if self.use_packet_feats:
                        packet_feat_arg = packet_input_batch[to_push_mask] 

                    self.push_to_replay_buffers(
                        flow_input_batch[to_push_mask], 
                        packet_feat_arg,
                        node_feat_arg,  
                        batch_labels=batch_labels[to_push_mask],
                        zda_batch_labels=zda_labels[to_push_mask],
                        test_zda_batch_labels=test_zda_labels[to_push_mask])
                else:
                    known_mask = ~zda_labels.bool()
                    to_push_mask = torch.logical_or(test_zda_labels.bool(), known_mask)

                    if self.use_node_feats:
                        node_feat_arg = node_feat_input_batch[to_push_mask]
                    if self.use_packet_feats:
                        packet_feat_arg = packet_input_batch[to_push_mask] 

                    self.push_to_test_replay_buffers(
                        flow_input_batch[to_push_mask], 
                        packet_feat_arg, 
                        node_feat_arg,  
                        batch_labels=batch_labels[to_push_mask],
                        zda_batch_labels=zda_labels[to_push_mask],
                        test_zda_batch_labels=test_zda_labels[to_push_mask])

                """
                if self.inference_allowed:

                    support_flow_batch, \
                        support_packet_batch, \
                            support_labels, \
                                support_zda_labels, \
                                    support_test_zda_labels = self.sample_from_replay_buffers(
                        samples_per_class=self.k_shot)
                    
                    query_mask = torch.zeros(
                        size=(support_labels.shape[0],), 
                        device=self.device).to(torch.bool)

                    query_mask = torch.cat([query_mask, torch.ones_like(batch_labels).to(torch.bool)])
                    flow_input_batch = torch.vstack([support_flow_batch, flow_input_batch])
                    if self.use_packet_feats:
                        packet_input_batch = torch.vstack([support_packet_batch, packet_input_batch])
                    batch_labels = torch.cat([support_labels.squeeze(1), batch_labels]).unsqueeze(1)

                    predictions, _, _ = self.infer(
                        flow_input_batch=flow_input_batch,
                        packet_input_batch=packet_input_batch,
                        batch_labels=batch_labels,
                        query_mask=query_mask
                        )

                    accuracy = self.learning_step(batch_labels, predictions, INFERENCE, query_mask)

                    

                    if self.AI_DEBUG: 
                        self.logger_instance.info(f'inference accuracy: {accuracy}')
                """

                if self.experience_learning_allowed:
                    self.experience_learning()
                    self.backprop_counter += 1

        
    def sample_from_replay_buffers(self, samples_per_class, mode):
        balanced_packet_batch = None
        balanced_node_feat_batch = None

        init = True
        if mode == TRAINING:
            buffers_dict = self.replay_buffers
        elif mode == INFERENCE:
            buffers_dict = self.test_replay_buffers

        for replay_buff in buffers_dict.values():
            flow_batch, \
                packet_batch, \
                    node_feat_batch, \
                        batch_labels, \
                            zda_batch_labels, \
                                test_zda_batch_labels = replay_buff.sample(samples_per_class)
            
            if init:
                balanced_flow_batch = flow_batch
                balanced_labels = batch_labels
                balanced_zda_labels = zda_batch_labels
                balanced_test_zda_labels = test_zda_batch_labels
                if packet_batch is not None:
                    balanced_packet_batch = packet_batch
                if node_feat_batch is not None:
                    balanced_node_feat_batch = node_feat_batch

            else: 
                balanced_flow_batch = torch.vstack(
                    [balanced_flow_batch, flow_batch])
                balanced_labels = torch.vstack(
                    [balanced_labels, batch_labels])
                balanced_zda_labels = torch.vstack(
                    [balanced_zda_labels, zda_batch_labels])
                balanced_test_zda_labels = torch.vstack(
                    [balanced_test_zda_labels, test_zda_batch_labels])
                if packet_batch is not None:
                    balanced_packet_batch = torch.vstack(
                        [balanced_packet_batch, packet_batch])
                if node_feat_batch is not None:
                    balanced_node_feat_batch = torch.vstack(
                       [balanced_node_feat_batch, node_feat_batch])

            init = False

        return balanced_flow_batch, balanced_packet_batch, balanced_node_feat_batch, balanced_labels, balanced_zda_labels, balanced_test_zda_labels


    def get_canonical_query_mask(self, phase):
        if phase == TRAINING:
            class_count = self.current_training_known_classes_count
        elif phase == INFERENCE:
            class_count = self.current_test_known_classes_count

        query_mask = torch.zeros(
            size=(class_count, self.replay_buff_batch_size),
            device=self.device).to(torch.bool)
        query_mask[:, self.k_shot:] = True
        return query_mask.view(-1)


    def get_oh_labels(self, curr_shape, targets):
        targets = targets.to(torch.int64)
        # Create a one-hot encoding of the targets.
        targets_onehot = torch.zeros(
            size=curr_shape,
            device=targets.device)
        targets_onehot.scatter_(1, targets.view(-1, 1), 1)
        return targets_onehot
    

    def AD_step(self, zda_labels, preds, query_mask, mode):
        

        zda_predictions = self.confidence_decoder(scores=preds)

        os_loss = self.os_criterion(
            input=zda_predictions,
            target=zda_labels[query_mask])
        
        onehot_zda_labels = torch.zeros(size=(zda_labels.shape[0],2)).long()
        onehot_zda_labels.scatter_(1, zda_labels.long().view(-1, 1), 1)

        if mode == TRAINING:
            self.training_os_cm += efficient_os_cm(
                preds=(zda_predictions.detach() > 0.5).long(),
                targets_onehot=onehot_zda_labels[query_mask].long()
                )
            os_acc = get_balanced_accuracy(self.training_os_cm, negative_weight=0.5)

        elif mode == INFERENCE:
            self.eval_os_cm += efficient_os_cm(
                preds=(zda_predictions.detach() > 0.5).long(),
                targets_onehot=onehot_zda_labels[query_mask].long()
                )
            os_acc = get_balanced_accuracy(self.eval_os_cm, negative_weight=0.5)

        zda_balance = zda_labels[query_mask].to(torch.float16).mean().item()
        if self.wbt:
            self.wbl.log({mode+'_'+OS_ACC: os_acc.item(), STEP_LABEL:self.backprop_counter})
            self.wbl.log({mode+'_'+OS_LOSS: os_loss.item(), STEP_LABEL:self.backprop_counter})
            self.wbl.log({mode+'_'+ANOMALY_BALANCE: zda_balance, STEP_LABEL:self.backprop_counter})

        if self.AI_DEBUG: 
            self.logger_instance.info(f'{mode} batch AD labels mean: {zda_balance} '+\
                                      f'{mode} batch AD prediction mean: {zda_predictions.to(torch.float32).mean()}')
            self.logger_instance.info(f'{mode} mean AD training accuracy: {os_acc}')

        return os_loss, os_acc
    

    def kernel_regression_step(self, predicted_kernel, one_hot_labels, mode):

        if self.kernel_regression:
            
            semantic_kernel = one_hot_labels @ one_hot_labels.T

            kernel_loss = self.kr_criterion(
                baseline_kernel=semantic_kernel,
                predicted_kernel=predicted_kernel
            )

            decimal_sematic_kernel = one_hot_labels.max(1)[1].detach().numpy()
            decimal_predicted_kernel = get_clusters(predicted_kernel.detach())
            np_dec_pred_kernel = decimal_predicted_kernel.numpy()

            # Compute clustering metrics
            kr_ari = adjusted_rand_score(
                decimal_sematic_kernel, 
                np_dec_pred_kernel)
            kr_nmi = normalized_mutual_info_score(
                decimal_sematic_kernel,
                np_dec_pred_kernel)

            if self.wbt:
                self.wbl.log({mode+'_'+KR_ARI: kr_ari, STEP_LABEL:self.backprop_counter})
                self.wbl.log({mode+'_'+KR_NMI: kr_nmi, STEP_LABEL:self.backprop_counter})

                self.wbl.log({mode+'_'+KR_LOSS: kernel_loss.item(), STEP_LABEL:self.backprop_counter})

            if self.AI_DEBUG: 
                self.logger_instance.info(f'{mode} kernel regression ARI: {kr_ari} NMI:{kr_nmi}')
                self.logger_instance.info(f'{mode} kernel regression loss: {kernel_loss.item()}')

            return kernel_loss, decimal_predicted_kernel, kr_ari


    def experience_learning(self):

        balanced_flow_batch, \
            balanced_packet_batch, \
                balanced_node_feat_batch, \
                    balanced_labels, \
                        balanced_zda_labels, \
                            balanced_test_zda_labels = self.sample_from_replay_buffers(
                samples_per_class=self.replay_buff_batch_size,
                mode=TRAINING)
        
        query_mask = self.get_canonical_query_mask(TRAINING)

        assert query_mask.shape[0] == balanced_labels.shape[0]

        logits, hidden_vectors, predicted_kernel = self.infer(
            flow_input_batch=balanced_flow_batch,
            packet_input_batch=balanced_packet_batch,
            node_feat_input_batch=balanced_node_feat_batch,
            batch_labels=balanced_labels,
            query_mask=query_mask)
        
        loss = 0

        # one_hot_labels
        one_hot_labels = self.get_oh_labels(
            curr_shape=(balanced_labels.shape[0],logits.shape[1]), 
            targets=balanced_labels)
        
        # known class horizonal mask:
        known_oh_labels = one_hot_labels[~balanced_zda_labels.squeeze(1).bool()]
        known_class_h_mask = known_oh_labels.sum(0)>0

        kr_loss, predicted_clusters, _ = self.kernel_regression_step(
            predicted_kernel, 
            one_hot_labels, 
            TRAINING)
    
        if REGULARIZATION:
            loss += kr_loss
        
        if self.multi_class:
            ad_loss, _ = self.AD_step(
                zda_labels=balanced_zda_labels, 
                preds=logits[:, known_class_h_mask], 
                query_mask=query_mask,
                mode=TRAINING)
            loss += ad_loss

        self.training_cs_cm += efficient_cm(
        preds=logits.detach(),
        targets_onehot=one_hot_labels[query_mask])
        

        if self.backprop_counter % REPORT_STEP_FREQUENCY == 0:
            self.report(
                preds=logits[:,known_class_h_mask],  
                hiddens=hidden_vectors.detach(), 
                labels=balanced_labels,
                predicted_clusters=predicted_clusters, 
                query_mask=query_mask,
                phase=TRAINING)

            if self.eval_allowed:
                self.evaluate_models()

        cs_acc = self.learning_step(balanced_labels, logits, TRAINING, query_mask, loss)
            
        if self.AI_DEBUG: 
            self.logger_instance.info(f'{TRAINING} batch labels mean: {balanced_labels.to(torch.float16).mean().item()} '+\
                                      f'{TRAINING} batch prediction mean: {logits.max(1)[1].to(torch.float32).mean()}')
            self.logger_instance.info(f'{TRAINING} mean multiclass classif accuracy: {cs_acc}')


    def evaluate_models(self):

        self.classifier.eval()
        self.confidence_decoder.eval()
        
        mean_eval_ad_acc = 0
        mean_eval_cs_acc = 0
        mean_eval_kr_ari = 0

        for _ in range(EVALUATION_ROUNDS):
                
            balanced_flow_batch, \
                balanced_packet_batch, \
                    balanced_node_feat_batch, \
                        balanced_labels, \
                            balanced_zda_labels, \
                                balanced_test_zda_labels = self.sample_from_replay_buffers(
                                    samples_per_class=self.replay_buff_batch_size,
                                    mode=INFERENCE)
            
            query_mask = self.get_canonical_query_mask(INFERENCE)

            assert query_mask.shape[0] == balanced_labels.shape[0]

            logits, hidden_vectors, predicted_kernel = self.infer(
                flow_input_batch=balanced_flow_batch,
                packet_input_batch=balanced_packet_batch,
                node_feat_input_batch=balanced_node_feat_batch,
                batch_labels=balanced_labels,
                query_mask=query_mask)

            # one_hot_labels
            one_hot_labels = self.get_oh_labels(
                curr_shape=(balanced_labels.shape[0],logits.shape[1]), 
                targets=balanced_labels)
            
            # known class horizonal mask:
            known_oh_labels = one_hot_labels[~balanced_zda_labels.squeeze(1).bool()]
            known_class_h_mask = known_oh_labels.sum(0)>0

            _, predicted_clusters, kr_precision = self.kernel_regression_step(
            predicted_kernel, 
            one_hot_labels,
            INFERENCE)

            _, ad_acc = self.AD_step(
                zda_labels=balanced_zda_labels, 
                preds=logits[:, known_class_h_mask], 
                query_mask=query_mask,
                mode=INFERENCE)

            self.eval_cs_cm += efficient_cm(
            preds=logits.detach(),
            targets_onehot=one_hot_labels[query_mask])
            
            cs_acc = self.learning_step(balanced_labels, logits, INFERENCE, query_mask)

            mean_eval_ad_acc += (ad_acc / EVALUATION_ROUNDS)
            mean_eval_cs_acc += (cs_acc / EVALUATION_ROUNDS)
            mean_eval_kr_ari += (kr_precision / EVALUATION_ROUNDS)

        if self.AI_DEBUG: 
            self.logger_instance.info(f'{INFERENCE} mean eval AD accuracy: {mean_eval_ad_acc.item()} '+\
                                    f'{INFERENCE}  mean eval CS accuracy: {mean_eval_cs_acc.item()}')
            self.logger_instance.info(f'{INFERENCE} mean eval KR accuracy: {mean_eval_kr_ari}')
        if self.wbt:
            self.wbl.log({'Mean EVAL AD ACC': mean_eval_ad_acc.item(), STEP_LABEL:self.backprop_counter})
            self.wbl.log({'Mean EVAL CS ACC': mean_eval_cs_acc.item(), STEP_LABEL:self.backprop_counter})
            self.wbl.log({'Mean EVAL KR PREC': mean_eval_kr_ari, STEP_LABEL:self.backprop_counter})

        if not self.eval:
            self.check_kr_progress(curr_kr_acc=mean_eval_kr_ari)
            self.check_cs_progress(curr_cs_acc=mean_eval_cs_acc.item())
            self.check_AD_progress(curr_ad_acc=mean_eval_ad_acc.item())

        self.report(
                preds=logits[:,known_class_h_mask], 
                hiddens=hidden_vectors.detach(), 
                labels=balanced_labels,
                predicted_clusters=predicted_clusters, 
                query_mask=query_mask,
                phase=INFERENCE)

        self.classifier.train()
        self.confidence_decoder.train()


    def report(self, preds, hiddens, labels, predicted_clusters, query_mask, phase):

        if phase == TRAINING:
            cs_cm_to_plot = self.training_cs_cm
            os_cm_to_plot = self.training_os_cm
        elif phase == INFERENCE:
            cs_cm_to_plot = self.eval_cs_cm
            os_cm_to_plot = self.eval_os_cm

        if self.wbt:
            self.plot_confusion_matrix(
                mod=CLOSED_SET,
                cm=cs_cm_to_plot,
                phase=phase,
                norm=False,
                classes=self.encoder.get_labels())
            self.plot_confusion_matrix(
                mod=ANOMALY_DETECTION,
                cm=os_cm_to_plot,
                phase=phase,
                norm=False,
                classes=['Known', 'ZdA'])
            self.plot_hidden_space(hiddens=hiddens, labels=labels, predicted_labels=predicted_clusters, phase=phase)
            self.plot_scores_vectors(score_vectors=preds, labels=labels[query_mask], phase=phase)

        if self.AI_DEBUG:
            self.logger_instance.info(f'{phase} CS Conf matrix: \n {cs_cm_to_plot}')
            self.logger_instance.info(f'{phase} AD Conf matrix: \n {os_cm_to_plot}')
        
        if phase == TRAINING:
            self.reset_train_cms()
        elif phase == INFERENCE:
            self.reset_test_cms()


    def check_cs_progress(self, curr_cs_acc):
        self.best_cs_accuracy = curr_cs_acc
        self.save_cs_model()

    
    def check_AD_progress(self, curr_ad_acc):
        self.best_AD_accuracy = curr_ad_acc
        self.save_ad_model()

    
    def check_kr_progress(self, curr_kr_acc):
        self.best_KR_accuracy = curr_kr_acc
        self.save_models() 


    def save_cs_model(self, postfix='single'):
        torch.save(
            self.classifier.state_dict(), 
            self.classifier_path+postfix+'.pt')
        if self.AI_DEBUG: 
            self.logger_instance.info(f'New {postfix} flow classifier model version saved to {self.classifier_path}{postfix}.pt')


    def save_ad_model(self, postfix='single'):
        torch.save(
            self.confidence_decoder.state_dict(), 
            self.confidence_decoder_path+postfix+'.pt')
        if self.AI_DEBUG: 
            self.logger_instance.info(f'New {postfix} confidence decoder model version saved to {self.confidence_decoder_path}{postfix}.pt')


    def save_models(self):
        self.save_cs_model(postfix='coupled')
        if self.multi_class:
            self.save_ad_model(postfix='coupled')


    def learning_step(self, labels, predictions, mode, query_mask, prev_loss=0):
        
        cs_loss = self.cs_criterion(input=predictions,
                                    target=labels[query_mask].squeeze(1))

        if mode == TRAINING:
            loss = prev_loss + cs_loss
            # backward pass
            self.cs_optimizer.zero_grad()
            loss.backward()
            # update weights
            self.cs_optimizer.step()

        # compute accuracy
        acc = self.get_accuracy(logits_preds=predictions, decimal_labels=labels, query_mask=query_mask)

        # report progress
        if self.wbt:
            self.wbl.log({mode+'_'+CS_ACC: acc.item(), STEP_LABEL:self.backprop_counter})
            self.wbl.log({mode+'_'+CS_LOSS: cs_loss.item(), STEP_LABEL:self.backprop_counter})

        return acc
    

    def get_accuracy(self, logits_preds, decimal_labels, query_mask):
        """
        labels must not be one hot!
        """
        match_mask = logits_preds.max(1)[1] == decimal_labels.max(1)[0][query_mask]
        return match_mask.sum() / match_mask.shape[0]



    def get_labels(self, flows):

        string_labels = [flow.element_class for flow in flows]
        new_classes = self.encoder.fit(string_labels)
        for new_class in new_classes:
            self.add_class_to_knowledge_base(new_class)

        encoded_labels = self.encoder.transform(string_labels)
        zda_labels = torch.Tensor([flow.zda for flow in flows])
        test_zda_labels = torch.Tensor([flow.test_zda for flow in flows])

        return encoded_labels.to(torch.long), zda_labels, test_zda_labels
    

    def assembly_input_tensor(
            self,
            flows,
            node_feats):
        """
        A batch is composed of a set of flows. 
        Each Flow has a bidimensional feature tensor. 
        (self.MAX_FLOW_TIMESTEPS x 4 features)
        """
        flow_input_batch = flows[0].get_feat_tensor().unsqueeze(0)
        packet_input_batch = None
        node_feat_input_batch = None

        if self.use_packet_feats:
            packet_input_batch = flows[0].packets_tensor.buffer.unsqueeze(0)
        if self.use_node_feats:
            flows[0].node_feats = -1 * torch.ones(
                    size=(10,5),
                    device=self.device)
            if flows[0].dest_ip in node_feats.keys():
                flows[0].node_feats[:len(node_feats[flows[0].dest_ip][CPU]),:]  = torch.hstack([
                        torch.Tensor(node_feats[flows[0].dest_ip][CPU]).unsqueeze(1),
                        torch.Tensor(node_feats[flows[0].dest_ip][RAM]).unsqueeze(1),
                        torch.Tensor(node_feats[flows[0].dest_ip][IN_TRAFFIC]).unsqueeze(1),
                        torch.Tensor(node_feats[flows[0].dest_ip][OUT_TRAFFIC]).unsqueeze(1),
                        torch.Tensor(node_feats[flows[0].dest_ip][DELAY]).unsqueeze(1)])
            
            node_feat_input_batch = flows[0].node_feats.unsqueeze(0)

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
            if self.use_node_feats:
                flow.node_feats = -1 * torch.ones(
                        size=(10,5),
                        device=self.device)
                if flow.dest_ip in node_feats.keys():
                    flow.node_feats[:len(node_feats[flow.dest_ip][CPU]),:] = torch.hstack([
                        torch.Tensor(node_feats[flow.dest_ip][CPU]).unsqueeze(1),
                        torch.Tensor(node_feats[flow.dest_ip][RAM]).unsqueeze(1),
                        torch.Tensor(node_feats[flow.dest_ip][IN_TRAFFIC]).unsqueeze(1),
                        torch.Tensor(node_feats[flow.dest_ip][OUT_TRAFFIC]).unsqueeze(1),
                        torch.Tensor(node_feats[flow.dest_ip][DELAY]).unsqueeze(1)]) 
                       
                node_feat_input_batch = torch.cat(
                            [node_feat_input_batch,
                            flow.node_feats.unsqueeze(0)],
                            dim=0)
                        
        return flow_input_batch, packet_input_batch, node_feat_input_batch
    

    def plot_confusion_matrix(
            self,
            mod,
            cm,
            phase,
            norm=True,
            dims=(10,10),
            classes=None):

        if norm:
            # Rapresented classes:
            rep_classes = cm.sum(1) > 0
            # Normalize
            denom = cm.sum(1).reshape(-1, 1)
            denom[~rep_classes] = 1
            cm = cm / denom
            fmt_str = ".2f"
        else:
            fmt_str = ".0f"

        # Plot heatmap using seaborn
        sns.set_theme()
        plt.figure(figsize=dims)
        ax = sns.heatmap(
            cm,
            annot=True,
            cmap='Blues',
            fmt=fmt_str,
            xticklabels=classes, 
            yticklabels=classes)

        # Rotate x-axis and y-axis labels vertically
        ax.set_xticklabels(classes, rotation=90)
        ax.set_yticklabels(classes, rotation=0)

        # Add x and y axis labels
        plt.xlabel("Predicted")
        plt.ylabel("Baseline")
        plt.title(f'{phase} Confusion Matrix')
        
        if self.wbl is not None:
            self.wbl.log({f'{phase} {mod} Confusion Matrix': wandbImage(plt), STEP_LABEL:self.backprop_counter})

        plt.cla()
        plt.close()
    
    
    def plot_hidden_space(
        self,
        hiddens,
        labels, 
        predicted_labels,
        phase):

        color_iterator = itertools.cycle(colors)
        # If dimensionality is > 2, reduce using PCA
        if hiddens.shape[1]>2:
            pca = PCA(n_components=2)
            hiddens = pca.fit_transform(hiddens)

        plt.figure(figsize=(16, 6))

        # Real labels
        plt.subplot(1, 2, 1)
        # List of attacks:
        unique_labels = torch.unique(labels)
        for label in unique_labels:
            data = hiddens[labels.squeeze(1) == label]
            p_label = self.encoder.inverse_transform(label.unsqueeze(0))[0]
            color_for_scatter = next(color_iterator)
            plt.scatter(
                data[:, 0],
                data[:, 1],
                label=p_label,
                c=color_for_scatter,
                alpha=0.5,
                s=200)
        plt.title(f'{phase} Ground-truth clusters')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        # Predicted labels
        plt.subplot(1, 2, 2)
        unique_labels = torch.unique(predicted_labels)
        for label in unique_labels:
            data = hiddens[predicted_labels == label]
            color_for_scatter = next(color_iterator)
            plt.scatter(
                data[:, 0],
                data[:, 1],
                label=label.item(),
                c=color_for_scatter,
                alpha=0.5,
                s=200)
        plt.title(f'{phase} Predicted clusters')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()

        if self.wbl is not None:
            self.wbl.log({f"{phase} Latent Space Representations": wandbImage(plt)})

        plt.cla()
        plt.close()


    def plot_scores_vectors(
        self,
        score_vectors,
        labels,
        phase):

        # Create an iterator that cycles through the colors
        color_iterator = itertools.cycle(colors)
        
        pca = PCA(n_components=2)
        score_vectors = pca.fit_transform(score_vectors.detach())

        plt.figure(figsize=(10, 6))

        # Two plots:
        plt.subplot(1, 1, 1)
        
        # List of attacks:
        unique_labels = torch.unique(labels)

        # Print points for each attack
        for label in unique_labels:

            data = score_vectors[labels.squeeze(1) == label]
            p_label = self.encoder.inverse_transform(label.unsqueeze(0))[0]

            color_for_scatter = next(color_iterator)

            plt.scatter(
                data[:, 0],
                data[:, 1],
                label=p_label,
                c=color_for_scatter,
                alpha=0.5,
                s=200)
                
        plt.title(f'{phase} PCA reduction of association scores')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))


        plt.tight_layout()
        
        if self.wbl is not None:
            self.wbl.log({f"{phase} PCA of ass. scores": wandbImage(plt)})

        plt.cla()
        plt.close()