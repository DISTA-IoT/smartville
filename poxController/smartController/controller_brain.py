from smartController.neural_modules import BinaryFlowClassifier, \
    TwoStreamBinaryFlowClassifier, MultiClassFlowClassifier, \
        TwoStreamMulticlassFlowClassifier, ConfidenceDecoder
from smartController.replay_buffer import ReplayBuffer
import os
import torch
import torch.optim as optim
import torch.nn as nn
from smartController.wandb_tracker import WandBTracker
from pox.core import core  # only for logging.
import seaborn as sns
import matplotlib.pyplot as plt
import threading
from wandb import Image as wandbImage
import itertools
from sklearn.decomposition import PCA


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

SAVING_MODULES_FREQ = 50

PRETRAINED_MODEL_PATH = 'models/BinaryFlowClassifier.pt'

REPLAY_BUFFER_MAX_CAPACITY=1000

LEARNING_RATE=1e-3

REPORT_STEP_FREQUENCY = 3

# Constants for wandb monitoring:
INFERENCE = 'Inference'
TRAINING = 'Training'
ACC = 'Acc'
LOSS = 'Loss'
STEP_LABEL = 'step'

# Create a lock object
lock = threading.Lock()


def efficient_cm(preds, targets):

    predictions_decimal = preds.argmax(dim=1).to(torch.int64)
    predictions_onehot = torch.zeros_like(
        preds,
        device=preds.device)
    predictions_onehot.scatter_(1, predictions_decimal.view(-1, 1), 1)

    targets = targets.to(torch.int64)
    # Create a one-hot encoding of the targets.
    targets_onehot = torch.zeros_like(
        preds,
        device=targets.device)
    targets_onehot.scatter_(1, targets.view(-1, 1), 1)

    return targets_onehot.T @ predictions_onehot



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
                 use_packet_feats,
                 flow_feat_dim,
                 packet_feat_dim,
                 multi_class,
                 k_shot,
                 replay_buffer_batch_size,
                 device='cpu',
                 seed=777,
                 debug=False,
                 wb_track=False,
                 wb_project_name='',
                 wb_run_name='',
                 **wb_config_dict):
                
        self.use_packet_feats = use_packet_feats
        self.flow_feat_dim = flow_feat_dim
        self.packet_feat_dim = packet_feat_dim
        self.multi_class = multi_class
        self.AI_DEBUG = debug
        self.best_accuracy = 0
        self.inference_counter = 0
        self.wbt = wb_track
        self.wbl = None
        self.logger_instance = core.getLogger()
        self.device=device
        self.seed = seed
        self.current_known_classes_count = 0
        self.k_shot = k_shot
        self.replay_buff_batch_size = replay_buffer_batch_size
        self.encoder = DynamicLabelEncoder()
        self.confidence_decoder = ConfidenceDecoder(device=device)
        self.replay_buffers = {}
        self.cs_cm = torch.zeros(size=(1,1), device=self.device)
        self.initialize_classifier(LEARNING_RATE, seed)
        
        if self.wbt:

            wb_config_dict['SAVING_MODULES_FREQ'] = SAVING_MODULES_FREQ
            wb_config_dict['PRETRAINED_MODEL_PATH'] = PRETRAINED_MODEL_PATH
            wb_config_dict['REPLAY_BUFFER_MAX_CAPACITY'] = REPLAY_BUFFER_MAX_CAPACITY
            wb_config_dict['LEARNING_RATE'] = LEARNING_RATE
            wb_config_dict['REPORT_STEP_FREQUENCY'] = REPORT_STEP_FREQUENCY

            self.wbl = WandBTracker(
                wanb_project_name=wb_project_name,
                run_name=wb_run_name,
                config_dict=wb_config_dict).wb_logger        


    def add_replay_buffer(self):
        self.inference_allowed = False
        self.experience_learning_allowed = False
        if self.AI_DEBUG:
            self.logger_instance.info(f'Adding a replay buffer with code {self.current_known_classes_count-1}')
            self.logger_instance.info(f'Encoder state mapping: {self.encoder.get_mapping()}')
        self.replay_buffers[self.current_known_classes_count-1] = ReplayBuffer(
            capacity=REPLAY_BUFFER_MAX_CAPACITY,
            batch_size=self.replay_buff_batch_size,
            seed=self.seed)


    def add_class_to_knowledge_base(self, new_class):
        if self.AI_DEBUG:
            self.logger_instance.info(f'New class found: {new_class}')
        self.current_known_classes_count += 1
        self.add_replay_buffer()
        self.cs_cm = torch.zeros(
            size=(self.current_known_classes_count, self.current_known_classes_count),
            device=self.device
            )


    def reset_cm(self):
        self.cs_cm = torch.zeros(
            [self.current_known_classes_count, self.current_known_classes_count],
            device=self.device)


    def initialize_classifier(self, lr, seed):
        
        torch.manual_seed(seed)
        if self.use_packet_feats:

            if self.multi_class:
                self.flow_classifier = TwoStreamMulticlassFlowClassifier(
                flow_input_size=self.flow_feat_dim, 
                packet_input_size=self.packet_feat_dim,
                hidden_size=40,
                dropout_prob=0.1,
                device=self.device)
                self.criterion = nn.CrossEntropyLoss().to(self.device)

            else:
                self.flow_classifier = TwoStreamBinaryFlowClassifier(
                    flow_input_size=self.flow_feat_dim, 
                    packet_input_size=self.packet_feat_dim,
                    hidden_size=40,
                    dropout_prob=0.1,
                    device=self.device)
                self.criterion = nn.BCELoss().to(self.device)
        else:

            if self.multi_class:
                self.flow_classifier = MultiClassFlowClassifier(
                    input_size=self.flow_feat_dim, 
                    hidden_size=40,
                    dropout_prob=0.1,
                    device=self.device)
                self.criterion = nn.CrossEntropyLoss().to(self.device)

            else:
                self.flow_classifier = BinaryFlowClassifier(
                    input_size=self.flow_feat_dim, 
                    hidden_size=40,
                    dropout_prob=0.1,
                    device=self.device)
                self.criterion = nn.BCELoss().to(self.device)
                
        self.check_pretrained()
        self.flow_classifier.to(self.device)
        self.fc_optimizer = optim.Adam(
            self.flow_classifier.parameters(), 
            lr=lr)
        

    def check_pretrained(self):
        # Check if the file exists
        if os.path.exists(PRETRAINED_MODEL_PATH):
            # Load the pre-trained weights
            self.flow_classifier.load_state_dict(torch.load(PRETRAINED_MODEL_PATH))
            if self.AI_DEBUG:
                self.logger_instance.info(f"Pre-trained weights loaded successfully from {PRETRAINED_MODEL_PATH}.")
        elif self.AI_DEBUG:
            self.logger_instance.info(f"Pre-trained weights not found at {PRETRAINED_MODEL_PATH}.")


    def infer(self, flow_input_batch, packet_input_batch, batch_labels, query_mask):

        if self.use_packet_feats:
            if self.multi_class:
                predictions, hiddens = self.flow_classifier(
                    flow_input_batch, 
                    packet_input_batch, 
                    batch_labels, 
                    self.current_known_classes_count,
                    query_mask)
            else:
                predictions, hiddens = self.flow_classifier(
                    flow_input_batch, 
                    packet_input_batch)
        else:
            if self.multi_class:
                predictions, hiddens = self.flow_classifier(
                    flow_input_batch, 
                    batch_labels, 
                    self.current_known_classes_count,
                    query_mask)
            else:
                predictions, hiddens = self.flow_classifier(
                    flow_input_batch)

        return predictions, hiddens


    def push_to_replay_buffers(
            self,
            flow_input_batch, 
            packet_input_batch, 
            batch_labels,
            zda_batch_labels):
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
                for sample_idx in range(flow_input_batch[mask].shape[0]):
                    self.replay_buffers[label.item()].push(
                        flow_input_batch[mask][sample_idx].unsqueeze(0), 
                        packet_input_batch[mask][sample_idx].unsqueeze(0), 
                        label=batch_labels[mask][sample_idx].unsqueeze(0),
                        zda_label=zda_batch_labels[mask][sample_idx].unsqueeze(0))
            else: 
                for sample_idx in range(flow_input_batch[mask].shape[0]):
                    self.replay_buffers[label.item()].push(
                        flow_input_batch[mask][sample_idx].unsqueeze(0), 
                        None, 
                        label=batch_labels[mask][sample_idx].unsqueeze(0),
                        zda_label=zda_batch_labels[mask][sample_idx].unsqueeze(0))
                
        if not self.inference_allowed or not self.experience_learning_allowed:
            buff_lengths = [len(replay_buff) for replay_buff in self.replay_buffers.values()]
            if self.AI_DEBUG:
                self.logger_instance.info(f'Buffer lengths: {buff_lengths}')
            self.inference_allowed = torch.all(
                torch.Tensor([buff_len  > self.k_shot for buff_len in buff_lengths]))
            self.experience_learning_allowed = torch.all(
                torch.Tensor([buff_len  > self.replay_buff_batch_size for buff_len in buff_lengths]))


    def classify_duet(self, flows):
        """
        makes inferences about a duet flow (source ip, dest ip)
        """
        with lock:

            if len(flows) == 0:
                return None
            else:
                flow_input_batch, packet_input_batch = self.assembly_input_tensor(flows)
                batch_labels, zda_labels = self.get_labels(flows)

                self.push_to_replay_buffers(
                    flow_input_batch, 
                    packet_input_batch, 
                    batch_labels=batch_labels,
                    zda_batch_labels=zda_labels)

                if self.inference_allowed:

                    support_flow_batch, support_packet_batch, support_labels, zda_labels = self.sample_from_replay_buffers(
                        samples_per_class=self.k_shot)
                    
                    query_mask = torch.zeros(
                        size=(support_labels.shape[0],), 
                        device=self.device).to(torch.bool)

                    query_mask = torch.cat([query_mask, torch.ones_like(batch_labels).to(torch.bool)])
                    flow_input_batch = torch.vstack([support_flow_batch, flow_input_batch])
                    if self.use_packet_feats:
                        packet_input_batch = torch.vstack([support_packet_batch, packet_input_batch])
                    batch_labels = torch.cat([support_labels.squeeze(1), batch_labels]).unsqueeze(1)

                    predictions, _ = self.infer(
                        flow_input_batch=flow_input_batch,
                        packet_input_batch=packet_input_batch,
                        batch_labels=batch_labels,
                        query_mask=query_mask
                        )

                    accuracy = self.learning_step(batch_labels, predictions, INFERENCE, query_mask)

                    self.inference_counter += 1

                    if self.AI_DEBUG: 
                        self.logger_instance.info(f'inference accuracy: {accuracy}')

                    if self.experience_learning_allowed:
                        self.experience_learning()
                            
        
    def sample_from_replay_buffers(self, samples_per_class):
        balanced_packet_batch = None
        init = True
        for replay_buff in self.replay_buffers.values():
            flow_batch, packet_batch, batch_labels, zda_batch_labels = replay_buff.sample(samples_per_class)
            if init:
                balanced_flow_batch = flow_batch
                balanced_labels = batch_labels
                balanced_zda_labels = zda_batch_labels
                if packet_batch is not None:
                    balanced_packet_batch = packet_batch
            else: 
                balanced_flow_batch = torch.vstack(
                    [balanced_flow_batch, flow_batch])
                balanced_labels = torch.vstack(
                    [balanced_labels, batch_labels])
                balanced_zda_labels = torch.vstack(
                    [balanced_zda_labels, zda_batch_labels])
                if packet_batch is not None:
                    balanced_packet_batch = torch.vstack(
                        [balanced_packet_batch, packet_batch]) 
            init = False

        return balanced_flow_batch, balanced_packet_batch, balanced_labels, balanced_zda_labels


    def get_canonical_query_mask(self):
        query_mask = torch.zeros(
            size=(self.current_known_classes_count, self.replay_buff_batch_size),
            device=self.device).to(torch.bool)
        query_mask[:, self.k_shot:] = True
        return query_mask.view(-1)


    def experience_learning(self):

        balanced_flow_batch, balanced_packet_batch, balanced_labels, balanced_zda_labels = self.sample_from_replay_buffers(
            samples_per_class=self.replay_buff_batch_size)
        
        query_mask = self.get_canonical_query_mask()

        assert query_mask.shape[0] == balanced_labels.shape[0]

        more_predictions, hidden_vectors = self.infer(
            flow_input_batch=balanced_flow_batch,
            packet_input_batch=balanced_packet_batch,
            batch_labels=balanced_labels,
            query_mask=query_mask
        )
        
        self.cs_cm += efficient_cm(
        preds=more_predictions.detach(),
        targets=balanced_labels[query_mask]) 
        
        self.report(
            preds=more_predictions, 
            hiddens=hidden_vectors, 
            labels=balanced_labels, 
            query_mask=query_mask)
        
        accuracy = self.learning_step(balanced_labels, more_predictions, TRAINING, query_mask)

        self.inference_counter += 1
        self.check_progress(curr_acc=accuracy)
        if self.AI_DEBUG: 
            self.logger_instance.info(f'batch labels mean: {balanced_labels.to(torch.float16).mean().item()} '+\
                                      f'batch prediction mean: {more_predictions.max(1)[1].to(torch.float32).mean()}')
            self.logger_instance.info(f'mean training accuracy: {accuracy}')


    def report(self, preds, hiddens, labels, query_mask):

        if self.inference_counter % REPORT_STEP_FREQUENCY == 0:

            if self.wbt:
                self.plot_confusion_matrix(
                    self.cs_cm,phase=TRAINING,
                    norm=False,
                    classes=self.encoder.get_labels())
                self.plot_hidden_space(hiddens=hiddens, labels=labels)
                self.plot_scores_vectors(score_vectors=preds, labels=labels[query_mask])
            elif self.AI_DEBUG:
                    self.logger_instance.info(f'Conf matrix: \n {self.cs_cm}')
            self.reset_cm()


    def check_progress(self, curr_acc):
        if (self.inference_counter % SAVING_MODULES_FREQ == 0) and\
            (self.inference_counter > 0) and\
                  (self.best_accuracy < curr_acc):
            self.best_accuracy = curr_acc
            self.save_models()


    def save_models(self):
        torch.save(
            self.flow_classifier.state_dict(), 
            'BinaryFlowClassifier.pt')
        if self.AI_DEBUG: 
            self.logger_instance.info(f'New model version saved')


    def learning_step(self, labels, predictions, mode, query_mask):
        
        if self.multi_class:
            loss = self.criterion(input=predictions,
                                        target=labels[query_mask].squeeze(1))
        else: 
            loss = self.criterion(input=predictions,
                                        target=labels.to(torch.float32))
        # backward pass
        self.fc_optimizer.zero_grad()
        loss.backward()
        # update weights
        self.fc_optimizer.step()
        # compute accuracy
        acc = self.get_accuracy(logits_preds=predictions, decimal_labels=labels, query_mask=query_mask)

        # report progress
        if self.wbt:
            self.wbl.log({mode+'_'+ACC: acc.item(), STEP_LABEL:self.inference_counter})
            self.wbl.log({mode+'_'+LOSS: loss.item(), STEP_LABEL:self.inference_counter})

        return acc
    

    def get_accuracy(self, logits_preds, decimal_labels, query_mask):
        """
        labels must not be one hot!
        """
        if self.multi_class:
            match_mask = logits_preds.max(1)[1] == decimal_labels.max(1)[0][query_mask]
            return match_mask.sum() / match_mask.shape[0]
        else:
            return (logits_preds.round() == decimal_labels).float().mean()


    def get_labels(self, flows):

        string_labels = [flow.element_class for flow in flows]
        new_classes = self.encoder.fit(string_labels)
        for new_class in new_classes:
            self.add_class_to_knowledge_base(new_class)

        encoded_labels = self.encoder.transform(string_labels)

        zda_labels = [flow.zda for flow in flows]
        zda_labels = torch.Tensor([flow.zda for flow in flows])

        return encoded_labels.to(torch.long), zda_labels
    

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
    

    def plot_confusion_matrix(
            self,
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
            self.wbl.log({f'{phase} Confusion Matrix': wandbImage(plt), STEP_LABEL:self.inference_counter})

        plt.cla()
        plt.close()



    def plot_hidden_space(
        self,
        hiddens,
        labels):

        # Create an iterator that cycles through the colors
        color_iterator = itertools.cycle(colors)

        # If dimensionality is > 2, reduce using PCA
        if hiddens.shape[1]>2:
            pca = PCA(n_components=2)
            hiddens = pca.fit_transform(hiddens)

        plt.figure(figsize=(10, 6))

        # Two plots:
        plt.subplot(1, 1, 1)
        
        # List of attacks:
        unique_labels = torch.unique(labels)

        # Print points for each attack
        for label in unique_labels:
            data = hiddens[labels.squeeze(1) == label]
            p_label = self.encoder.inverse_transform(label.unsqueeze(0))[0]

            color_for_scatter = next(color_iterator)

            if 'ZdA' in p_label:
                color_for_scatter = 'gray'

            plt.scatter(
                data[:, 0],
                data[:, 1],
                label=p_label,
                c=color_for_scatter)
                
        plt.title(f'Latent Space Representations')

        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

        plt.tight_layout()

        if self.wbl is not None:
            self.wbl.log({f"Latent Space Representations": wandbImage(plt)})

        plt.cla()
        plt.close()


    def plot_scores_vectors(
        self,
        score_vectors,
        labels):

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
            if 'ZdA' in p_label:
                color_for_scatter = 'black'

            plt.scatter(
                data[:, 0],
                data[:, 1],
                label=p_label,
                c=color_for_scatter)
                
        plt.title(f'PCA reduction of association scores')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))


        plt.tight_layout()
        
        if self.wbl is not None:
            self.wbl.log({f"PCA of ass. scores": wandbImage(plt)})

        plt.cla()
        plt.close()
