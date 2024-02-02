from pox.core import core
from pox.openflow.of_json import flow_stats_to_list
from smartController.flow import Flow
import torch

class FlowLogger(object):
    
    def __init__(
      self,
      ipv4_blacklist_for_training):

      self.flows_dict = {}
      self.logger_instance = core.getLogger()
      self.ipv4_blacklist_for_training = ipv4_blacklist_for_training


    def extract_flow_feature_tensor(self, flow):
       return torch.Tensor(
          [flow['byte_count'], 
            flow['duration_nsec'] / 10e9,
            flow['duration_sec'],
            flow['packet_count']]).to(torch.float32)


    def process_received_flow(self, flow):
        
        sender_ip_addr = flow['match']['nw_src'].split('/')[0]

        new_flow = Flow(
          source_ip=sender_ip_addr, 
          dest_ip=flow['match']['nw_dst'].split('/')[0], 
          switch_output_port=flow['actions'][1]['port'])
        
        new_flow.infected = sender_ip_addr in self.ipv4_blacklist_for_training
        
        flow_features = self.extract_flow_feature_tensor(flow=flow)

        if new_flow.flow_id in self.flows_dict.keys():
          self.flows_dict[new_flow.flow_id].enrich_features(flow_features)
        else:
          new_flow.enrich_features(flow_features)
          self.flows_dict[new_flow.flow_id] = new_flow


    
    def _handle_flowstats_received (self, event):
      self.logger_instance.debug("FlowStatsReceived")
      stats = flow_stats_to_list(event.stats)
      for sender_flow in stats:
        self.process_received_flow(flow=sender_flow)
        

    def reset_all_flows_metadata(self):
       self.flows_dict = {}


    def reset_single_flow_metadata(self, flow_id):
       del self.flows_dict[flow_id]




