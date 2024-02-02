from pox.core import core
from pox.openflow.of_json import flow_stats_to_list
from smartController.flow import Flow, IPPacket2Tensor, CircularBuffer
import torch

class FlowLogger(object):
    
    def __init__(
      self,
      ipv4_blacklist_for_training):

      """
      TODO: flows_dict should be one for each switch... or, equivalently, we should use one 
      switch_logger per switch.
      """
      self.flows_dict = {}
      self.unprocessed_flows_packet_cache = {}
      self.logger_instance = core.getLogger()
      self.ipv4_blacklist_for_training = ipv4_blacklist_for_training


    def extract_flow_feature_tensor(self, flow):
       return torch.Tensor(
          [flow['byte_count'], 
            flow['duration_nsec'] / 10e9,
            flow['duration_sec'],
            flow['packet_count']]).to(torch.float32)


    def cache_unprocessed_flow_packet(self, src_ip, dst_ip, packet):
        """
        We need to add some packets among the features of flows to augment the perceptive field of our AI. 
        The packets that arrive at the controller, however, are by definition orphans of flow rules. 
        We cache them until the flow rules are available. Whenever flowstats arrive, we will
        query this cache memory to populate flow features with packet data.
        """
        partial_flow_id = str(src_ip) + "_" + str(dst_ip)
        packet_tensor = IPPacket2Tensor(packet.next).feature_tensor

        if partial_flow_id in self.unprocessed_flows_packet_cache.keys():
            # A tensor already exists:
            curr_packets_circular_buffer = self.unprocessed_flows_packet_cache[partial_flow_id]
        else:
           # Create new circular buffer:
           curr_packets_circular_buffer = CircularBuffer(buffer_size=5, feature_size=100)

        curr_packets_circular_buffer.add(packet_tensor)
        self.unprocessed_flows_packet_cache[partial_flow_id] = curr_packets_circular_buffer


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




