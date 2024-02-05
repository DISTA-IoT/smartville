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
      self.packet_cache = {}
      self.logger_instance = core.getLogger()
      self.ipv4_blacklist_for_training = ipv4_blacklist_for_training

    def extract_flow_feature_tensor(self, flow):
       return torch.Tensor(
          [flow['byte_count'], 
            flow['duration_nsec'] / 10e9,
            flow['duration_sec'],
            flow['packet_count']]).to(torch.float32)


    def cache_unprocessed_packets(self, src_ip, dst_ip, packet):
        """
        We need to add some packets among the features of flows to augment the perceptive field of our AI. 
        The packets that arrive at the controller, however, are by definition orphans of flow rules. 
        We cache them until the flow rules are available. Whenever flowstats arrive, we will
        query this cache memory to populate flow features with packet data.

        returns a flag indicating if the buffer is full of data.
        """
        partial_flow_id = str(src_ip) + "_" + str(dst_ip)
        packet_tensor = IPPacket2Tensor(packet.next).feature_tensor

        if partial_flow_id in self.packet_cache.keys():
            # A tensor already exists:
            curr_packets_circ_buff = self.packet_cache[partial_flow_id]
        else:
           # Create new circular buffer:
           curr_packets_circ_buff = CircularBuffer(buffer_size=5, feature_size=100)

        curr_packets_circ_buff.add(packet_tensor)
        self.packet_cache[partial_flow_id] = curr_packets_circ_buff

        return curr_packets_circ_buff.is_full


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

        self.update_packet_buffer(new_flow)


    def update_packet_buffer(self, flow_object):
       """
       Attack the cached packets tensor to the flow entry in flows_dict
       """

       partial_flow_id = "_".join(flow_object.flow_id.split("_")[:-1])

       if partial_flow_id in self.packet_cache.keys():
          
          packets_buffer = self.packet_cache[partial_flow_id]
          del self.packet_cache[partial_flow_id]

          if self.flows_dict[flow_object.flow_id].packets_tensor == None:
             self.flows_dict[flow_object.flow_id].packets_tensor = packets_buffer
          else: 
             for single_packet_tensor in packets_buffer.buffer:
                self.flows_dict[flow_object.flow_id].packets_tensor.add(single_packet_tensor)

    
    def _handle_flowstats_received (self, event):
      self.logger_instance.debug("FlowStatsReceived")
      stats = flow_stats_to_list(event.stats)
      for sender_flow in stats:
        self.process_received_flow(flow=sender_flow)
        

    def reset_all_flows_metadata(self):
       self.flows_dict = {}


    def reset_single_flow_metadata(self, flow_id):
       del self.flows_dict[flow_id]
