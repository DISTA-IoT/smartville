from pox.core import core
from pox.lib.packet.ipv4 import ipv4
from collections import defaultdict
from pox.openflow.of_json import flow_stats_to_list
from smartController.flow import Flow
import torch
import numpy as np

class PacketLogger(object):
    
    def __init__(
      self,
      ipv4_blacklist_for_training):

      self.packet_lists = {}
        
      self.portstats_lists = defaultdict(list)

      self.flows_dict = {}

      self.logger_instance = core.getLogger()

      self.ipv4_blacklist_for_training = ipv4_blacklist_for_training



    def handle_packet_lists(self, packet, port, packet_data):
        #if the packet is in the blacklist add it as true, if not, as false
        if isinstance(packet.next, ipv4):
          if str(packet.next.srcip) in self.ipv4_blacklist_for_training:
            self.logger_instance.debug(f"IP {packet.next.srcip} in blacklist")
            labeled_packet = (packet_data,True)
          else:
            self.logger_instance.debug(f"IP {packet.next.srcip} not in blacklist")
            labeled_packet = (packet_data,False)

          # append packet:
          self.packet_lists[port][str(packet.next.srcip)].append(labeled_packet)
        
        else:
          self.logger_instance.debug("not an ipv4 packet!")



    def _handle_PacketIn(self, event):
      
      packet= event.parsed
      port = event.port

      if not packet.parsed:
        self.logger_instance.warning("Ignoring incomplete packet")
        return
      
      #Ignore LLC
      if packet.type == 38: return
      
      self.logger_instance.debug("Received packet on port %s:", port)
      
      #if port is not in the dict, then add the port's entry
      if port not in self.packet_lists:
          self.packet_lists[port] = defaultdict(list)

      self.handle_packet_lists(packet=event.parsed, 
                               port=event.port, 
                               packet_data=event.data)


    def reset_packet_lists(self):
      for port in self.packet_lists.keys():
         self.reset_port_list(port=port)

    def reset_port_list(self, port):
       for sender_ip in self.packet_lists[port].keys():
          self.reset_sender_list(sender_port=port, sender_ip=sender_ip)

    def reset_sender_list(self, sender_port, sender_ip):
       self.packet_lists[sender_port][str(sender_ip)]= []

    
    def _handle_portstats_received(self, event):
      stats = flow_stats_to_list(event.stats)
      for stats_dict in stats:
        stats_array = flat_dict_to_tensor(stats_dict)
        self.portstats_lists[stats_dict['port_no']].append(stats_array)
      # self.logger_instance.debug("PortStatsReceived")
      
    
    def reset_all_portstats_lists(self):
      self.portstats_lists = defaultdict(list)


    def reset_portstats_list(self, port):
      self.portstats_lists[port] = []


    def exctract_flow_feature_tensor(self, flow):
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
        
        flow_features = self.exctract_flow_feature_tensor(flow=flow)

        if new_flow.flow_id in self.flows_dict.keys():
          self.flows_dict[new_flow.flow_id].enrich_features(flow_features)
        else:
          new_flow.enrich_features(flow_features)
          self.flows_dict[new_flow.flow_id] = new_flow


    # structure of event.stats is defined by ofp_flow_stats()
    def _handle_flowstats_received (self, event):

      self.logger_instance.debug("FlowStatsReceived")
      stats = flow_stats_to_list(event.stats)

      for sender_flow in stats:
        self.process_received_flow(flow=sender_flow)
        

    def reset_all_flows_metadata(self):
       self.flows_dict = {}


    def reset_single_flow_metadata(self, flow_id):
       del self.flows_dict[flow_id]


def flat_dict_to_tensor(flat_dict):
  array_to_return = list(flat_dict.items())
  array_to_return = np.array(array_to_return)
  array_to_return = array_to_return[:,1].astype(np.float32)
  return torch.from_numpy(array_to_return).to(torch.float32)