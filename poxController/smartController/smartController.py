# Copyright 2011,2012 James McCauley
#
# This file is part of POX.
#
# POX is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# POX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with POX.  If not, see <http://www.gnu.org/licenses/>.

"""
A Smart L3 switch

For each switch:
1) Keep a table that maps IP addresses to MAC addresses and switch ports.
   Stock this table using information from ARP and IP packets.
2) When you see an ARP query, try to answer it using information in the table
   from step 1.  If the info in the table is old, just flood the query.
3) Flood all other ARPs.
4) When you see an IP packet, if you know the destination port (because it's
   in the table from step 1), install a flow for it.
"""
from pox.core import core
from pox.lib.packet.ethernet import ethernet, ETHER_BROADCAST
from pox.lib.packet.ipv4 import ipv4
from pox.lib.packet.arp import arp
from pox.lib.recoco import Timer
from pox.lib.util import str_to_bool
import pox.openflow.libopenflow_01 as of
from pox.lib.revent import *
import time
import argparse
from pox.lib.recoco import Timer
from pox.openflow.of_json import *
from pox.lib.addresses import EthAddr
from smartController.entry import Entry
from smartController.flowlogger import FlowLogger
from smartController.controller_brain import ControllerBrain
from smartController.metricslogger import MetricsLogger
from collections import defaultdict
from smartController.curricula import \
  AC0_TRAINING_LABELS_DICT, AC0_TEST_ZDA_DICT, AC0_ZDA_DICT, \
  AC1_TRAINING_LABELS_DICT, AC1_TEST_ZDA_DICT, AC1_ZDA_DICT, \
  AC2_TRAINING_LABELS_DICT, AC2_TEST_ZDA_DICT, AC2_ZDA_DICT


log = core.getLogger()

openflow_connection = None  # openflow connection to switch is stored here
FLOWSTATS_FREQ_SECS = None  # Interval in which the FLOW stats request is triggered
PORTSTATS_FREQ_SECS = None  # Interval in which the PORT stats request is triggered

class SmartSwitch(EventMixin):
  """
  For each switch:
  1) Keep a table that maps IP addresses to MAC addresses and switch ports.
    Stock this table using information from ARP and IP packets.
  2) When you see an ARP query, try to answer it using information in the table
    from step 1.  If the info in the table is old, just flood the query.
  3) Flood all other ARPs.
  4) When you see an IP packet, if you know the destination port (because it's
    in the table from step 1), install a flow for it.
   """
  def __init__ (
        self, 
        flow_logger, 
        metrics_logger, 
        brain,
        flow_idle_timeout: int = 10,
        arp_timeout: int = 120,
        max_buffered_packets:int = 5,
        max_buffering_secs:int = 5,
        arp_req_exp_secs:int = 4,
        inference_freq_secs:int = 5
        ):

    self.flow_idle_timeout = flow_idle_timeout
    self.arp_timeout = arp_timeout
    self.max_buffered_packets = max_buffered_packets
    self.max_buffering_secs = max_buffering_secs
    self.arp_req_exp_secs = arp_req_exp_secs

    # We use this to prevent ARP flooding
    # Key: (switch_id, ARPed_IP) Values: ARP request expire time
    self.recently_sent_ARPs = {}

    # self.unprocessed_flows is a dict where:
    # keys 2-tuples: (switch_id, dst_ip)
    # values: list of 4-tuples: [(expire_time, packet_id, input_port, scr_ip), ...]
    # It flow packets which we can't deliver because we don't know where they go.
    self.unprocessed_flows = {}

    # For each switch, we map destination IP addresses to Entries
    # (Entries are pairs of switch output ports and MAC addresses)
    self.arpTables = {}

    self.brain = brain

    # This timer handles expiring stuff 
    # Doesnt seems having to do with time to live stuff
    self._expire_timer = Timer(5, self._handle_expiration, recurring=True)

    # Call the smart check function repeatedly:
    self.smart_check_timer = Timer(inference_freq_secs, self.smart_check, recurring=True)

    # Our flow logger instance:
    self.flow_logger = flow_logger

    self.metrics_logger = metrics_logger

    core.listen_to_dependencies(self)


  def smart_check(self):
      self.brain.classify_duet(
        flows=list(self.flow_logger.flows_dict.values()))


  def _handle_expiration(self):
    # Called by a timer so that we can remove old items.
    to_delete_flows = []

    for flow_metadata, packet_metadata_list in self.unprocessed_flows.items():
      switch_id, _ = flow_metadata

      if len(packet_metadata_list) == 0: to_delete_flows.append(flow_metadata)
      else: 
        for packet_metadata in list(packet_metadata_list):
          
          expires_at, packet_id, in_port, _ = packet_metadata

          if expires_at < time.time():
            # This packet is old. Remove it from the buffer.
            packet_metadata_list.remove(packet_metadata)
            # Tell this switch to drop such a packet:
            # To do that we simply send an action-empty openflow message
            # containing the buffer id and the input port of the switch.
            po = of.ofp_packet_out(buffer_id=packet_id, in_port = in_port)
            core.openflow.sendToDPID(switch_id, po)

    # Remove empty flow entries from the unprocessed_flows dictionary
    for flow_metadata in to_delete_flows:
      del self.unprocessed_flows[flow_metadata]


  def _send_unprocessed_flows(self, switch_id, port, dest_mac_addr, dest_ip_addr):
    """
    Unprocessed flows are those we didn't know
    where to send at the time of arrival.  We may know now.  Try and see.
    """
    query_tuple = (switch_id, dest_ip_addr)
    if query_tuple in self.unprocessed_flows.keys():
      
      bucket = self.unprocessed_flows[query_tuple]    
      del self.unprocessed_flows[query_tuple]

      log.debug(f"Sending {len(bucket)} buffered packets to {dest_ip_addr}")
      
      for _, packet_id, in_port, _ in bucket:
        po = of.ofp_packet_out(buffer_id=packet_id, in_port=in_port)
        po.actions.append(of.ofp_action_dl_addr.set_dst(dest_mac_addr))
        po.actions.append(of.ofp_action_output(port = port))
        core.openflow.sendToDPID(switch_id, po)


  def delete_ip_flow_matching_rules(self, dest_ip, connection):
      switch_id = connection.dpid

      msg = of.ofp_flow_mod(command=of.OFPFC_DELETE)
      msg.match.nw_dst = dest_ip
      msg.match.dl_type = ethernet.IP_TYPE
      connection.send(msg)

      log.info(f"Switch {switch_id} will delete flow rules matching nw_dst={dest_ip}")


  def learn_or_update_arp_table(
        self, 
        ip_addr,
        mac_addr,
        port, 
        connection):
      
      switch_id = connection.dpid 

      if ip_addr in self.arpTables[switch_id] and \
        self.arpTables[switch_id][ip_addr] != (port, mac_addr):
            
            # Update switch_port/MAC info
            self.delete_ip_flow_matching_rules(
              dest_ip=ip_addr,
              connection=connection)

      # Learn switch_port/MAC info
      self.arpTables[switch_id][ip_addr] = Entry(
                                            port=port, 
                                            mac=mac_addr, 
                                            ARP_TIMEOUT=self.arp_timeout)
      
      log.debug(f"Entry added/updated to switch {switch_id}'s internal arp table: "+\
                f"(port:{port} ip:{ip_addr})")
        

  def add_ip_to_ip_flow_matching_rule(self, 
                                 switch_id,
                                 source_ip_addr, 
                                 dest_ip_addr, 
                                 dest_mac_addr, 
                                 outgoing_port,
                                 connection,
                                 packet_id,
                                 type):

      actions = []
      actions.append(of.ofp_action_dl_addr.set_dst(dest_mac_addr))
      actions.append(of.ofp_action_output(port = outgoing_port))

      match = of.ofp_match(
        dl_type = type, 
        nw_src = source_ip_addr,
        nw_dst = dest_ip_addr)

      msg = of.ofp_flow_mod(command=of.OFPFC_ADD,
                            idle_timeout=self.flow_idle_timeout,
                            hard_timeout=of.OFP_FLOW_PERMANENT,
                            buffer_id=packet_id,
                            actions=actions,
                            match=match)
      
      connection.send(msg.pack())

      log.debug(f"Added new flow rule to:{switch_id}"+\
                f"match: {match} actions: {actions}")


  def build_and_send_ARP_request(
        self, 
        switch_id, 
        incomming_port,
        source_mac_addr,
        source_ip_addr,
        dest_ip_addr,
        connection):
      
      request = arp()
      request.hwtype = request.HW_TYPE_ETHERNET
      request.prototype = request.PROTO_TYPE_IP
      request.hwlen = 6
      request.protolen = request.protolen
      request.opcode = request.REQUEST
      request.hwdst = ETHER_BROADCAST
      request.protodst = dest_ip_addr
      request.hwsrc = source_mac_addr
      request.protosrc = source_ip_addr
      e = ethernet(type=ethernet.ARP_TYPE, src=source_mac_addr,
                    dst=ETHER_BROADCAST)
      e.set_payload(request)
      
      log.debug(f"{switch_id}'s port {incomming_port} ARPing for {dest_ip_addr} on behalf of {source_ip_addr}")

      msg = of.ofp_packet_out()
      msg.data = e.pack()
      msg.actions.append(of.ofp_action_output(port = of.OFPP_FLOOD))
      msg.in_port = incomming_port
      connection.send(msg)


  def add_unprocessed_packet(self, switch_id,dst_ip,port,src_ip,buffer_id):
    
    tuple_key = (switch_id, dst_ip)
    if tuple_key not in self.unprocessed_flows: 
      self.unprocessed_flows[tuple_key] = []
    packet_metadata_list = self.unprocessed_flows[tuple_key]
    packet_metadata = (time.time() + self.max_buffering_secs, 
                       buffer_id, 
                       port,
                       src_ip)
    packet_metadata_list.append(packet_metadata)
    while len(packet_metadata_list) > self.max_buffered_packets: 
       del packet_metadata_list[0]


  def handle_unknown_ip_packet(self, switch_id, incomming_port, packet_in_event):
    """
    First, track this buffer so that we can try to resend it later, when we will learn the destination.
    Second, ARP for the destination, which should ultimately result in it responding and us learning where it is
    """

    packet = packet_in_event.parsed
    source_mac_addr = packet.src
    source_ip_addr = packet.next.srcip
    dest_ip_addr = packet.next.dstip
    
    self.add_unprocessed_packet(switch_id=switch_id,
                                dst_ip=dest_ip_addr,
                                port=incomming_port,
                                src_ip=source_ip_addr,
                                buffer_id=packet_in_event.ofp.buffer_id)

    # Expire things from our recently_sent_ARP list...
    self.recently_sent_ARPs = {k:v for k, v in self.recently_sent_ARPs.items() if v > time.time()}

    # Check if we've already ARPed recently
    if (switch_id, dest_ip_addr) in self.recently_sent_ARPs:
      # Oop, we've already done this one recently.
      return

    # Otherwise, ARP...
    self.recently_sent_ARPs[(switch_id, dest_ip_addr)] = time.time() + self.arp_req_exp_secs

    self.build_and_send_ARP_request(
        switch_id, 
        incomming_port,
        source_mac_addr,
        source_ip_addr,
        dest_ip_addr,
        connection=packet_in_event.connection)
    
  
  def try_creating_flow_rule(self, switch_id,incomming_port, packet_in_event):
      packet = packet_in_event.parsed
      source_ip_addr = packet.next.srcip
      dest_ip_addr = packet.next.dstip

      if dest_ip_addr in self.arpTables[switch_id]:
          # destination address is present in the arp table
          # get mac and out port
          outgoing_port = self.arpTables[switch_id][dest_ip_addr].port

          if outgoing_port != incomming_port:
              
              dest_mac_addr = self.arpTables[switch_id][dest_ip_addr].mac
              self.add_ip_to_ip_flow_matching_rule(
                                switch_id,
                                source_ip_addr, 
                                dest_ip_addr, 
                                dest_mac_addr, 
                                outgoing_port,
                                connection=packet_in_event.connection,
                                packet_id=packet_in_event.ofp.buffer_id,
                                type=packet.type)

      else:
          self.handle_unknown_ip_packet(switch_id, incomming_port, packet_in_event)


  def handle_ipv4_packet_in(self, switch_id, incomming_port, packet_in_event):
      
      packet = packet_in_event.parsed

      log.debug("IPV4 DETECTED - SWITCH: %i ON PORT: %i IP SENDER: %s IP RECEIVER %s", 
                switch_id,
                incomming_port,
                packet.next.srcip,
                packet.next.dstip)
      
      # Save the first packets of each flow for inference purposes...
      create_flow_rule = self.flow_logger.cache_unprocessed_packets(
         src_ip=packet.next.srcip,
         dst_ip=packet.next.dstip,
         packet=packet)
      
      # Send any waiting packets for that ip
      self._send_unprocessed_flows(
         switch_id, 
         incomming_port, 
         dest_mac_addr=packet.src,
         dest_ip_addr=packet.next.srcip)

      self.learn_or_update_arp_table(ip_addr=packet.next.srcip,
                                     mac_addr=packet.src,
                                     port=incomming_port, 
                                     connection=packet_in_event.connection)

      if create_flow_rule:
          self.try_creating_flow_rule(switch_id, 
                                      incomming_port, 
                                      packet_in_event)


  def send_arp_response(
        self, 
        connection,
        l2_packet,
        l3_packet,
        outgoing_port):
      
      switch_id = connection.dpid

      arp_response = arp()
      arp_response.hwtype = l3_packet.hwtype
      arp_response.prototype = l3_packet.prototype
      arp_response.hwlen = l3_packet.hwlen
      arp_response.protolen = l3_packet.protolen
      arp_response.opcode = arp.REPLY
      arp_response.hwdst = l3_packet.hwsrc
      arp_response.protodst = l3_packet.protosrc
      arp_response.protosrc = l3_packet.protodst
      arp_response.hwsrc = self.arpTables[switch_id][l3_packet.protodst].mac

      ethernet_wrapper = ethernet(type=l2_packet.type, 
                   src=dpid_to_mac(switch_id),
                    dst=l3_packet.hwsrc)
      
      ethernet_wrapper.set_payload(arp_response)

      log.debug(f"ARP ANSWER from switch {switch_id}: ADDRESS:{arp_response.protosrc}")

      msg = of.ofp_packet_out()
      msg.data = ethernet_wrapper.pack()
      msg.actions.append(of.ofp_action_output(port =of.OFPP_IN_PORT))
      msg.in_port = outgoing_port
      connection.send(msg)
      
      
  def handle_arp_packet_in(self, switch_id, incomming_port, packet_in_event):
      
      packet = packet_in_event.parsed
      inner_packet = packet.next

      arp_operation = ''
      if inner_packet.opcode == arp.REQUEST: arp_operation = 'request'
      elif inner_packet.opcode == arp.REPLY: arp_operation = 'reply'
      else: arp_operation = 'op_'+ str(inner_packet.opcode)

      log.debug(f"ARP {arp_operation} received: SWITCH: {switch_id} IN PORT:{incomming_port} ARP FROM: {inner_packet.protosrc} TO {inner_packet.protodst}")

      if inner_packet.prototype == arp.PROTO_TYPE_IP and \
        inner_packet.hwtype == arp.HW_TYPE_ETHERNET and \
          inner_packet.protosrc != 0:

          self.learn_or_update_arp_table(ip_addr=inner_packet.protosrc,
                                     mac_addr=packet.src,
                                     port=incomming_port, 
                                     connection=packet_in_event.connection)
          
          # Send any waiting packets...
          self._send_unprocessed_flows(
              switch_id, 
              incomming_port, 
              dest_mac_addr=packet.src,
              dest_ip_addr=inner_packet.protosrc)

          if inner_packet.opcode == arp.REQUEST and \
            inner_packet.protodst in self.arpTables[switch_id] and \
              not self.arpTables[switch_id][inner_packet.protodst].isExpired():
                '''
                An ARP request has been received, the corresponding 
                switch has the answer, and such an answer is not expired
                '''

                self.send_arp_response(connection=packet_in_event.connection,
                                       l2_packet=packet,
                                       l3_packet=inner_packet,
                                       outgoing_port=incomming_port)

                return

      # Didn't know how to answer or otherwise handle the received ARP, so just flood it
      log.debug(f"Flooding ARP {arp_operation} Switch: {switch_id} IN_PORT: {incomming_port} from:{inner_packet.protosrc} to:{inner_packet.protodst}")

      msg = of.ofp_packet_out(
         in_port = incomming_port, 
         data = packet_in_event.ofp,
         action = of.ofp_action_output(port = of.OFPP_FLOOD))
      
      packet_in_event.connection.send(msg)



  def _handle_openflow_PacketIn(self, event):
    switch_id = event.connection.dpid
    incomming_port = event.port
    packet = event.parsed

    if not packet.parsed:
      log.warning(f"switch {switch_id}, port {incomming_port}: ignoring unparsed packet")
      return
  
    if switch_id not in self.arpTables:
      # New switch -- create an empty table
      log.info(f"New switch detected - creating empty flow table with id {switch_id}")
      self.arpTables[switch_id] = {}
      
    if packet.type == ethernet.LLDP_TYPE:
      #Ignore lldp packets
      return

    if isinstance(packet.next, ipv4):
      
        self.handle_ipv4_packet_in(
          switch_id=switch_id,
          incomming_port=incomming_port,
          packet_in_event=event)

    elif isinstance(packet.next, arp):
        self.handle_arp_packet_in(
          switch_id=switch_id,
          incomming_port=incomming_port,
          packet_in_event=event
        )
      


#global methods
def dpid_to_mac (dpid):
  return EthAddr("%012x" % (dpid & 0xffFFffFFffFF,))
   

#handle switch section
#connection enstablished 
def _handle_ConnectionUp (event):
  global openflow_connection
  openflow_connection=event.connection
  log.info("Connection is UP")
  # Request stats periodically
  Timer(FLOWSTATS_FREQ_SECS, requests_stats, recurring=True)


def requests_stats():
  for connection in core.openflow._connections.values():
    connection.send(of.ofp_stats_request(body=of.ofp_flow_stats_request()))
    connection.send(of.ofp_stats_request(body=of.ofp_port_stats_request()))
  log.debug("Sent %i flow/port stats request(s)", len(core.openflow._connections))


def launch(**kwargs):
    """
    Launches the SmartController Component. This method is automatically invoked when 
    the smartController component is run from pox using the following command:
    
    ```
    $ pox.py smartController.smartController 
    ```

    It migth be useful to launch some PoX logging components alonside the smartController:

    ```
    $ /pox/pox.py samples.pretty_log smartController.smartController 
    ```
    
    You can also add parameters for this component using the longparam syntax (i.e., double hyphens):

      ```
    $ /pox/pox.py samples.pretty_log smartController.smartController --use_packet_feats=false --multi_class=false --flow_idle_timeout=3
    ```

    The available parameters are:

    Parameters:
    ----------
    **kwargs : dict, optional
        Keyword arguments to customize the behavior of the function.
        
        - 'eval' (bool, optional): Use neural modules in evaluation mode, no training. Default is false.
        - 'device' (str, optional): Device for the neural modules, can be 'cpu' or 'cuda:0', 'cuda:1', etc. Depending on hw availability. Default is 'cpu'.
        - 'seed' (int, optional): A fixed initial seed for random sampling and storing in the experience buffer. Defatul is 777.
        - 'ai_debug' (bool, optional): A boolean flag for a verbose version of ML related code. Default is False.
        - 'multi_class' (bool, optional): A boolean flag indicating if the ML classfier should perform multi-class classificaion, otherwise binary (attack, bening). Default is True.
        - 'use_packet_feats' (bool, optional):  Use first bytes of first packets of each flow alonside flowstats to build feature vectors. Default is True.
        - 'packet_buffer_len' (int, optional):  If 'use_packet_feats'=True, indicates how many packets are retained from each flow to build feature vectors. Default is 1.
        - 'flow_buff_len' (int, optional): Indicates the flowstats time-window size to build feature vectors. Default is 10.
        - 'node_features' (bool, optional): Use node features to build the feature vectors. Default is False.
        - 'metric_buffer_len' (int, optional): If 'node_features'=True, node features time-window size to build feat. vectors. Default is 10. 
        - 'inference_freq_secs' (int, optional): Time in secs. between two consecutive calls to an inference. (And a batch training from experience buffer).
        - 'grafana_user' (str, optional): username for the grafana dashboard. Default is 'admin'
        - 'grafana_password' (str, optional): password for the grafana dashboard. Default is 'admin'
        - 'max_kafka_conn_retries' (int, optional): max. num. of connections attempts to the kafka broker. Default is 5.
        - 'curriculum' (int, optional): labelling scheme for known attacks, train ZdAs and test ZdAs. (can be 0,1, or 2). Default is 1. 
        - 'wb_tracking' (bool, optional): Track this run with WeightsAndBiases. (default is False)
        - 'wb_project_name' (str, optional): if 'wb_tracking'=True, The name of the W&B project to associate the run with. Default is 'SmartVille'
        - 'wb_run_name' (str, optional): if 'wb_tracking'=True, the name of the W&B run to track training metrics. Default is AC{curriculum}|DROP {dropout}|H_DIM {h_dim}|{packet_buffer_len}-PKT|{flow_buff_len}TS
        - 'FLOWSTATS_FREQ_SECS' (int, optional): Num. of seconds between two consecutive flowstats request from this controller to its assigned switch. Default is 5.
        - 'PORTSTATS_FREQ_SECS' (int, optional): Num. of seconds between two consecutive portstats request from this controller to its assigned switch. Default is 5.
        - 'flow_idle_timeout' (int, optional): Number of no-activity seconds  for a flow that triggers the switch to delete the corresponding entry in the flow table. Default is 10.
        - 'arp_timeout' (int, optional): Number of seconds to delete and ARP entry in the switch. Default is 120-
        - 'max_buffered_packets' (int, optional): Max num. of packets the switch buffers while waiting to solve the dest MAC address and forward. Default is 5. 
        - 'max_buffering_secs' (int, optional): Max time in seconds the switch buffers a packet while waiting to solve the dest MAC address and forward. Default is 5. 
        - 'arp_req_exp_secs' (int, optional): Max time in secs. that the switch waits for an ARP response before issuing another requets. (prevents ARP flooding). Default is 4. 

    """
    global FLOWSTATS_FREQ_SECS

    eval = str_to_bool(kwargs.get('eval', False))
    device = kwargs.get('device', 'cpu')
    seed = int(kwargs.get('seed', 777))
    ai_debug = str_to_bool(kwargs.get('ai_debug', False))


    multi_class = str_to_bool(kwargs.get('multi_class', True))
    use_packet_feats = str_to_bool(kwargs.get('use_packet_feats', True))
    packet_buffer_len = int(kwargs.get('packet_buffer_len', 1))
    packet_feat_dim = int(kwargs.get('packet_feat_dim', 64))
    h_dim = int(kwargs.get('h_dim', 800))
    dropout = float(kwargs.get('dropout', 0.6))
    k_shot = int(kwargs.get('k_shot', 5))
    batch_size = int(kwargs.get('batch_size', 20))
    anonymize_transport_ports = str_to_bool(kwargs.get('anonym_ports', True))
    flow_buff_len = int(kwargs.get('flow_buff_len', 10))
    node_features = str_to_bool(kwargs.get('node_features', False))
    metric_buffer_len = int(kwargs.get('metric_buffer_len', 10))
    grafana_user = kwargs.get('grafana_user', 'admin')
    grafana_password = kwargs.get('grafana_password', 'admin')
    max_kafka_conn_retries = int(kwargs.get('max_kafka_conn_retries', 5))
    curriculum = int(kwargs.get('curriculum', 1))

    wb_tracking = str_to_bool(kwargs.get('wb_tracking', False))
    wb_project_name = kwargs.get('wb_project_name', 'SmartVille')
    wb_run_name = kwargs.get('wb_run_name', f"AC{curriculum}|DROP {dropout}|H_DIM {h_dim}|{packet_buffer_len}-PKT|{flow_buff_len}TS")
    FLOWSTATS_FREQ_SECS = int(kwargs.get('flowstats_freq_secs', 5))
    PORTSTATS_FREQ_SECS = int(kwargs.get('portstats_freq_secs', 5))


    # Switching arguments:
    switching_args = {}
    switching_args['flow_idle_timeout'] = int(kwargs.get('flow_idle_timeout', 10))
    switching_args['arp_timeout'] = int(kwargs.get('arp_timeout', 120))
    switching_args['max_buffered_packets'] = int(kwargs.get('max_buffered_packets', 5))
    switching_args['max_buffering_secs'] = int(kwargs.get('max_buffering_secs', 5))
    switching_args['arp_req_exp_secs'] = int(kwargs.get('arp_req_exp_secs', 4))
    switching_args['inference_freq_secs'] = int(kwargs.get('inference_freq_secs', 5))


    ########################### LABELING:

    if multi_class:

        if curriculum == 0:
              TRAINING_LABELS_DICT = AC0_TRAINING_LABELS_DICT
              ZDA_DICT = AC0_ZDA_DICT
              TEST_ZDA_DICT = AC0_TEST_ZDA_DICT
        if curriculum == 1:
              TRAINING_LABELS_DICT = AC1_TRAINING_LABELS_DICT
              ZDA_DICT = AC1_ZDA_DICT
              TEST_ZDA_DICT = AC1_TEST_ZDA_DICT
        if curriculum == 2:
          TRAINING_LABELS_DICT = AC2_TRAINING_LABELS_DICT
          ZDA_DICT = AC2_ZDA_DICT
          TEST_ZDA_DICT = AC2_TEST_ZDA_DICT

    else:
        
        TRAINING_LABELS_DICT= defaultdict(lambda: "Bening") # class "bening" is default and is reserved for leggittimate traffic. 
        TRAINING_LABELS_DICT["192.168.1.7"] = "Attack"
        TRAINING_LABELS_DICT["192.168.1.8"] = "Attack"
        TRAINING_LABELS_DICT["192.168.1.9"] = "Attack"

        TRAINING_LABELS_DICT["192.168.1.10"] = "Attack"
        TRAINING_LABELS_DICT["192.168.1.11"] = "Attack"
        TRAINING_LABELS_DICT["192.168.1.12"] = "Attack"
        TRAINING_LABELS_DICT["192.168.1.13"] = "Attack"

        TRAINING_LABELS_DICT["192.168.1.14"] = "Attack"
        TRAINING_LABELS_DICT["192.168.1.15"] = "Attack"
        TRAINING_LABELS_DICT["192.168.1.16"] = "Attack"
    
    ########################### END OF LABELING



    # Registering PacketLogger component:
    flow_logger = FlowLogger(
      training_labels_dict=TRAINING_LABELS_DICT,
      zda_dict=ZDA_DICT,
      test_zda_dict=TEST_ZDA_DICT,
      multi_class=multi_class,
      packet_buffer_len=packet_buffer_len,
      packet_feat_dim=packet_feat_dim,
      anonymize_transport_ports=anonymize_transport_ports,
      flow_feat_dim=4,
      flow_buff_len=flow_buff_len)

    metrics_logger=None

    if node_features:
      metrics_logger = MetricsLogger(
        server_addr = "192.168.1.1:9092",
        max_conn_retries = max_kafka_conn_retries,
        metric_buffer_len = metric_buffer_len,
        grafana_user=grafana_user, 
        grafana_pass=grafana_password,
        )
  
    # The controllerBrain holds the ML functionalities.
    controller_brain = ControllerBrain(
        eval=eval,
        use_packet_feats=use_packet_feats,
        flow_feat_dim=4,
        packet_feat_dim=packet_feat_dim,
        h_dim=h_dim,
        dropout=dropout,
        multi_class=multi_class, 
        k_shot=k_shot,
        replay_buffer_batch_size=batch_size,
        kernel_regression=True,
        device=device,
        seed=seed,
        debug=ai_debug,
        wb_track=wb_tracking,
        wb_project_name=wb_project_name,
        wb_run_name=wb_run_name,
        wb_config_dict=kwargs)
      
    # Registering Switch component:
    smart_switch = SmartSwitch(
      flow_logger=flow_logger,
      metrics_logger=metrics_logger,
      brain=controller_brain,
      **switching_args
      )
    
    core.register("smart_switch", smart_switch)  
    
    # attach handlers to listeners
    core.openflow.addListenerByName(
      "ConnectionUp", 
      _handle_ConnectionUp)


    if FLOWSTATS_FREQ_SECS > 0:
      core.openflow.addListenerByName(
        "FlowStatsReceived", 
        flow_logger._handle_flowstats_received)
      
    """
    if PORTSTATS_FREQ_SECS > 0:
      core.openflow.addListenerByName(
        "PortStatsReceived", 
        flow_logger._handle_portstats_received) 
    """
