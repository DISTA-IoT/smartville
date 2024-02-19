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
import pox.openflow.libopenflow_01 as of
from pox.lib.revent import *
import time
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
  
openflow_connection = None #openflow connection to switch is stored here

log = core.getLogger()


############ SWITCH CONF #######################################
# Timeout for flows
FLOW_IDLE_TIMEOUT = 10
# Timeout for ARP entries
ARP_TIMEOUT = 60 * 2
# Maximum number of packet to buffer on a switch for an unknown IP
MAX_BUFFERED_PER_IP = 5
# Maximum time to hang on to a buffer for an unknown IP in seconds
MAX_BUFFER_TIME = 5
# Don't re-send ARPs before expiring this interval:
ARP_REQUEST_EXPIRATION_SECONDS = 4
# print(f"HARD TIMEOUT IS SET TO {of.OFP_FLOW_PERMANENT} WHICH IS DEFAULT")
#################################################################

######### MONITORING ###########################################
# Interval in which the stats request is triggered
REQUEST_STATS_PERIOD_SECONDS = 5
NODE_FEATURES = False  # Requires Prometheus, Grafana, Zookeeper and Kafka...
GRAFANA_USER='admin'
GRAFANA_PASSWORD='admin'
################################################################


######## AI #####################################################

AI_DEBUG = True
BRAIN_DEVICE = 'cpu' # eventually, the neural networks could be on a GPU.

SEED = 777  # For reproducibility purposes
INFERENCE_FREQ_SECONDS = 5  # Seconds between consecutive calls to forward passes
# Dimention of the feature tensors
PACKET_FEAT_DIM = 64
FLOW_FEAT_DIM = 4

MAX_PACKETS_PER_FEAT_TENSOR = 3  # Max number of packets in the packets feature vector for each flow.
MAX_FLOWSTATS_PER_FEAT_TENSOR = 10  # Max number of flowstats in the feature vector for each flow.
ANONYMIZE_TRANSPORT_PORTS = True  # Mask port info in packets for AI? (IP adresses are masked by default!)
K_SHOT = 5  # FOR EPISODIC LEARNING:
REPLAY_BUFFER_BATCH_SIZE= 20  # MUST BE GREATER THAN K_SHOT!

KERNEL_REGRESSION = True  # learn relations between attacks.
PACKET_FEATURES = True  # use packet features
MULTI_CLASS_CLASSIFICATION = True  # Otherwise binary (attack / normal) Requires multiclass labels!
EVAL = True  # use models in eval mode
CURRICULUM = 2

WB_TRACKING = False
WAND_RUN_NAME=f"HE_8heads-AC{CURRICULUM}|{MAX_PACKETS_PER_FEAT_TENSOR}-PKT|{MAX_FLOWSTATS_PER_FEAT_TENSOR}TS"
###################################################################


if MULTI_CLASS_CLASSIFICATION:

    if CURRICULUM == 0:
          TRAINING_LABELS_DICT = AC0_TRAINING_LABELS_DICT
          ZDA_DICT = AC0_ZDA_DICT
          TEST_ZDA_DICT = AC0_TEST_ZDA_DICT
    if CURRICULUM == 1:
          TRAINING_LABELS_DICT = AC1_TRAINING_LABELS_DICT
          ZDA_DICT = AC1_ZDA_DICT
          TEST_ZDA_DICT = AC1_TEST_ZDA_DICT
    if CURRICULUM == 2:
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



WANDB_PROJECT_NAME = "StarWars"

WANDB_CONFIG_DICT = {"FLOW_IDLE_TIMEOUT": FLOW_IDLE_TIMEOUT,
                     "ARP_TIMEOUT": ARP_TIMEOUT,
                     "MAX_BUFFERED_PER_IP": MAX_BUFFERED_PER_IP,
                     "MAX_PACKETS_PER_FEAT_TENSOR": MAX_PACKETS_PER_FEAT_TENSOR,
                     "MAX_FLOWSTATS_PER_FEAT_TENSOR": MAX_FLOWSTATS_PER_FEAT_TENSOR,
                     "MAX_BUFFER_TIME": MAX_BUFFER_TIME,
                     "REQUEST_STATS_PERIOD_SECONDS": REQUEST_STATS_PERIOD_SECONDS,
                     "ARP_REQUEST_EXPIRATION_SECONDS": ARP_REQUEST_EXPIRATION_SECONDS,
                     "ANONYMIZE_TRANSPORT_PORTS": ANONYMIZE_TRANSPORT_PORTS,
                     "TRAINING_LABELS_DICT": TRAINING_LABELS_DICT,
                     "AI_DEBUG": AI_DEBUG,
                     "SEED": SEED,
                     "PACKET_FEAT_DIM": PACKET_FEAT_DIM,
                     "FLOW_FEAT_DIM": FLOW_FEAT_DIM,
                     "PACKET_FEATURES": PACKET_FEATURES,
                     "NODE_FEATURES": NODE_FEATURES,
                     "MULTI_CLASS_CLASSIFICATION": MULTI_CLASS_CLASSIFICATION,
                     "BRAIN_DEVICE": BRAIN_DEVICE,
                     "INFERENCE_FREQ_SECONDS": INFERENCE_FREQ_SECONDS,
                     "K_SHOT": K_SHOT,
                     "REPLAY_BUFFER_BATCH_SIZE": REPLAY_BUFFER_BATCH_SIZE
                     }


class Smart_Switch(EventMixin):
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
  def __init__ (self, flow_logger, metrics_logger):
    
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

    self.brain = ControllerBrain(
      eval=EVAL,
      use_packet_feats=PACKET_FEATURES,
      flow_feat_dim=FLOW_FEAT_DIM,
      packet_feat_dim=PACKET_FEAT_DIM,
      multi_class=MULTI_CLASS_CLASSIFICATION, 
      k_shot=K_SHOT,
      replay_buffer_batch_size=REPLAY_BUFFER_BATCH_SIZE,
      kernel_regression=KERNEL_REGRESSION,
      device=BRAIN_DEVICE,
      seed=SEED,
      debug=AI_DEBUG,
      wb_track=WB_TRACKING,
      wb_project_name=WANDB_PROJECT_NAME,
      wb_run_name=WAND_RUN_NAME,
      wb_config_dict=WANDB_CONFIG_DICT)

    # This timer handles expiring stuff 
    # Doesnt seems having to do with time to live stuff
    self._expire_timer = Timer(5, self._handle_expiration, recurring=True)

    # Call the smart check function repeatedly:
    self.smart_check_timer = Timer(INFERENCE_FREQ_SECONDS, self.smart_check, recurring=True)

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
                                            ARP_TIMEOUT=ARP_TIMEOUT)
      
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
                            idle_timeout=FLOW_IDLE_TIMEOUT,
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
    packet_metadata = (time.time() + MAX_BUFFER_TIME, 
                       buffer_id, 
                       port,
                       src_ip)
    packet_metadata_list.append(packet_metadata)
    while len(packet_metadata_list) > MAX_BUFFERED_PER_IP: 
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
    self.recently_sent_ARPs[(switch_id, dest_ip_addr)] = time.time() + ARP_REQUEST_EXPIRATION_SECONDS

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
  Timer(REQUEST_STATS_PERIOD_SECONDS, requests_stats, recurring=True)


def requests_stats():
  for connection in core.openflow._connections.values():
    connection.send(of.ofp_stats_request(body=of.ofp_flow_stats_request()))
    connection.send(of.ofp_stats_request(body=of.ofp_port_stats_request()))
  log.debug("Sent %i flow/port stats request(s)", len(core.openflow._connections))


def launch():
  
  # Registering PacketLogger component:
  flow_logger = FlowLogger(
    training_labels_dict=TRAINING_LABELS_DICT,
    zda_dict=ZDA_DICT,
    test_zda_dict=TEST_ZDA_DICT,
    multi_class=MULTI_CLASS_CLASSIFICATION,
    packet_buffer_len=MAX_PACKETS_PER_FEAT_TENSOR,
    packet_feat_dim=PACKET_FEAT_DIM,
    anonymize_transport_ports=ANONYMIZE_TRANSPORT_PORTS,
    flow_feat_dim=FLOW_FEAT_DIM,
    flow_buff_len=MAX_FLOWSTATS_PER_FEAT_TENSOR)

  metrics_logger=None
  if NODE_FEATURES:
    metrics_logger = MetricsLogger(
      server_addr = "192.168.1.1:9092",
      max_conn_retries = 5,
      metric_buffer_len = 40,
      grafana_user=GRAFANA_USER, 
      grafana_pass=GRAFANA_PASSWORD,
      )
 
     
  # Registering Switch component:
  smart_switch = Smart_Switch(
     flow_logger=flow_logger,
     metrics_logger=metrics_logger
     )
  
  core.register("smart_switch", smart_switch)  
  
  # attach handlers to listeners
  core.openflow.addListenerByName(
    "ConnectionUp", 
    _handle_ConnectionUp)

  core.openflow.addListenerByName(
    "FlowStatsReceived", 
    flow_logger._handle_flowstats_received) 
  
  """
  core.openflow.addListenerByName(
    "PortStatsReceived", 
    flow_logger._handle_portstats_received) 
  """

