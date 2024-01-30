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
from smartController.packetlogger import PacketLogger
from smartController.controller_brain import ControllerBrain

openflow_connection = None #openflow connection to switch is stored here

log = core.getLogger()

# Timeout for flows
FLOW_IDLE_TIMEOUT = 10

# Timeout for ARP entries
ARP_TIMEOUT = 60 * 2

# Maximum number of packet to buffer on a switch for an unknown IP
MAX_BUFFERED_PER_IP = 5

# Maximum time to hang on to a buffer for an unknown IP in seconds
MAX_BUFFER_TIME = 5

# Interval in which the stats request is triggered
REQUEST_STATS_PERIOD_SECONDS = 5

# Don't re-send ARPs before expiring this interval:
ARP_REQUEST_EXPIRATION_SECONDS = 4

# IpV4 attackers (for training purposes) Also victim response flows are considered infected
IPV4_BLACKLIST=["192.168.1.3", "192.168.1.11", "192.168.1.13", "192.168.1.14", "192.168.1.9", "192.168.1.12"]

print(f"HARD TIMEOUT IS SET TO {of.OFP_FLOW_PERMANENT} WHICH IS DEFAULT")

AI_DEBUG = True

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
  def __init__ (self, packetlogger):

    # We use this to prevent ARP flooding
    # Key: (switch_id, ARPed_IP) Values: ARP request expire time
    self.recently_sent_ARPs = {}


    # (dpid,IP) -> [(expire_time,buffer_id,in_port), ...]
    # These are buffers we've gotten at this switch_port for this IP which
    # we can't deliver because we don't know where they go.
    self.lost_buffers = {}

    # For each switch, we map destination IP addresses to Entries
    # (Entries are pairs of switch output ports and MAC addresses)
    self.arpTables = {}

    self.brain = ControllerBrain(log, AI_DEBUG)

    # This timer handles expiring stuff 
    # Doesnt seems having to do with time to live stuff
    self._expire_timer = Timer(5, self._handle_expiration, recurring=True)

    # Call the smart check function repeatedly:
    self.smart_check_timer = Timer(5, self.smart_check, recurring=True)

    # Our packetlogger instance:
    self.packetlogger = packetlogger

    core.listen_to_dependencies(self)


  def smart_check(self):
      self.brain.classify_duet(
        flows=list(self.packetlogger.flows_dict.values()))
    # self.packetlogger.reset_packet_lists()
    # self.packetlogger.reset_all_portstats_lists()
    # self.packetlogger.reset_all_flows_metadata()
    # if AI_DEBUG: log.info('cleaned packetlogger info!')


  def _handle_expiration(self):
    # Called by a timer so that we can remove old items.
    empty = []
    for k,v in self.lost_buffers.items():
      dpid,ip = k

      for item in list(v):
        expires_at, buffer_id, in_port = item
        if expires_at < time.time():
          # This packet is old.  Tell this switch to drop it.
          v.remove(item)
          po = of.ofp_packet_out(buffer_id = buffer_id, in_port = in_port)
          core.openflow.sendToDPID(dpid, po)
      if len(v) == 0: empty.append(k)

    # Remove empty buffer bins
    for k in empty:
      del self.lost_buffers[k]


  def _send_lost_buffers(self, switch_id, port, dest_mac_addr, dest_ip_addr):
    """
    We may have "lost" buffers -- packets we got but didn't know
    where to send at the time.  We may know now.  Try and see.
    """
    
    if (switch_id, dest_ip_addr) in self.lost_buffers:
      
      bucket = self.lost_buffers[(switch_id, dest_ip_addr)]
      del self.lost_buffers[(switch_id, dest_ip_addr)]

      log.debug("Sending %i buffered packets to %s" % (len(bucket), dest_ip_addr))
      
      for _, buffer_id, in_port in bucket:

        po = of.ofp_packet_out(buffer_id=buffer_id, in_port=in_port)
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
                                 buffer_id,
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
                            buffer_id=buffer_id,
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


  def handle_unknown_ip_packet(self, switch_id, incomming_port, packet_in_event):
    """
    First, track this buffer so that we can try to resend it later, when we will learn the destination.
    Second, ARP for the destination, which should ultimately result in it responding and us learning where it is
    """

    packet = packet_in_event.parsed
    source_mac_addr = packet.src
    source_ip_addr = packet.next.srcip
    dest_ip_addr = packet.next.dstip
    
    # Add to tracked buffers
    if (switch_id, dest_ip_addr) not in self.lost_buffers:
        self.lost_buffers[(switch_id, dest_ip_addr)] = []

    bucket = self.lost_buffers[(switch_id, dest_ip_addr)]
    entry = (time.time() + MAX_BUFFER_TIME, packet_in_event.ofp.buffer_id, incomming_port)
    bucket.append(entry)

    while len(bucket) > MAX_BUFFERED_PER_IP: del bucket[0]


    # Expire things from our recently_sent_ARP list...
    self.recently_sent_ARPs = {k:v for k, v in self.recently_sent_ARPs.items() if v > time.time()}

    # Check if we've already ARPed recently
    if (switch_id, dest_ip_addr) in self.recently_sent_ARPs:
      # Oop, we've already done this one recently.
      return

    # And ARP...
    self.recently_sent_ARPs[(switch_id, dest_ip_addr)] = time.time() + ARP_REQUEST_EXPIRATION_SECONDS

    self.build_and_send_ARP_request(
        switch_id, 
        incomming_port,
        source_mac_addr,
        source_ip_addr,
        dest_ip_addr,
        connection=packet_in_event.connection)
    
  
  def try_forwarding_packet(self, switch_id,incomming_port, packet_in_event):
      
      packet = packet_in_event.parsed
      source_ip_addr = packet.next.srcip
      dest_ip_addr = packet.next.dstip

      if dest_ip_addr in self.arpTables[switch_id]:
          # destination address is present in the arp table
          # get mac and out port
          outgoing_port = self.arpTables[switch_id][dest_ip_addr].port
          dest_mac_addr = self.arpTables[switch_id][dest_ip_addr].mac

          if outgoing_port != incomming_port:

              self.add_ip_to_ip_flow_matching_rule(
                                switch_id,
                                source_ip_addr, 
                                dest_ip_addr, 
                                dest_mac_addr, 
                                outgoing_port,
                                connection=packet_in_event.connection,
                                buffer_id=packet_in_event.ofp.buffer_id,
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
      
      
      # Send any waiting packets for that ip
      self._send_lost_buffers(
         switch_id, 
         incomming_port, 
         dest_mac_addr=packet.src,
         dest_ip_addr=packet.next.srcip)


      self.learn_or_update_arp_table(ip_addr=packet.next.srcip,
                                     mac_addr=packet.src,
                                     port=incomming_port, 
                                     connection=packet_in_event.connection)

      self.try_forwarding_packet(switch_id, 
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
          self._send_lost_buffers(
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
  packetLogger = PacketLogger(
    ipv4_blacklist_for_training=IPV4_BLACKLIST)

  core.openflow.addListeners(packetLogger)
  core.register("PacketLogger", packetLogger)

  # Registering Switch component:
  smart_switch = Smart_Switch(packetLogger)
  core.register("smart_switch", smart_switch)  
  
  # attach handlers to listeners
  core.openflow.addListenerByName(
    "ConnectionUp", 
    _handle_ConnectionUp)

  core.openflow.addListenerByName(
    "FlowStatsReceived", 
    packetLogger._handle_flowstats_received) 
  
  core.openflow.addListenerByName(
    "PortStatsReceived", 
    packetLogger._handle_portstats_received) 
  