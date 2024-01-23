# Copyright 2012-2013 James McCauley
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
A stupid L3 switch

For each switch:
1) Keep a table that maps IP addresses to MAC addresses and switch ports.
   Stock this table using information from ARP and IP packets.
2) When you see an ARP query, try to answer it using information in the table
   from step 1.  If the info in the table is old, just flood the query.
3) Flood all other ARPs.
4) When you see an IP packet, if you know the destination port (because it's
   in the table from step 1), install a flow for it.
"""

import datetime
import random
from pox.core import core
import pox
from pox.lib.packet.ethernet import ethernet, ETHER_BROADCAST
from pox.lib.packet.ipv4 import ipv4
from pox.lib.packet.arp import arp
from pox.lib.addresses import IPAddr, EthAddr
from pox.lib.util import str_to_bool, dpid_to_str
from pox.lib.recoco import Timer
import pox.openflow.libopenflow_01 as of
import pox.lib.packet as pkt
import logging
from pox.lib.revent import *
import time
import logging
import os
from scapy.all import wrpcap
from pox.openflow.discovery import graph

# Timeout for flows
FLOW_IDLE_TIMEOUT = 10

# Timeout for ARP entries
ARP_TIMEOUT = 60 * 2

# Maximum number of packet to buffer on a switch for an unknown IP
MAX_BUFFERED_PER_IP = 5

# Maximum time to hang on to a buffer for an unknown IP in seconds
MAX_BUFFER_TIME = 5



log = core.getLogger()
log_file_path = os.path.join(os.getcwd(), 'packet_log.txt')
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(message)s')
print("path: ",log_file_path)
class Entry (object):
  """
  Not strictly an ARP entry.
  We use the port to determine which port to forward traffic out of.
  We use the MAC to answer ARP replies.
  We use the timeout so that if an entry is older than ARP_TIMEOUT, we
   flood the ARP request rather than try to answer it ourselves.
  """
  def __init__ (self, port, mac):
    self.timeout = time.time() + ARP_TIMEOUT
    self.port = port
    self.mac = mac

  def __eq__ (self, other):
    if type(other) == tuple:
      return (self.port,self.mac)==other
    else:
      return (self.port,self.mac)==(other.port,other.mac)
  def __ne__ (self, other):
    return not self.__eq__(other)

  def isExpired (self):
    if self.port == of.OFPP_NONE: return False
    return time.time() > self.timeout


def dpid_to_mac (dpid):
  return EthAddr("%012x" % (dpid & 0xffFFffFFffFF,))

active_topology = {}  # Initialize active topology dictionary
def refresh_topology(ip,port):

    if port in active_topology:
        if ip in active_topology[port]:
            print(f"key: {ip} value: {active_topology[port]} is already present in the topology")
        else:
            active_topology[port].append(ip)
            print(f"key: {ip} value: {active_topology[port]} added in the topology")
    else:
        active_topology[port] = [ip]
        print(f"key: {ip} value: {active_topology[port]} added in the topology")


def print_topology():
  print("---Printing topology---")
  for port, ip_list in active_topology.items():
    print(f"PORT: {port}, IP: {str(ip_list)}")


class PacketLogger(object):
    def __init__(self):
        core.openflow.addListeners(self)
        pcap_folder = os.path.join(os.getcwd(), 'pcap_folder')
        # Check if the pcap folder exists, create it if not
        if not os.path.exists(pcap_folder):
            os.makedirs(pcap_folder)
        self.pcap_folder = pcap_folder
        self.packet_lists = {}  # Initialize the packet_lists dictionary
       
        self.max_packets_per_port = 5

    def _handle_PacketIn(self, event):
        if event.parsed.type == 38:
          #Ignore LLC packets
          return

        packet = event.data
        port = event.port

        log.info("Received packet on port %s:", port)
        log.info(f"type: {event.parse}")
        
        # Add the packet to the list for the corresponding port
        if port not in self.packet_lists:
            self.packet_lists[port] = []

        self.packet_lists[port].append(packet)

        # Check if the list has reached 100 packets
        if len(self.packet_lists[port]) >= self.max_packets_per_port:
            """
            # Write the list of packets to a pcap file
            pcap_filename = f"{self.pcap_folder}/port_{port}_packets.pcap"
            wrpcap(pcap_filename, self.packet_lists[port])
            log.info("Packets written to pcap file: %s", pcap_filename)
            #send list of packet to ai placeholder
            ai_placeholder(self.packet_lists[port],port)
            # Clear the list for this port
            
            """
            
            print_topology()


class Entry (object):
  """
  Not strictly an ARP entry.
  We use the port to determine which port to forward traffic out of.
  We use the MAC to answer ARP replies.
  We use the timeout so that if an entry is older than ARP_TIMEOUT, we
   flood the ARP request rather than try to answer it ourselves.
  """
  def __init__ (self, port, mac):
    self.timeout = time.time() + ARP_TIMEOUT
    self.port = port
    self.mac = mac

  def __eq__ (self, other):
    if type(other) == tuple:
      return (self.port,self.mac)==other
    else:
      return (self.port,self.mac)==(other.port,other.mac)
  def __ne__ (self, other):
    return not self.__eq__(other)

  def isExpired (self):
    if self.port == of.OFPP_NONE: return False
    return time.time() > self.timeout


def dpid_to_mac (dpid):
  return EthAddr("%012x" % (dpid & 0xffFFffFFffFF,))


class l3_switch (EventMixin):
  def __init__ (self, fakeways = [], arp_for_unknowns = False, wide = False):
    # These are "fake gateways" -- we'll answer ARPs for them with MAC
    # of the switch they're connected to.
    self.fakeways = set(fakeways)

    # If True, we create "wide" matches.  Otherwise, we create "narrow"
    # (exact) matches.
    self.wide = wide

    # If this is true and we see a packet for an unknown
    # host, we'll ARP for it.
    self.arp_for_unknowns = arp_for_unknowns

    # (dpid,IP) -> expire_time
    # We use this to keep from spamming ARPs
    self.outstanding_arps = {}

    # (dpid,IP) -> [(expire_time,buffer_id,in_port), ...]
    # These are buffers we've gotten at this datapath for this IP which
    # we can't deliver because we don't know where they go.
    self.lost_buffers = {}

    # For each switch, we map IP addresses to Entries
    self.arpTable = {}

    # This timer handles expiring stuff
    self._expire_timer = Timer(5, self._handle_expiration, recurring=True)

    core.listen_to_dependencies(self)

  def _log_packet_info(self, event, packet_type):
      packet_info = f"Timestamp: {time.time()}, Sender: {event.connection.dpid}, Receiver: {event.port}, Packet Type: {packet_type}"
      log.info(packet_info)
      logging.info(packet_info)

  def _handle_expiration (self):
    # Called by a timer so that we can remove old items.
    empty = []
    for k,v in self.lost_buffers.items():
      dpid,ip = k

      for item in list(v):
        expires_at,buffer_id,in_port = item
        if expires_at < time.time():
          # This packet is old.  Tell this switch to drop it.
          v.remove(item)
          po = of.ofp_packet_out(buffer_id = buffer_id, in_port = in_port)
          core.openflow.sendToDPID(dpid, po)
      if len(v) == 0: empty.append(k)

    # Remove empty buffer bins
    for k in empty:
      del self.lost_buffers[k]

  def _send_lost_buffers (self, dpid, ipaddr, macaddr, port):
    """
    We may have "lost" buffers -- packets we got but didn't know
    where to send at the time.  We may know now.  Try and see.
    """
    if (dpid,ipaddr) in self.lost_buffers:
      # Yup!
      bucket = self.lost_buffers[(dpid,ipaddr)]
      del self.lost_buffers[(dpid,ipaddr)]
      log.debug("Sending %i buffered packets to %s from %s"
                % (len(bucket),ipaddr,dpid_to_str(dpid)))
      for _,buffer_id,in_port in bucket:
        po = of.ofp_packet_out(buffer_id=buffer_id,in_port=in_port)
        po.actions.append(of.ofp_action_dl_addr.set_dst(macaddr))
        po.actions.append(of.ofp_action_output(port = port))
        core.openflow.sendToDPID(dpid, po)

  def _handle_openflow_PacketIn (self, event):
    dpid = event.connection.dpid
    inport = event.port
    packet = event.parsed

    if not packet.parsed:
      log.warning("%i %i ignoring unparsed packet", dpid, inport)
      
      return
  
    if dpid not in self.arpTable:
      # New switch -- create an empty table
      log.info("New switch detected - creating empty flow table with id: %s",dpid)
      self.arpTable[dpid] = {}
      for fake in self.fakeways:
        self.arpTable[dpid][IPAddr(fake)] = Entry(of.OFPP_NONE,
         dpid_to_mac(dpid))

    #if packet.type == ethernet.LLDP_TYPE:
      # Ignore LLDP packets
    #  return

  
    
    

    if isinstance(packet.next, ipv4):
      log.debug("IPV4 DETECTED - SWITCH: %i ON PORT: %i IP SENDER: %s IP RECEIVER %s", dpid,inport,
                packet.next.srcip,packet.next.dstip)

      # Send any waiting packets for that ip
      self._send_lost_buffers(dpid, packet.next.srcip, packet.src, inport)

      # Learn or update port/MAC info
      if packet.next.srcip in self.arpTable[dpid]:
        if self.arpTable[dpid][packet.next.srcip] != (inport, packet.src):
          log.info("%i %i RE-learned %s", dpid,inport,packet.next.srcip)
          if self.wide:
            # Make sure we don't have any entries with the old info...
            msg = of.ofp_flow_mod(command=of.OFPFC_DELETE)
            msg.match.nw_dst = packet.next.srcip
            msg.match.dl_type = ethernet.IP_TYPE
            event.connection.send(msg)
      else:
        log.debug(f"ADD TO ARP TABLE NEW ENTRY (SENDER) - PORT:{inport} IP:{packet.next.srcip}")
        self.arpTable[dpid][packet.next.srcip] = Entry(inport, packet.src)
        #add sender to topology
        refresh_topology(packet.next.srcip,inport)

      # Try to forward
      dstaddr = packet.next.dstip
      if dstaddr in self.arpTable[dpid]:
        # destination address is present in the arp table
        # get mac and out port
        prt = self.arpTable[dpid][dstaddr].port
        mac = self.arpTable[dpid][dstaddr].mac
        if prt == inport:
          log.warning("not sending packet out of in port")
        else:
          log.debug(f"ADD NEW FLOW RULE TO:{dpid} - IN PORT:{inport} SENDER IP: {packet.next.srcip} TO RECEIVER IP:{dstaddr} OUT PORT: {prt}")
          #add dest. to topology
          refresh_topology(dstaddr,prt)
          #prepare the flow rule and send it to the switch
          actions = []
          actions.append(of.ofp_action_dl_addr.set_dst(mac))
          actions.append(of.ofp_action_output(port = prt))
          if self.wide:
            match = of.ofp_match(dl_type = packet.type, nw_dst = dstaddr)
          else:
            match = of.ofp_match.from_packet(packet, inport)

          msg = of.ofp_flow_mod(command=of.OFPFC_ADD,
                                idle_timeout=FLOW_IDLE_TIMEOUT,
                                hard_timeout=of.OFP_FLOW_PERMANENT,
                                buffer_id=event.ofp.buffer_id,
                                actions=actions,
                                match=match)
          event.connection.send(msg.pack())

      elif self.arp_for_unknowns:
        # We don't know this destination.
        # First, we track this buffer so that we can try to resend it later
        # if we learn the destination, second we ARP for the destination,
        # which should ultimately result in it responding and us learning
        # where it is

        # Add to tracked buffers
        if (dpid,dstaddr) not in self.lost_buffers:
          self.lost_buffers[(dpid,dstaddr)] = []
        bucket = self.lost_buffers[(dpid,dstaddr)]
        entry = (time.time() + MAX_BUFFER_TIME,event.ofp.buffer_id,inport)
        bucket.append(entry)
        while len(bucket) > MAX_BUFFERED_PER_IP: del bucket[0]

        # Expire things from our outstanding ARP list...
        self.outstanding_arps = {k:v for k,v in
         self.outstanding_arps.items() if v > time.time()}

        # Check if we've already ARPed recently
        if (dpid,dstaddr) in self.outstanding_arps:
          # Oop, we've already done this one recently.
          return

        # And ARP...
        self.outstanding_arps[(dpid,dstaddr)] = time.time() + 4

        r = arp()
        r.hwtype = r.HW_TYPE_ETHERNET
        r.prototype = r.PROTO_TYPE_IP
        r.hwlen = 6
        r.protolen = r.protolen
        r.opcode = r.REQUEST
        r.hwdst = ETHER_BROADCAST
        r.protodst = dstaddr
        r.hwsrc = packet.src
        r.protosrc = packet.next.srcip
        e = ethernet(type=ethernet.ARP_TYPE, src=packet.src,
                     dst=ETHER_BROADCAST)
        e.set_payload(r)
        log.debug("%i %i ARPing for %s on behalf of %s" % (dpid, inport,
         r.protodst, r.protosrc))
        msg = of.ofp_packet_out()
        msg.data = e.pack()
        msg.actions.append(of.ofp_action_output(port = of.OFPP_FLOOD))
        msg.in_port = inport
        event.connection.send(msg)

    elif isinstance(packet.next, arp):
      a = packet.next

      log.debug(f"SWITCH: {dpid} IN PORT:{inport} ARP FROM: {a.protosrc} TO:{a.protodst} => {a.protodst} ({'request' if a.opcode == arp.REQUEST else 'reply' if a.opcode == arp.REPLY else 'op:%i' % a.opcode})")


      if a.prototype == arp.PROTO_TYPE_IP:
        if a.hwtype == arp.HW_TYPE_ETHERNET:
          if a.protosrc != 0:

            # Learn or update port/MAC info
            if a.protosrc in self.arpTable[dpid]:
              if self.arpTable[dpid][a.protosrc] != (inport, packet.src):
                log.info("%i %i RE-learned %s", dpid,inport,a.protosrc)
                if self.wide:
                  # Make sure we don't have any entries with the old info...
                  msg = of.ofp_flow_mod(command=of.OFPFC_DELETE)
                  msg.match.dl_type = ethernet.IP_TYPE
                  msg.match.nw_dst = a.protosrc
                  event.connection.send(msg)
            else:
              log.debug(f"ADD TO ARP TABLE NEW ENTRY (SRC): IN PORT:{inport} IP:{a.protosrc}")
              self.arpTable[dpid][a.protosrc] = Entry(inport, packet.src)

            # Send any waiting packets...
            self._send_lost_buffers(dpid, a.protosrc, packet.src, inport)

            if a.opcode == arp.REQUEST:
              # If an arp request is received

              if a.protodst in self.arpTable[dpid]:
                # and the destination address is in the arp table

                if not self.arpTable[dpid][a.protodst].isExpired():
                  # and it is not expired, answer the ARP

                  r = arp()
                  r.hwtype = a.hwtype
                  r.prototype = a.prototype
                  r.hwlen = a.hwlen
                  r.protolen = a.protolen
                  r.opcode = arp.REPLY
                  r.hwdst = a.hwsrc
                  r.protodst = a.protosrc
                  r.protosrc = a.protodst
                  r.hwsrc = self.arpTable[dpid][a.protodst].mac
                  e = ethernet(type=packet.type, src=dpid_to_mac(dpid),
                               dst=a.hwsrc)
                  e.set_payload(r)
                  log.debug(f"ARP ANSWER - {dpid} ADDRESS:{r.protosrc}  ON PORT{inport} ")

                  msg = of.ofp_packet_out()
                  msg.data = e.pack()
                  msg.actions.append(of.ofp_action_output(port =
                                                          of.OFPP_IN_PORT))
                  msg.in_port = inport
                  event.connection.send(msg)
                  return

      # Didn't know how to answer or otherwise handle this ARP, so just flood it
      log.debug(f"FLOODING ARPS: {dpid} IN PORT: {inport} SENDER ADDRESS:{a.protosrc} RECEIVER ADDRESS:{a.protodst} => {a.protodst} ({'request' if a.opcode == arp.REQUEST else 'reply' if a.opcode == arp.REPLY else 'op:%i' % a.opcode})")


      msg = of.ofp_packet_out(in_port = inport, data = event.ofp,
          action = of.ofp_action_output(port = of.OFPP_FLOOD))
      event.connection.send(msg)

def ai_placeholder(packetlist,port):
  log.info(f"AI: RECEIVED {len(packetlist)} PACKETS FOR PORT: {port}")
  choose = random.choice([True, False])
  log.info(f"AI: I HAVE CHOSEN: {choose} FOR PORT: {port}")
  if choose:
    mitigate_attack(port)

def mitigate_attack(port):
  block_traffic(port)

def block_traffic(port):
  # Creating a flow rule to drop all packets coming in on the specified port
  msg = of.ofp_flow_mod()
  msg.match.in_port = port  # Replace with the desired input port
  msg.idle_timeout = 0  # Set to 0 for no idle timeout
  msg.hard_timeout = 0  # Set to 0 for no hard timeout
  openflow_connection.send(msg)
  log.info(f"SWITCH FLOW MOD SENT - BLOCKED PORT {port}")


openflow_connection = None

def _handle_ConnectionUp (event):
  global openflow_connection
  openflow_connection=event.connection
  log.info("Connection is UP")



def launch (fakeways="", arp_for_unknowns=None, wide=False):
  core.openflow.addListenerByName("ConnectionUp", _handle_ConnectionUp)
  fakeways = fakeways.replace(","," ").split()
  fakeways = [IPAddr(x) for x in fakeways]
  if arp_for_unknowns is None:
    arp_for_unknowns = len(fakeways) > 0
  else:
    arp_for_unknowns = str_to_bool(arp_for_unknowns)
  core.registerNew(l3_switch, fakeways, arp_for_unknowns, wide)
  core.registerNew(PacketLogger)
   

