from pox.core import core
from pox.lib.packet.ipv4 import ipv4
import os


log = core.getLogger()

class PacketLogger(object):
    def __init__(self):
        core.openflow.addListeners(self)
        pcap_folder = os.path.join(os.getcwd(), 'pcap_folder')
        # Check if the pcap folder exists, create it if not
        if not os.path.exists(pcap_folder):
            os.makedirs(pcap_folder)
        self.pcap_folder = pcap_folder
        self.packet_lists = {}  # Initialize the packet_lists dictionary
        self.max_packets_per_port = 10

    def _handle_PacketIn(self, event):

      BLACKLIST_IP={
              "192.168.1.50",
              "192.168.1.51",
              "192.168.1.52",
              "192.168.1.53",
              "192.168.1.54"}
          
      if event.parsed.type == 38:
        #Ignore LLC
        return

      packet= event.parsed
      packet_data = event.data
      port = event.port

      log.info("Received packet on port %s:", port)
      
      #if port is not in the list add the port
      if port not in self.packet_lists:
          self.packet_lists[port] = []

      #if the packet is in the blacklist add it as true, if not as false
      if isinstance(packet.next, ipv4):
        if str(packet.next.srcip) in BLACKLIST_IP:
          log.info(f"IP {packet.next.srcip} in blacklist")
          packet_labeled = (packet_data,True)
          self.packet_lists[port].append(packet_labeled)
        else:
          log.info(f"IP {packet.next.srcip} not in blacklist")
          packet_labeled = (packet_data,False)
          self.packet_lists[port].append(packet_labeled)
      else:
        print("packet not ipv4")
  
      # Check if the list has reached 100 packets
      if len(self.packet_lists[port]) >= self.max_packets_per_port: 
          print("PACKET LIMIT REACHED ON PORT ",port)
          for packet in self.packet_lists[port]:
            print(packet)
          #ai_placeholder(self.packet_lists[port],port)
          # Clear the list for this port
          self.packet_lists[port]= []
          