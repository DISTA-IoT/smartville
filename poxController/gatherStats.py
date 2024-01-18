from pox.core import core
import pox.openflow.libopenflow_01 as of
import pox.lib.packet as pkt
from pox.lib.util import dpidToStr

# implemented in l3_learning_mod
log = core.getLogger()

def _handle_PacketIn(event):
    packet = event.parsed

    src_mac = packet.src
    dst_mac = packet.dst
    switch_dpid = dpidToStr(event.dpid)
    in_port = event.port

    log_message = f"Packet received on Switch {switch_dpid}, Port {in_port}:\n"
    log_message += f"  - Source MAC: {src_mac}\n"
    log_message += f"  - Destination MAC: {dst_mac}\n"
    
    # Extract additional information based on the packet type
    if isinstance(packet.next, pkt.ipv4):
        log_message += f"  - IP Source: {packet.next.srcip}\n"
        log_message += f"  - IP Destination: {packet.next.dstip}\n"
    elif isinstance(packet.next, pkt.arp):
        log_message += f"  - ARP Sender IP: {packet.next.protosrc}\n"
        log_message += f"  - ARP Target IP: {packet.next.protodst}\n"

    # Log the information to a file
    with open("packet_logs.txt", "a") as log_file:
        log_file.write(log_message)
        log_file.write("----------------------------------------\n")

def launch():
    core.openflow.addListenerByName("PacketIn", _handle_PacketIn)
    log.info("Packet logger running")
