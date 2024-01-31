import os
from scapy.all import *
import netifaces as ni
from scapy.all import IP


SOURCE_IP = None
SOURCE_MAC = None
TARGET_IP = None
IFACE_NAME = 'eth0'
PATTERN_TO_REPLAY = None

def get_source_ip_address(interface=IFACE_NAME):
    try:
        ip = ni.ifaddresses(interface)[ni.AF_INET][0]['addr']
        return ip
    except ValueError:
        return "Interface not found"


def get_source_mac(interface=IFACE_NAME):
    try:
        mac_address = ni.ifaddresses(interface)[ni.AF_LINK][0]['addr']
        return mac_address
    except ValueError:
        return "Interface not found"
    

def modify_and_send(packet):

    # focus on level 3
    ip_packet = packet[IP]

    # Modify source IP
    ip_packet.src = SOURCE_IP

    # Modify destination IP
    ip_packet.dst = TARGET_IP

    # Send the modified packet
    send(ip_packet, iface=IFACE_NAME)
   


def resend_pcap_with_modification():
    # Iterate over files in the directory
    for filename in os.listdir(PATTERN_TO_REPLAY):
        if filename.endswith(".pcap"):
            pcap_file = os.path.join(PATTERN_TO_REPLAY, filename)
            print("Processing file:", pcap_file)

            # Read the PCAP file
            packets = rdpcap(pcap_file)

            # Iterate through each packet in the PCAP file
            for packet in packets:
                # Check if the packet contains IP layer
                if IP in packet:
                    modify_and_send(packet)


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: python3 replay.py PATTERN_TO_REPLAY TARGET_IP")
        sys.exit(1)

    # Your new source IP
    SOURCE_IP = get_source_ip_address()
    SOURCE_MAC = get_source_mac()

    PATTERN_TO_REPLAY = sys.argv[1]  # Get the PATTERN_TO_REPLAY DIRECTORY from the command line arguments
    TARGET_IP = sys.argv[2]  # Get the TARGET_IP from the command line arguments

    print(f'Source IP {SOURCE_IP}')
    print(f'Source MAC {SOURCE_MAC}')
    print(f'Target IP {TARGET_IP}')
    print(f'Pattern to replay: {PATTERN_TO_REPLAY}')


    # Resend packets with modified IPs
    resend_pcap_with_modification()