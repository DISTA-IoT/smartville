import os
from scapy.all import *
import netifaces as ni
from scapy.all import IP
import threading
import argparse


SOURCE_IP = None
SOURCE_MAC = None
TARGET_IP = None
IFACE_NAME = 'eth0'
PATTERN_TO_REPLAY = None
REPEAT_PATTERN_SECS = None

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
            # print("Processing file:", pcap_file)

            # Read the PCAP file
            packets = rdpcap(pcap_file)

            # Iterate through each packet in the PCAP file
            for packet in packets:
                # Check if the packet contains IP layer
                if IP in packet:
                    modify_and_send(packet)



def repeat_function():
    
    resend_pcap_with_modification()  # Execute the function immediately
    
    if REPEAT_PATTERN_SECS is not None:
        print(f'Will repeat pattern in {REPEAT_PATTERN_SECS} seconds')
        threading.Timer(
            REPEAT_PATTERN_SECS, 
            repeat_function).start()
    

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Usage: python3 replay.py PATTERN_TO_REPLAY TARGET_IP")
        sys.exit(1)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Replay script with optional repeat argument")
    parser.add_argument("PATTERN_TO_REPLAY", help="Pattern to replay directory")
    parser.add_argument("TARGET_IP", help="Target IP address")
    parser.add_argument("--repeat", type=int, default=None, help="Number of seconds before repeating the pattern (default: Don't repeat)")

    args = parser.parse_args()

    # Your new source IP
    SOURCE_IP = get_source_ip_address()
    SOURCE_MAC = get_source_mac()

    # Get the values from command line arguments
    PATTERN_TO_REPLAY = args.PATTERN_TO_REPLAY
    TARGET_IP = args.TARGET_IP
    REPEAT_PATTERN_SECS = args.repeat


    print(f'Source IP {SOURCE_IP}')
    print(f'Source MAC {SOURCE_MAC}')
    print(f'Target IP {TARGET_IP}')
    print(f'Pattern to replay: {PATTERN_TO_REPLAY}')
    print(f'Interval between replays: {REPEAT_PATTERN_SECS}')


    # Resend packets with modified IPs
    repeat_function()