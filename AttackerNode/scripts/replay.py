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
PREPROCESSED = None

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
   

def modify_and_save_pcap(input_pcap_file, output_pcap_file):
    # Read the PCAP file
    packets = rdpcap(input_pcap_file)

    # Modify source and destination IP addresses of each packet
    for packet in packets:
        if IP in packet:
            packet[IP].src = SOURCE_IP
            packet[IP].dst = TARGET_IP

    # Save the modified packets to another PCAP file
    wrpcap(output_pcap_file, packets)


def resend_pcap_with_modification():
    # Iterate over files in the directory
    for filename in os.listdir(PATTERN_TO_REPLAY):
        if filename.endswith(".pcap"):
            pcap_file = os.path.join(PATTERN_TO_REPLAY, filename)
            # print("Processing file:", pcap_file)

            # Read the PCAP file
            packets = rdpcap(pcap_file)
            timestamps = [packet.time for packet in packets]
            time_diffs = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]

            for packet, time_diff in zip(packets, time_diffs):
                if IP in packet:
                    modify_and_send(packet)
                time.sleep(time_diff)


def resend_pcap_with_modification_tcpreplay():

    # Iterate over files in the directory
    for filename in os.listdir(PATTERN_TO_REPLAY):
        if filename.endswith(".pcap"):
            # print("Processing file:", filename)
            pcap_file = os.path.join(PATTERN_TO_REPLAY, filename)
            if not PREPROCESSED:
                print('processing...')
                # Modify and send packets using tcpreplay
                modify_and_save_pcap(pcap_file, 'output_file.pcap')
                print('sending...')
                # Use tcpreplay command to send the modified packets
                cmd = f"tcpreplay -i {IFACE_NAME} output_file.pcap"
            else:
                print('sending...')
                # Use tcpreplay command to send the modified packets
                cmd = f"tcpreplay -i {IFACE_NAME} {PATTERN_TO_REPLAY}/{PATTERN_TO_REPLAY}-from{SOURCE_IP}to{TARGET_IP}.pcap"
            subprocess.run(cmd, shell=True)


def repeat_function():
    
    resend_pcap_with_modification_tcpreplay()  # Execute the function immediately
    
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
    parser.add_argument("--preprocessed", type=int, default=True, help="Search for preprocessed pcap file instead of processing it. (Default: true)")

    args = parser.parse_args()

    # Your new source IP
    SOURCE_IP = get_source_ip_address()
    SOURCE_MAC = get_source_mac()

    # Get the values from command line arguments
    PATTERN_TO_REPLAY = args.PATTERN_TO_REPLAY
    TARGET_IP = args.TARGET_IP
    REPEAT_PATTERN_SECS = args.repeat
    PREPROCESSED = args.preprocessed


    print(f'Source IP {SOURCE_IP}')
    print(f'Source MAC {SOURCE_MAC}')
    print(f'Target IP {TARGET_IP}')
    print(f'Pattern to replay: {PATTERN_TO_REPLAY}')
    print(f'Interval between replays: {REPEAT_PATTERN_SECS}')
    print(f'Preprocessed Pcap Replay: {PREPROCESSED}')


    # Resend packets with modified IPs
    repeat_function()