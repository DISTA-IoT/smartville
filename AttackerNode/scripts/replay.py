# This file is part of the "Smartville" project.
# Copyright (c) 2024 University of Insubria
# Licensed under the Apache License 2.0.
# SPDX-License-Identifier: Apache-2.0
# For the full text of the license, visit:
# https://www.apache.org/licenses/LICENSE-2.0

# Smartville is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# Apache License 2.0 for more details.

# You should have received a copy of the Apache License 2.0
# along with Smartville. If not, see <https://www.apache.org/licenses/LICENSE-2.0>.

# Additional licensing information for third-party dependencies
# used in this file can be found in the accompanying `NOTICE` file.
import os
from scapy.all import *
import netifaces as ni
from scapy.all import IP
import threading
import argparse
from tqdm import tqdm


SOURCE_IP = None
SOURCE_MAC = None
TARGET_IP = None
IFACE_NAME = 'eth0'
PATTERN_TO_REPLAY = None
REPEAT_PATTERN_SECS = None
PREPROCESSED = None

def get_static_source_ip_address(interface=IFACE_NAME):
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
    

def modify_and_save_pcap(input_pcap_file, output_pcap_file):
    # Read the PCAP file
    print(f'Opening {input_pcap_file} file, please wait...')
    packets = rdpcap(input_pcap_file)
    print('File opened!')
    print(f'Now rewritting packets with source {SOURCE_IP} and dest {TARGET_IP}')
    # Modify source and destination IP addresses of each packet
    for packet in tqdm(packets):
        if IP in packet:
            packet[IP].src = SOURCE_IP
            packet[IP].dst = TARGET_IP
    print(f'Packets re-written. NOW SAVING, please wait...')
    # Save the modified packets to another PCAP file
    wrpcap(output_pcap_file, packets)
    print(f'File saved! ready to go!!')


def resend_pcap_with_modification_tcpreplay():

    original_pcap_file = os.path.join(f"{PATTERN_TO_REPLAY}/{PATTERN_TO_REPLAY}.pcap")
    file_to_replay = f"{PATTERN_TO_REPLAY}/{PATTERN_TO_REPLAY}-from{SOURCE_IP}to{TARGET_IP}.pcap"
            
    if not os.path.exists(file_to_replay):
        print(f'FILE NOT FOUND: {file_to_replay}')
        print("Rewriting pattern with new addressses...")
        # Modify and send packets using tcpreplay
        modify_and_save_pcap(original_pcap_file, file_to_replay)
    else:
        print(f'REWRITEN {PATTERN_TO_REPLAY} PATTERN FOUND from {SOURCE_IP} to {TARGET_IP}')

    print('sending...')
    # Tcpreplay command to send the modified packets
    cmd = f"tcpreplay -i {IFACE_NAME}  --stats 3 {file_to_replay}"
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
    parser.add_argument("PATTERN_TO_REPLAY", help="Pattern to replay")
    parser.add_argument("TARGET_IP", help="Target IP address")
    parser.add_argument("--repeat", type=int, default=5, help="Number of seconds before repeating the pattern (default: 5)")

    args = parser.parse_args()

    # Your new source IP
    SOURCE_IP = get_static_source_ip_address()
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