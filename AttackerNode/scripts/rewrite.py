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
from scapy.all import *
from scapy.all import IP
import argparse
from tqdm import tqdm

SOURCE_IP = None
TARGET_IP = None
PATTERN_TO_REWRITE = None


def modify_and_save_pcap(input_pcap_file, output_pcap_file):
    # Read the PCAP file
    packets = rdpcap(input_pcap_file)

    # Modify source and destination IP addresses of each packet
    for packet in tqdm(packets):
        if IP in packet:
            packet[IP].src = SOURCE_IP
            packet[IP].dst = TARGET_IP

    # Save the modified packets to another PCAP file
    wrpcap(output_pcap_file, packets)


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Usage: python3 rewriter.py PATTERN_TO_REWRITE SOURCE_IP TARGET_IP")
        sys.exit(1)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Replay script with optional repeat argument")
    parser.add_argument("PATTERN_TO_REWRITE", help="Pattern to replay directory")
    parser.add_argument("SOURCE_IP", help="Source IP address")
    parser.add_argument("TARGET_IP", help="Target IP address")

    args = parser.parse_args()

    # Your new source IP
    SOURCE_IP = args.SOURCE_IP
    PATTERN_TO_REWRITE = args.PATTERN_TO_REWRITE
    TARGET_IP = args.TARGET_IP

    print(f'Source IP {SOURCE_IP}')
    print(f'Target IP {TARGET_IP}')
    print(f'Pattern to rewrite: {PATTERN_TO_REWRITE}')
    input_file_name = PATTERN_TO_REWRITE+'/'+PATTERN_TO_REWRITE+'.pcap'
    output_file_name = PATTERN_TO_REWRITE+'/'+PATTERN_TO_REWRITE+'-from'+SOURCE_IP+'to'+TARGET_IP+'.pcap'
    # Resend packets with modified IPs
    modify_and_save_pcap(input_file_name, output_file_name)