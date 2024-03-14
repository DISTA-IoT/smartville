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
import socket
import random
import time
from scapy.all import IP, ICMP, TCP, UDP, send

def send_icmp(target_ip):
    # Send ICMP (ping) message
    packet = IP(dst=target_ip) / ICMP()
    send(packet, verbose=False)

def send_tcp(target_ip):
    # Send TCP message
    dest_port = random.randint(1024, 65535)
    packet = IP(dst=target_ip) / TCP(dport=dest_port)
    send(packet, verbose=False)

def send_udp(target_ip):
    # Send UDP message
    dest_port = random.randint(1024, 65535)
    packet = IP(dst=target_ip) / UDP(dport=dest_port)
    send(packet, verbose=False)

def main():
    target_ip = input("Enter the target IP address: ")

    while True:
        # Generate a random interval between 1 and 10 seconds
        random_interval = random.randint(1, 10)
        
        # Randomly select a message type (ICMP, TCP, or UDP)
        message_type = random.choice(["icmp", "tcp", "udp"])

        # Send the corresponding message
        if message_type == "icmp":
            send_icmp(target_ip)
            print(f"Sent ICMP message to {target_ip}")
        elif message_type == "tcp":
            send_tcp(target_ip)
            print(f"Sent TCP message to {target_ip}")
        elif message_type == "udp":
            send_udp(target_ip)
            print(f"Sent UDP message to {target_ip}")

        # Wait for the next random interval
        time.sleep(random_interval)

if __name__ == "__main__":
    main()
