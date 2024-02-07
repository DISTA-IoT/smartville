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
    input_file_name = 'VictimNode/'+PATTERN_TO_REWRITE+'/'+PATTERN_TO_REWRITE
    # Resend packets with modified IPs
    for i in range(1,4):
        output_file_name = 'VictimNode/'+PATTERN_TO_REWRITE+'/'+PATTERN_TO_REWRITE+'_'+str(i)+'-from'+SOURCE_IP+'to'+TARGET_IP
        modify_and_save_pcap(input_file_name+'_'+str(i)+'.pcap', output_file_name+'.pcap')