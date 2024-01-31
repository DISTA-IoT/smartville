FROM ubuntu:focal


RUN     apt-get -y update && \
        apt-get -y upgrade

# dependencies of TCP Replay
RUN     apt-get -y --force-yes install \
        wget curl build-essential tcpdump tcpreplay

RUN apt-get install -y python3 python3-pip netcat wget\
    net-tools iputils-ping  tcpdump && \
    rm -rf /var/lib/apt/lists/*

RUN     apt-get clean && \
        rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


RUN pip3 install --upgrade pip

RUN pip3 install scapy

RUN pip3 install netifaces


WORKDIR /app

COPY scripts/. /app/

COPY  bening_traffic/. /app/bening_traffic/
COPY  cc_heartbeat/. /app/cc_heartbeat/
COPY  generic_ddos/. /app/generic_ddos/
COPY  h_scan/. /app/h_scan/
COPY  muhstik/. /app/muhstik/
COPY  okiru/. /app/okiru/

RUN chmod +x /app/icmp_flood.py

