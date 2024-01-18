FROM ubuntu:focal

RUN apt-get update && \
    apt-get install -y python3 python3-pip netcat wget\
    net-tools iputils-ping  tcpdump && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

RUN pip3 install scapy

WORKDIR /app

COPY icmp_flood.py /app/icmp_flood.py
COPY msg_send_sim.py /app/msg_send_sim.py
COPY wget_send_sim.py /app/wget_send_sim.py

RUN chmod +x /app/icmp_flood.py
RUN chmod +x /app/msg_send_sim.py
RUN chmod +x /app/wget_send_sim.py


