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

RUN pip3 install scapy confluent_Kafka netifaces psutil tqdm

WORKDIR /app

COPY scripts/. /app/
COPY  doorlock/. /app/doorlock/
COPY  echo/. /app/echo/
COPY  hue/. /app/hue/

RUN apt update
RUN apt install git -y
RUN git clone https://github.com/QwertyJacob/byob