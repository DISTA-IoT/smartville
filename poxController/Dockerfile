FROM python:3.8


RUN apt-get update && \
    apt-get install -y git nano vim python3-pip \
    libpcap0.8 libpcap0.8-dev wget\
    net-tools iputils-ping  tcpdump && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

RUN pip3 install scapy networkx pydot numpy scikit-learn wandb confluent_Kafka

RUN pip3 install torch --index-url https://download.pytorch.org/whl/cpu

# For AI logging purposes (can comment if u dont need it...)
COPY jesus_wandb_api_key.txt /root/.netrc

RUN git clone https://github.com/noxrepo/pox.git /pox

# Install Kafka
RUN apt-get update && \
    apt-get install -y default-jre default-jdk && \
    rm -rf /var/lib/apt/lists/*

# Download Kafka binaries
RUN wget https://dlcdn.apache.org/kafka/3.6.1/kafka_2.13-3.6.1.tgz && \
    tar -xzf kafka_2.13-3.6.1.tgz && \
    rm kafka_2.13-3.6.1.tgz && \
    mv kafka_2.13-3.6.1 /opt/kafka

COPY kafka_server.properties /pox/kafka_server.properties
COPY zookeeper.properties /pox/zookeeper.properties

# Expose Kafka port
EXPOSE 9092

# Set environment variables
ENV PATH="/opt/kafka/bin:${PATH}"


WORKDIR /pox

RUN git checkout gar-experimental 

RUN rm -r .git

COPY entrypoint.sh /pox/entrypoint.sh

COPY  smartController/. /pox/pox/smartController/

RUN chmod -R 777 /pox/

ENTRYPOINT ["/pox/entrypoint.sh"]