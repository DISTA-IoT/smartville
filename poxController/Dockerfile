FROM python:3.8


RUN apt-get update && \
    apt-get install -y git nano vim python3-pip \
    libpcap0.8 libpcap0.8-dev wget\
    net-tools iputils-ping  tcpdump && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

RUN pip3 install scapy networkx pydot numpy scikit-learn wandb

RUN pip3 install torch --index-url https://download.pytorch.org/whl/cpu

# For AI logging purposes (can comment if u dont need it...)
COPY jesus_wandb_api_key.txt /root/.netrc

RUN git clone https://github.com/noxrepo/pox.git /pox

WORKDIR /pox

RUN git checkout gar-experimental 

RUN rm -r .git

COPY entrypoint.sh /pox/entrypoint.sh

COPY  smartController/. /pox/pox/smartController/

RUN chmod -R 777 /pox/

ENTRYPOINT ["/pox/entrypoint.sh"]