FROM python:3.8


RUN apt-get update && apt-get install -y git nano vim python3-pip

RUN apt-get install -y libpcap0.8 libpcap0.8-dev

RUN pip3 install --upgrade pip

RUN pip3 install scapy networkx pydot numpy scikit-learn wandb

RUN pip3 install torch --index-url https://download.pytorch.org/whl/cpu

RUN git clone https://github.com/noxrepo/pox.git /pox

WORKDIR /pox

RUN git checkout gar-experimental 

RUN rm -r .git

COPY entrypoint.sh /pox/entrypoint.sh

COPY  smartController/. /pox/pox/smartController/

RUN chmod -R 777 /pox/

ENTRYPOINT ["/pox/entrypoint.sh"]