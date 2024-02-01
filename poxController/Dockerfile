FROM python:3.8


RUN apt-get update && apt-get install -y git nano vim python3-pip

RUN apt-get install -y libpcap0.8 libpcap0.8-dev telnet

RUN pip3 install --upgrade pip

RUN pip3 install scapy networkx pydot

RUN git clone https://github.com/noxrepo/pox.git /pox

WORKDIR /pox

COPY entrypoint.sh /pox/entrypoint.sh

RUN mkdir /pox/pox/customScript

COPY  py/. /pox/pox/customScript/

RUN chmod -R 777 /pox/

CMD ["/pox/entrypoint.sh"]