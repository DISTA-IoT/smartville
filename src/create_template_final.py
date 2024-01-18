
from gns3util import Server, create_docker_template, create_docker_template_router, create_docker_template_switch


indirizzo_ip = "127.0.0.1"
porta = 3080
utente = "admin"
password = "gns3"

# Crea l'oggetto Server
server = Server(addr=indirizzo_ip, port=porta, auth=(utente, password), user=utente, password=password)
create_docker_template(server, "pox-controller", "pox-controller:latest", '')
"""
create_docker_template_switch(server, "matgo01-my-sdnswitch2-mt", "matgo01/my-sdnswitch2-mt", '')
create_docker_template_router(server, "matgo01-my-router2-mt", "matgo01/my-router2-mt", ' ')
create_docker_template(server, "matgo01-iot-device-n6-mt", "matgo01/iot-device-n6-mt", ' ')
create_docker_template(server, "matgo01-iot-device-n7-mt", "matgo01/iot-device-n7-mt", ' ')
create_docker_template(server, "matgo01-iot-device-n8-mt", "matgo01/iot-device-n8-mt", ' ')
#create_docker_template(server,"matgo01-mqtt-broker2-mt","matgo01/mqtt-broker2-mt")
In questo momento per congifurare l'interfaccia di rete è necessario farlo manualmento andando
sulle configurazione del nodo e eliminando i commenti """