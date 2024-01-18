from gns3util import Server,start_node_by_name, start_capture, get_project_by_name

indirizzo_ip = "192.168.105.130"
porta = 80
utente = "admin"
password = "gns3"

# Crea l'oggetto Server
server = Server(addr=indirizzo_ip, port=porta,auth=(utente, password),user=utente, password=password)
# Ottiene il progetto GNS3 specifico per il nome
project = get_project_by_name(server,"myProject_matteo70")
#avvia i nodi presenti nel progetto dato il loro nome
start_node_by_name(server,project,'matgo01-my-sdnswitch2-mt')
start_node_by_name(server,project,'matgo01-my-router2-mt')
start_node_by_name(server,project,'matgo01-my-iot-device-n6-mt')
start_node_by_name(server,project,'matgo01-my-iot-device-n7-mt')
start_node_by_name(server,project,'matgo01-iot-device-n8-mt')
start_node_by_name(server,project,'matgo01-mqttbroker2-mt')
