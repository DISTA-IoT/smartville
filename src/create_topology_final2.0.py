
from gns3util import Server, create_node, create_project, get_template_id_from_name, get_all_templates, create_link_by_name


#server = Server(*read_local_gns3_config())
#check_server_version(server)
# Indirizzo IP e porta del server GNS3
indirizzo_ip = "192.168.105.130"
porta = 80
utente = "admin"
password = "gns3"

# Crea l'oggetto Server
server = Server(addr=indirizzo_ip, port=porta,auth=(utente, password),user=utente, password=password)
#crea il progetto
project = create_project(server, name="myProject_matteo-mt", height=100, width=150, zoom=50)
#crea una lista con tutti i template
templates_list = get_all_templates(server)
#ottiene l'id dei template dato il nome del template
sdnswitch_id = get_template_id_from_name(templates_list, "matgo01-my-sdnswitch2-mt")
router_id = get_template_id_from_name(templates_list, "matgo01-my-router2-mt")
iot6_id = get_template_id_from_name(templates_list, "matgo01-iot-device-n6-mt")
iot7_id = get_template_id_from_name(templates_list, "matgo01-iot-device-n7-mt")
iot8_id = get_template_id_from_name(templates_list, "matgo01-iot-device-n8-mt")
#mqttbroker_id=get_template_id_from_name(templates_list,"matgo01-mqtt-broker2-mt")
#creazione dei nodi
create_node(server, project, start_x=0, start_y=0, node_template_id=sdnswitch_id, node_name="matgo01-my-sdnswitch2-mt")
create_node(server, project, start_x=0, start_y=300, node_template_id=router_id, node_name="matgo01-my-router2-mt")
create_node(server, project, start_x=-300, start_y=0, node_template_id=iot6_id, node_name="matgo01-iot-device-n6-mt")
create_node(server, project, start_x=0, start_y=-300, node_template_id=iot7_id, node_name="matgo01-iot-device-n7-mt")
create_node(server, project, start_x=300, start_y=0, node_template_id=iot8_id, node_name="matgo01-iot-device-n8-mt")
#create_node(server,project,start_x=400,start_y=0,node_template_id=mqttbroker_id,node_name="matgo01-mqtt-broker2-mt")




#creazione collegamenti
# Collegamento tra router e switch
create_link_by_name(server, project, "matgo01-my-router2-mt", 1, "matgo01-my-sdnswitch2-mt", 0)
#collegamento iot6 con switch
create_link_by_name(server, project, "matgo01-iot-device-n6-mt", 1, "matgo01-my-sdnswitch2-mt", 1)
#collegamento iot7 con switch
create_link_by_name(server, project, "matgo01-iot-device-n7-mt", 1, "matgo01-my-sdnswitch2-mt", 2)
#collegamento iot8 con switch
create_link_by_name(server, project, "matgo01-iot-device-n8-mt", 1, "matgo01-my-sdnswitch2-mt", 3)
#collegamento mqtt_broker con router
#create_link_by_name(server,project,"matgo01-mqtt-broker2-mt",1,"matgo01-my-router2-mt",2)
