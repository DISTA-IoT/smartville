import ipaddress
from gns3util import *
import time

def mountController(PROJECT_NAME,CONTROLLER_NAME,switch_name,start_command,ip):
    server = Server(*read_local_gns3_config())
    project = get_project_by_name(server, PROJECT_NAME)

    templates = get_all_templates(server)
    controller_template_id = get_template_id_from_name(templates, CONTROLLER_NAME)

    if(controller_template_id is not None):
        delete_template(server,project,controller_template_id)
        print(f"old controller template {CONTROLLER_NAME} deleted")

    
    create_docker_template(server, CONTROLLER_NAME,start_command, str(CONTROLLER_NAME+":latest"),)
    templates = get_all_templates(server)
    controller_template_id = get_template_id_from_name(templates, CONTROLLER_NAME)
    print(f"new controller template id: {controller_template_id}")

    controller_name = "pox-controller-1 "+"("+ip+")"
    openvswitch_id = get_node_id_by_name(server,project,switch_name)
    controller_id = get_node_id_by_name(server,project,controller_name)
    print(f"controller id {controller_id}, switch id {openvswitch_id}")
    if(controller_id is not None):
        delete_node(server,project,controller_id)
        print("Old controller node deleted")

    controller = create_node(server, project, 0, -100, controller_template_id, controller_name)
    controller_id = controller['node_id']
    print(f"new {CONTROLLER_NAME} controller created ")
    time.sleep(2)
    create_link(server, project, controller_id,0,openvswitch_id,0)
    print(f"Created a link from {CONTROLLER_NAME} to {switch_name} on port eth0")
    set_node_network_interfaces(server, project, controller_id, "eth0", ipaddress.IPv4Interface("192.168.1.1/24"), None)
    print(f"{CONTROLLER_NAME}: assigned ip: {ip} on eth0")
    set_dhcp_node_network_interfaces(server,project,controller_id,"eth1")
    start_node(server,project,controller_id)
    print(f"{CONTROLLER_NAME}: started")
    return controller_name






