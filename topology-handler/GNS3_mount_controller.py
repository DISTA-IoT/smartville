import ipaddress
from gns3util import *

def mountController(PROJECT_NAME,CONTROLLER_NAME):
    server = Server(*read_local_gns3_config())
    project = get_project_by_name(server, PROJECT_NAME)

    templates = get_all_templates(server)
    controller_template_id = get_template_id_from_name(templates, CONTROLLER_NAME)

    if(controller_template_id is not None):
        delete_template(server,project,controller_template_id)
        print("old controller template deleted")

    create_docker_template(server, CONTROLLER_NAME, str(CONTROLLER_NAME+":latest"),)
    templates = get_all_templates(server)
    controller_template_id = get_template_id_from_name(templates, CONTROLLER_NAME)
    print(f"new controller template id: {controller_template_id}")

    switch_name = "FixedOpenvSwitch-1"
    controller_name = "pox-controller-1"
    openvswitch_id = get_node_id_by_name(server,project,switch_name)
    controller_id = get_node_id_by_name(server,project,controller_name)
    print(f"controller id {controller_id}, switch id {openvswitch_id}")
    if(controller_id is not None):
        delete_node(server,project,controller_id)
        print("deleted old controller")

    controller = create_node(server, project, 0, -100, controller_template_id)
    controller_id = get_node_id_by_name(server,project,controller_name)
    print("created new controller")
    create_link(server, project, controller_id,0,openvswitch_id,0)
    set_node_network_interfaces(server, project, controller_id, "eth0", ipaddress.IPv4Interface("192.168.1.1/24"), "192.168.1.1")
    print("controller started")
    start_node_by_name(server,project,controller_name)





