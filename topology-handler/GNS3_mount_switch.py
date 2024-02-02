from gns3util import *


def mountSwitch(PROJECT_NAME,SWITCH_NAME,SWITCH_LABEL,ip,gateway):
    server = Server(*read_local_gns3_config())
    project = get_project_by_name(server, PROJECT_NAME)

    templates = get_all_templates(server)
    switch_template_id = get_template_id_from_name(templates, SWITCH_NAME)

    if(switch_template_id is not None):
        delete_template(server,project,switch_template_id)
        print((f"{SWITCH_LABEL}: old switch template deleted"))

    print((f"{SWITCH_LABEL}: creating a new template using local image"))
    create_docker_template_switch(server, SWITCH_NAME, str(SWITCH_NAME+":latest"),)
    templates = get_all_templates(server)
    switch_template_id = get_template_id_from_name(templates, SWITCH_NAME)
    print(f"new switch template id: {switch_template_id}")

    openvswitch=create_node(server, project, 0, 0, switch_template_id)
    print(f"{SWITCH_LABEL}: created")
    openvswitch_id = openvswitch['node_id']

    set_node_network_interfaces(server, project, openvswitch_id, "eth0", ipaddress.IPv4Interface(ip), gateway)
    print(f"{SWITCH_LABEL}: assigned ip: {ip}, gateway: {gateway} on eth0")
    start_node_by_name(server,project,SWITCH_LABEL)
    print(f"{SWITCH_LABEL}: started")
