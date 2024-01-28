import ipaddress
from gns3util import *

#args: project name, host
def mountHost(PROJECT_NAME,HOST_NAME,HOST_LABEL,SWITCH_LABEL,switch_port,ip,gateway,x,y):
    server = Server(*read_local_gns3_config())
    project = get_project_by_name(server, PROJECT_NAME)

    templates = get_all_templates(server)
    host_template_id = get_template_id_from_name(templates, HOST_NAME)

    if(host_template_id is None):  
        print((f"{HOST_LABEL}: template is not present in gns3 -> trying to create a new template using local image"))
        create_docker_template(server, HOST_NAME, str(HOST_NAME+":latest"),)
        templates = get_all_templates(server)
        host_template_id = get_template_id_from_name(templates, HOST_NAME)
        print(f"new host template id: {host_template_id}")

    openvswitch_id = get_node_id_by_name(server,project,SWITCH_LABEL)

    host=create_node(server, project, x, y, host_template_id,HOST_LABEL)
    host_id=host['node_id']
    print(f"{HOST_LABEL}: created")
    set_node_network_interfaces(server, project, host_id, "eth0", ipaddress.IPv4Interface(ip), gateway)
    print(f"{HOST_LABEL}: assigned ip: {ip}, gateway: {gateway} on eth0")
    create_link(server, project,host_id,0,openvswitch_id,switch_port)
    print(f"{HOST_LABEL}: link to {SWITCH_LABEL} on port {switch_port} created")
    start_node(server,project,host_id)
    print(f"{HOST_LABEL}: started")