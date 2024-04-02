import ipaddress
from gns3util import *

#args: project name, host
def mountHost(PROJECT_NAME,HOST_NAME,HOST_LABEL,SWITCH_LABEL,EDGE_LABEL,switch_port,ip,gateway,x,y,start_command):
    server = Server(*read_local_gns3_config())
    project = get_project_by_name(server, PROJECT_NAME)

    templates = get_all_templates(server)
    host_template_id = get_template_id_from_name(templates, HOST_NAME)

    if(host_template_id is not None):  
        delete_template(server,project,host_template_id)
        print((f"{HOST_LABEL}: deleting old host template"))
        
    print((f"{HOST_LABEL}: creating a new template using local image"))
    create_docker_template(server, HOST_NAME, start_command, str(HOST_NAME+":latest"),)
    templates = get_all_templates(server)
    host_template_id = get_template_id_from_name(templates, HOST_NAME)
    print(f"new host template id: {host_template_id}")

    openvswitch_id = get_node_id_by_name(server,project,SWITCH_LABEL)
    edgeswitch_id = get_node_id_by_name(server,project,EDGE_LABEL)

    host=create_node(server, project, x, y, host_template_id,HOST_LABEL)
    host_id=host['node_id']
    print(f"{HOST_LABEL}: created")
    #set node interfaces
    set_node_network_interfaces(server, project, host_id, "eth0", ipaddress.IPv4Interface(ip), gateway)
    print(f"{HOST_LABEL}: assigned ip: {ip}, gateway: {gateway} on eth0")

    set_dhcp_node_network_interfaces(server, project, host_id, "eth1")
    print(f"{SWITCH_LABEL}: DHCP on ","eth1")

    #link to switch 1
    create_link(server, project,host_id,1,openvswitch_id,switch_port)
    #link to edge switch
    create_link(server, project,host_id,0,edgeswitch_id,switch_port)

    print(f"{HOST_LABEL}: link to {SWITCH_LABEL} on port {switch_port} created")
    start_node(server,project,host_id)
    print(f"{HOST_LABEL}: started")