# This file is part of the "Smartville" project.
# Copyright (c) 2024 University of Insubria
# Licensed under the Apache License 2.0.
# SPDX-License-Identifier: Apache-2.0
# For the full text of the license, visit:
# https://www.apache.org/licenses/LICENSE-2.0

# Smartville is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# Apache License 2.0 for more details.

# You should have received a copy of the Apache License 2.0
# along with Smartville. If not, see <https://www.apache.org/licenses/LICENSE-2.0>.

# Additional licensing information for third-party dependencies
# used in this file can be found in the accompanying `NOTICE` file.
from gns3util import *
import gns3fy as gfy
import argparse


PROJECT_NAME = None

GNS3_HOST = None
GNS3_PORT = None

ATTACKER_NODE_COUNT = None
VICTIM_NODE_COUNT = None

CONTROLLER_IMG_NAME = None
SWITCH_IMG_NAME = None
VICTIM_IMG_NAME = None
ATTACKER_IMG_NAME = None
NAT_IMG_NAME = "NAT"

CONTROLLER_START_COMMAND=None


node_ids = {}
template_ids = {}
server = None
gns3_server_connector = None
project = None
templates = None


def resetProject(PROJECT_NAME):
    global project

    delete_project(server,PROJECT_NAME)
    print("Project deleted")
    create_project(server,PROJECT_NAME,1000,1000)
    print("New project created")
    project = get_project_by_name(server, PROJECT_NAME)
    open_project_if_closed(server, project)


def generateIPList(ip_nr,network,netmask):
    ip_pool = []
    for i in range(3,ip_nr+3):
        net_ip = network[:-1]
        ip = net_ip+str(i)+netmask
        ip_pool.append(ip)
        print("IP ADDED TO LIST: ",ip)
    return ip_pool


def mountSwitch(curr_switch_label,ip,gateway):
    template_id = get_template_id_from_name(templates, SWITCH_IMG_NAME)
    switch1_node_name = curr_switch_label + "("+ ip +")"
    openvswitch=create_node(server, project, 0, 100, template_id, switch1_node_name)

    print(f"{curr_switch_label}: created")
    openvswitch_id = openvswitch['node_id']

    set_node_network_interfaces(server, project, openvswitch_id, "eth0", ipaddress.IPv4Interface(ip), gateway)
    print(f"{curr_switch_label}: assigned ip: {ip}, gateway: {gateway} on eth0")
    start_node_by_name(server,project,switch1_node_name)
    print(f"{curr_switch_label}: started")
    return switch1_node_name


def mount_edge_switch():
    template_id = get_template_id_from_name(templates, SWITCH_IMG_NAME)
    curr_switch_label = "openvswitch-edge-1"
    edge_openvswitch=create_node(server, project, 0, -200, template_id,curr_switch_label)
    print(f"{curr_switch_label}: created")
    edge_openvswitch_id = edge_openvswitch['node_id']

    interface = 'eth1'
    set_dhcp_node_network_interfaces(server, project, edge_openvswitch_id, interface, None)
    print(f"{curr_switch_label}: DHCP on ",interface)

    start_node(server,project,edge_openvswitch_id)
    print(f"{curr_switch_label}: started")
    return curr_switch_label


def mountController(switch_name, ip):
    template_id = get_template_id_from_name(templates, CONTROLLER_IMG_NAME)
    controller_name = "pox-controller-1"+"("+ip+")"
    openvswitch_id = get_node_id_by_name(server,project,switch_name)
    controller_id = get_node_id_by_name(server,project,controller_name)
    print(f"controller id {controller_id}, switch id {openvswitch_id}")
    if(controller_id is not None):
        delete_node(server,project,controller_id)
        print("Old controller node deleted")

    controller = create_node(server, project, 0, 0, template_id, controller_name)
    controller_id = controller['node_id']
    print(f"new {CONTROLLER_IMG_NAME} controller created ")
    time.sleep(2)
    create_link(server, project, controller_id,0,openvswitch_id,0)
    print(f"Created a link from {CONTROLLER_IMG_NAME} to {switch_name} on port eth0")
    set_node_network_interfaces(server, project, controller_id, "eth0", ipaddress.IPv4Interface("192.168.1.1/24"), None)
    print(f"{CONTROLLER_IMG_NAME}: assigned ip: {ip} on eth0")
    set_dhcp_node_network_interfaces(server,project,controller_id,"eth1", "smartcontroller")
    start_node(server,project,controller_id)
    print(f"{CONTROLLER_IMG_NAME}: started")
    return controller_name


def mountNAT():    
    NAT_template_id = get_template_id_from_name(templates, NAT_IMG_NAME)
    print("NAT TEMPLATE ID: ",NAT_template_id)

    # Create a new node
    nat_node = gfy.Node(
        project_id=project.id, 
        connector=gns3_server_connector, 
        name=NAT_IMG_NAME, 
        template_id= NAT_template_id,
        x=0,
        y=-350)

    # Add the node to the project
    nat_node.create()
    print("NAT created")


def mount_single_Host(curr_img_name, curr_node_name,switch1_node_name,switch_port,ip,gateway,x,y):
    template_id = get_template_id_from_name(templates, curr_img_name)

    openvswitch_id = get_node_id_by_name(server,project,switch1_node_name)

    host=create_node(server, project, x, y, template_id,curr_node_name)
    host_id=host['node_id']
    print(f"{curr_node_name}: created")
    set_node_network_interfaces(server, project, host_id, "eth0", ipaddress.IPv4Interface(ip), gateway)
    set_dhcp_node_network_interfaces(server,project,host_id,"eth1", None)

    print(f"{curr_node_name}: assigned ip: {ip}, gateway: {gateway} on eth0")
    create_link(server, project,host_id,0,openvswitch_id,switch_port)
    print(f"{curr_node_name}: link to {switch1_node_name} on port {switch_port} created")
    start_node(server,project,host_id)
    print(f"{curr_node_name}: started")


def mount_all_hosts(switch_node_name):
    node_names = []
    # mounts hosts and links each one to a port of the switch
    gateway = None  
    switch_port = 3
    network = "192.168.1.0"
    netmask = "/24"
    #generate pool of ip addresses for specified network (es. 192.168.1.0)
    ip_pool = generateIPList(ATTACKER_NODE_COUNT + VICTIM_NODE_COUNT, network, netmask)
    x = 300
    y = -200
    i = 1
    half = False

    for idx, ip in enumerate(ip_pool):

        img_name = VICTIM_IMG_NAME
        curr_node_name = VICTIM_IMG_NAME+"-"+str(idx)+"("+ip+")"

        if (i > (len(ip_pool))/2) and not half:
            half = True
            x = -300
            y = -200
        
        if idx > VICTIM_NODE_COUNT-1:
            img_name = ATTACKER_IMG_NAME
            curr_node_name = ATTACKER_IMG_NAME+"-"+str(idx)+"("+ip+")"
        
        mount_single_Host(
            img_name,
            curr_node_name,
            switch_node_name,
            switch_port,
            ip,
            gateway,
            x,
            y)
        
        i = i+1
        y = y+100
        switch_port = switch_port+1
        node_names.append(curr_node_name)
    return node_names


def connect_all(edge_switch_node_name,controller_node_name,host_names):

    nat_id = get_node_id_by_name(server, project, NAT_IMG_NAME)
    edge_switch_id = get_node_id_by_name(server, project, edge_switch_node_name)
    create_link(server, project,str(nat_id),0,str(edge_switch_id),1)

    controller_id = get_node_id_by_name(server, project, controller_node_name)
    create_link(server, project,str(edge_switch_id),2,str(controller_id),1)

    for idx, host_name in enumerate(host_names):
        host_id = get_node_id_by_name(server, project, host_name)
        create_link(server, project,str(edge_switch_id),3+idx,str(host_id),1)


def starTopology():
    switch1_node_name = mountSwitch("openvswitch-1","192.168.1.2/24","192.168.1.1")
    edge_switch_node_name = mount_edge_switch()
    controller_node_name = mountController(switch1_node_name,"192.168.1.1/24")
    host_names = mount_all_hosts(switch1_node_name)
    mountNAT()
    connect_all(edge_switch_node_name,controller_node_name,host_names)


def update_generic_template(img_name, start_command):
    global project

    template_id = get_template_id_from_name(templates, img_name)
    if(template_id is not None):  
        delete_template(server,project,template_id)
        print((f"{template_id}: deleting old template"))
        
    print((f"{template_id}: creating a new template using local image"))
    create_docker_template(server, img_name, start_command, str(img_name+":latest"),)


def update_switch_template():
    global project

    switch_template_id = get_template_id_from_name(templates, SWITCH_IMG_NAME)
    if(switch_template_id is not None):
        delete_template(server,project,switch_template_id)
        print((f"{SWITCH_IMG_NAME}: old switch template deleted"))
    print((f"{SWITCH_IMG_NAME}: creating a new template using local image"))
    network_adapters_count = 3 + VICTIM_NODE_COUNT + ATTACKER_NODE_COUNT
    create_docker_template_switch(server, SWITCH_IMG_NAME, str(SWITCH_IMG_NAME+":latest"), adapter_count=network_adapters_count)


def update_controller_template():
    global project

    controller_template_id = get_template_id_from_name(templates, CONTROLLER_IMG_NAME)
    if(controller_template_id is not None):
        delete_template(server,project,controller_template_id)
        print(f"old controller template {CONTROLLER_IMG_NAME} deleted")

    create_docker_template(server, CONTROLLER_IMG_NAME, CONTROLLER_START_COMMAND, str(CONTROLLER_IMG_NAME+":latest"),)


def update_templates():
    update_switch_template()
    update_controller_template()
    update_generic_template(ATTACKER_IMG_NAME, 'sh')
    update_generic_template(VICTIM_IMG_NAME, 'sh')


if __name__ == "__main__":


    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Topology creation script")
    parser.add_argument("--project", help="GNS3 Project Name (Default is \"SmartVille\")", default="SmartVille")
    parser.add_argument("--controller_docker", help="Controller's Docker image Name (Default is \"pox-controller\")", default="pox-controller")
    parser.add_argument("--switch_docker", help="SDN Switch's Docker image Name (Default is \"openvswitch\")", default="openvswitch")
    parser.add_argument("--victim_docker", help="Victim's Docker image Name (Default is \"victim\")", default="victim")
    parser.add_argument("--attacker_docker", help="Attacker's Docker image Name (Default is \"attacker\")", default="attacker")

    parser.add_argument("--contr_start", help="Controller's Start Command.  (Default is \"sh\")\n "+ \
                        "Could also be:  \"./pox.py samples.pretty_log smartController.smartController\"", default="sh")
    parser.add_argument("--n_attackers", type=int, default=10, help="Number of attacker nodes in the topology. Default: 10")
    parser.add_argument("--n_victims", type=int, default=4, help="Number of victim nodes in the topology. Default: 4")


    parser.add_argument("--use_gns3_config_file", type=bool, default=True, help="Grab GNS3 Server configurations from file. Default: True")
    parser.add_argument("--gns3_config_path", type=str, default="~/.config/GNS3/2.2/gns3_server.conf", help="GNS3 Server config file, "+\
                        "Default: \"~/.config/GNS3/2.2/gns3_server.conf\"")

    parser.add_argument("--gns3_host", type=str, default="localhost", help="When not using config file. GNS3 server's hostname (Default is \"localhost\")")
    parser.add_argument("--gns3_port", type=int, default=3080, help="When not using config file. GNS3 server's port. Default: 3080")
    parser.add_argument("--gns3_auth", type=bool, default=True, help=" When not using config file. GNS3 server requires auth (Default True)")
    parser.add_argument("--gns3_username", type=str, default="admin", help="When not using config file. GNS3 server's admin username (Default is \"admin\")")
    parser.add_argument("--gns3_password", type=str, default="12345", help="When not using config file. GNS3 server's password (Default is \"12345\")")

    args = parser.parse_args()

    PROJECT_NAME = args.project
    
    USE_GNS3_FILE = args.use_gns3_config_file
    GNS3_CONFIG_PATH = args.gns3_config_path

    CONTROLLER_IMG_NAME = args.controller_docker
    SWITCH_IMG_NAME = args.switch_docker
    VICTIM_IMG_NAME = args.victim_docker
    ATTACKER_IMG_NAME = args.attacker_docker
    CONTROLLER_START_COMMAND = args.contr_start

    ATTACKER_NODE_COUNT = args.n_attackers
    VICTIM_NODE_COUNT = args.n_victims

    if USE_GNS3_FILE:
        server = Server(*read_local_gns3_config(GNS3_CONFIG_PATH))
        GNS3_HOST = server.addr
        GNS3_PORT = server.port
        GNS3_AUTH = server.auth
        GNS3_USERNAME = server.user
        GNS3_PASSWORD = server.password
    else:
        GNS3_HOST = args.gns3_host
        GNS3_PORT = args.gns3_port
        GNS3_AUTH = args.gns3_auth
        GNS3_USERNAME = args.gns3_username
        GNS3_PASSWORD = args.gns3_password    


    gns3_server_connector = gfy.Gns3Connector(f"http://{GNS3_HOST}:{GNS3_PORT}", user=GNS3_USERNAME, cred=GNS3_PASSWORD)

    resetProject(PROJECT_NAME)

    templates = get_all_templates(server)
    update_templates()
    templates = get_all_templates(server)

    starTopology()
