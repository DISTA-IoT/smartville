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


NAT_IMG_NAME = "NAT"

node_ids = []
template_ids = {}
gns3_server_connector = None


def resetProject(args):
    print(f"The {args.project} GNS3 project will be deleted if existent")
    delete_project(args.server,args.project)
    create_project(args.server,args.project,1000,1000)
    print(f"New GNS3 project created with name {args.project}")
    project = get_project_by_name(args.server, args.project)
    open_project_if_closed(args.server, project)
    return project

def generate_static_IP_list(ip_nr,network,netmask, starting_from=3):
    ip_pool = []
    for i in range(starting_from, ip_nr+starting_from):
        net_ip = network[:-1]
        ip = net_ip+str(i)+netmask
        ip_pool.append(ip)
        print("IP ADDED TO LIST: ",ip)
    return ip_pool


def mount_switches(args, templates, gateway=None):

    switch_node_names = []
    template_id = get_template_id_from_name(templates, args.switch_docker)

    for i in range(args.n_switches):
        curr_switch_label = f"openvswitch-"+str(i+1)
        ip = f"192.168.1.{i+2}/24"
        switch_node_name = curr_switch_label + "\n("+ ip +")"
        openvswitch=create_node(args.server, args.project, -400*(args.n_switches//2) + (400*i), 150, template_id, switch_node_name)
        print(f"{curr_switch_label}: created")
        openvswitch_id = openvswitch['node_id']

        set_node_network_interfaces(args.server, args.project, openvswitch_id, "eth0", ipaddress.IPv4Interface(ip), gateway)
        print(f"{curr_switch_label}: assigned ip: {ip}, gateway: {gateway} on eth0")
        print(f"{curr_switch_label}: started")
        node_ids.append(openvswitch_id)
        switch_node_names.append(switch_node_name)
        
    return switch_node_names


def mount_edge_switch(args, templates):
    """
    We did not add Gateways to node configuration in GNS3. If you need to do so, refer to the GNS3utils API.
    """
    template_id = get_template_id_from_name(templates, args.switch_docker)
    curr_switch_label = "openvswitch-edge-1"
    edge_openvswitch=create_node(args.server, args.project, 400*(args.n_switches//2), 150, template_id,curr_switch_label)
    print(f"{curr_switch_label}: created")
    edge_openvswitch_id = edge_openvswitch['node_id']

    interface = 'eth1'
    set_dhcp_node_network_interfaces(args.server, args.project, edge_openvswitch_id, interface, None)
    print(f"{curr_switch_label}: DHCP on ",interface)

    node_ids.append(edge_openvswitch_id)
    print(f"{curr_switch_label}: started")
    return curr_switch_label


def mountController(args, templates, last_switch_name):
    """
    We did not add Gateways to node configuration in GNS3. If you need to do so, refer to the GNS3utils API.
    """
    template_id = get_template_id_from_name(templates, args.controller_docker)
    ip_address = f"192.168.1.1/24"
    controller_name = "pox-controller-1\n"+ip_address
    controller_id = get_node_id_by_name(args.server,args.project,controller_name)
    
    if(controller_id is not None):
                delete_node(args.server,args.project,controller_id)
                print("Old controller node deleted")

    controller = create_node(args.server, args.project, 0, -50, template_id, controller_name)
    controller_id = controller['node_id']
    print(f"new {args.controller_docker} controller created ")
    time.sleep(2)

    openvswitch_id = get_node_id_by_name(args.server,args.project,last_switch_name)

    create_link(
        args.server, 
        args.project,
        controller_id,
        0,
        openvswitch_id,
        0)
        
    set_node_network_interfaces(
        args.server,
        args.project,
        controller_id,
        f"eth0",
        ipaddress.IPv4Interface(ip_address),
        None)
        
    node_ids.append(controller_id)

    return controller_name


def mountBotMaster(args, templates, first_switch_name):

    template_id = get_template_id_from_name(templates, args.botmaster_docker)
    ip_address = f"192.168.1.{args.n_victims + args.n_attackers + args.n_switches + 2}/24"
    botmaster_name = "botmaster-1\n"+ip_address
    botmaster_id = get_node_id_by_name(args.server,args.project,botmaster_name)
    if(botmaster_id is not None):
                delete_node(args.server,args.project,botmaster_id)
                print("Old botmaster node deleted")
    
    botmaster = create_node(args.server, args.project, -200, -50, template_id, botmaster_name)
    botmaster_id = botmaster['node_id']
    print(f"new {args.botmaster_docker} controller created ")
    time.sleep(2)

    openvswitch_id = get_node_id_by_name(args.server, args.project, first_switch_name)

    create_link(
        args.server, 
        args.project,
        botmaster_id,
        0,
        openvswitch_id,
        (args.nodes_per_switch*2)+2)
        
    
    set_node_network_interfaces(
        args.server,
        args.project,
        botmaster_id,
        f"eth0",
        ipaddress.IPv4Interface(ip_address),
        None)
        
    create_link(
        args.server, 
        args.project,
        botmaster_id,
        1,
        openvswitch_id,
        (args.nodes_per_switch*2)+3)
    set_dhcp_node_network_interfaces(
            args.server,
            args.project,
            botmaster_id,
            f"eth1", 
            None)
    
    node_ids.append(botmaster_id)

    return botmaster_name


def mountNAT(args, templates):    
    NAT_template_id = get_template_id_from_name(templates, NAT_IMG_NAME)
    print("NAT TEMPLATE ID: ", NAT_template_id)

    # Create a new node
    nat_node = gfy.Node(
        project_id=args.project.id, 
        connector=gns3_server_connector, 
        name=NAT_IMG_NAME, 
        template_id= NAT_template_id,
        x=0,
        y=-500)

    # Add the node to the project
    nat_node.create()
    print("NAT created")


def mount_single_Host(args, templates, curr_img_name, curr_node_name,switch1_node_name,switch_port,ip,gateway,x,y):

    template_id = get_template_id_from_name(templates, curr_img_name)
    openvswitch_id = get_node_id_by_name(args.server,args.project,switch1_node_name)
    host=create_node(args.server, args.project, x, y, template_id,curr_node_name)
    host_id=host['node_id']
    print(f"{curr_node_name}: created")
    set_node_network_interfaces(args.server, args.project, host_id, "eth0", ipaddress.IPv4Interface(ip), gateway)
    set_dhcp_node_network_interfaces(args.server,args.project,host_id,"eth1", None)

    print(f"{curr_node_name}: assigned ip: {ip}, gateway: {gateway} on eth0")
    create_link(args.server, args.project,host_id,0,openvswitch_id,switch_port)
    print(f"{curr_node_name}: link to {switch1_node_name} on port {switch_port} created")
    create_link(args.server, args.project,host_id,1,openvswitch_id,switch_port+1)
    node_ids.append(host_id)
    print(f"{curr_node_name}: started")


def mount_all_hosts(args, templates, switch_node_names, curr_node_count):
    node_names = []
    # mounts hosts and links each one to a port of the switch
    gateway = None  
    network = "192.168.1.0"
    netmask = "/24"
    node_count = args.n_victims + args.n_attackers

    # generate a pool of ip addresses for specified network (es. 192.168.1.0)
    ip_pool = generate_static_IP_list(
        node_count, 
        network, 
        netmask, 
        starting_from=curr_node_count+1)
    
    y = 250

    for switch_idx, switch_name in enumerate(switch_node_names):

        switch_horizontal_position = -400*(len(switch_node_names)//2) + (400*switch_idx)
        current_switch_port = 1

        pools_for_switch = ip_pool[
            switch_idx*args.nodes_per_switch:(switch_idx+1)*args.nodes_per_switch]

        for idx, ip in enumerate(pools_for_switch):

            img_name = args.victim_docker
            curr_node_name = f"User-{(switch_idx*args.nodes_per_switch)+idx}\n({ip})"
            
            node_horizontal_position =  switch_horizontal_position -150*(args.nodes_per_switch//2) + 150*idx

            mount_single_Host(
                args,
                templates,
                img_name,
                curr_node_name,
                switch_name,
                current_switch_port,
                ip,
                gateway,
                node_horizontal_position,
                y)
            
            current_switch_port = current_switch_port+2
            node_names.append(curr_node_name)

    return node_names


def connect_all(args, switch_node_names, controller_node_name):
    
    
    for idx in range(len(switch_node_names)-1):
        switch_name_1 = switch_node_names[idx]
        switch_name_2 = switch_node_names[idx+1]
        switch_id_1 = get_node_id_by_name(args.server, args.project, switch_name_1)
        switch_id_2 = get_node_id_by_name(args.server, args.project, switch_name_2)

        # links for generic traffic.
        create_link(
            args.server,
            args.project,
            str(switch_id_1),
            (args.nodes_per_switch*2)+1,
            str(switch_id_2),
            (args.nodes_per_switch*2)+2)
        set_dhcp_node_network_interfaces(
            args.server,
            args.project,
            switch_id_1,
            f"eth{(args.nodes_per_switch*2)+1}", 
            None)
        
        # control links: (the last switch already had it created)
        create_link(
            args.server,
            args.project,
            str(switch_id_1),
            0,
            str(switch_id_2),
            (args.nodes_per_switch*2)+3)


    # Connect the last switch to the NAT with a dhcp interface: 
    nat_node_id = get_node_id_by_name(args.server, args.project, NAT_IMG_NAME)

    create_link(
        args.server,
        args.project,
        str(switch_id_2),
        (args.nodes_per_switch*2)+1,
        str(nat_node_id),
        0
    )
    set_dhcp_node_network_interfaces(
        args.server,
        args.project,
        switch_id_2,
        f"eth{(args.nodes_per_switch*2)+1}", 
        None)
    
    # Connect also the controller to this edge switch
    controller_id = get_node_id_by_name(args.server, args.project, controller_node_name)
    create_link(
        args.server, 
        args.project, 
        str(switch_id_2), 
        (args.nodes_per_switch*2)+4, 
        str(controller_id), 
        1
    )
    set_dhcp_node_network_interfaces(
        args.server,
        args.project,
        controller_id,
        f"eth1",
        "smartcontroller")


def start_all(args):
    for id in node_ids:
        start_node(args.server, args.project, id)
        print("Node: ",id," started")


def tree_topology(args, templates):

    switch_node_names = mount_switches(args, templates)

    controller_node_name = mountController(args, templates, switch_node_names[-1])

    botmaster_node_name = mountBotMaster(args, templates, switch_node_names[0])

    host_names = mount_all_hosts(args, templates, switch_node_names, 
                                 curr_node_count=1+args.n_switches)
    
    mountNAT(args, templates)
    connect_all(args, switch_node_names, controller_node_name)
    start_all(args)


def update_generic_template(args, templates, img_name, start_command):

    template_id = get_template_id_from_name(templates, img_name)
    if(template_id is not None):  
        delete_template(args.server,args.project,template_id)
        print((f"{img_name}: deleting old template"))
        
    print((f"{img_name}: creating a new template using local image"))
    create_docker_template(
         args.server,
         img_name,
         start_command,
         str(img_name+":latest"),
         environment=args.env_vars)


def update_switch_template(args, templates):

    switch_template_id = get_template_id_from_name(templates, args.switch_docker)
    if(switch_template_id is not None):
        delete_template(args.server,args.project,switch_template_id)
        print((f"{args.switch_docker}: old switch template deleted"))
    print((f"{args.switch_docker}: creating a new template using local image"))

    # we multiply nodes_per_switch*2 because each node is gonna be connected to internet through a separated dhcp connection. 
    network_adapters_count = 2*args.nodes_per_switch
    # we also add:
    # two additional connections for a controller fixed + dhcp connection, 
    # one connection for an eventual NAT node, 
    # two connection for intra-switch communication.
    network_adapters_count += 5

    create_docker_template_switch(
         args.server,
         args.switch_docker,
         str(args.switch_docker+":latest"),
         adapter_count=network_adapters_count)


def update_controller_template(args, templates):

    controller_template_id = get_template_id_from_name(templates, args.controller_docker)
    if(controller_template_id is not None):
        delete_template(args.server,args.project,controller_template_id)
        print(f"old controller template {args.controller_docker} deleted")

    create_docker_template(
        args.server,
        args.controller_docker,
        args.contr_start,
        str(args.controller_docker+":latest"),
        environment=args.env_vars,
        adapters=3)


def update_botmaster_template(args, templates):

    botmaster_template_id = get_template_id_from_name(templates, args.botmaster_docker)
    if(botmaster_template_id is not None):
        delete_template(args.server,args.project,botmaster_template_id)
        print(f"old {args.botmaster_docker} template deleted")

    create_docker_template(
        args.server,
        args.botmaster_docker,
        args.botmaster_start,
        str(args.botmaster_docker+":latest"),
        environment=args.env_vars,
        adapters=2)
    

def update_templates(args, templates):
    update_switch_template(args, templates)
    update_controller_template(args, templates)
    update_botmaster_template(args, templates)
    update_generic_template(args, templates, args.attacker_docker, 'sh')
    update_generic_template(args, templates, args.victim_docker, 'sh')


if __name__ == "__main__":


    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Topology creation script")
    parser.add_argument("--project", help="GNS3 Project Name (Default is \"SmartVille\")", default="SmartVille")

    parser.add_argument("--controller_docker", help="Controller's Docker image Name (Default is \"pox-controller\")", default="pox-controller")
    parser.add_argument("--switch_docker", help="SDN Switch's Docker image Name (Default is \"openvswitch\")", default="openvswitch")
    parser.add_argument("--victim_docker", help="Victim's Docker image Name (Default is \"victim\")", default="victim")
    parser.add_argument("--attacker_docker", help="Attacker's Docker image Name (Default is \"attacker\")", default="attacker")
    parser.add_argument("--botmaster_docker", help="BotMaster's Docker image Name (Default is \"botmaster\")", default="botmaster")

    parser.add_argument("--contr_start", help="Controller's Start Command.  (Default is \"sh\")\n "+ \
                        "Could also be:  \"./pox.py samples.pretty_log smartController.smartController\"", default="sh")
    parser.add_argument("--botmaster_start", help="BotMaster's Start Command.  (Default is \"sh\")\n ", default="sh")
    parser.add_argument("--n_switches", type=int, default=4, help="Number of switches in the topology. Default: 4")
    parser.add_argument("--n_attackers", type=int, default=10, help="Number of attacker nodes in the topology. Default: 6")
    parser.add_argument("--n_victims", type=int, default=4, help="Number of victim nodes in the topology. Default: 10")


    parser.add_argument("--use_gns3_config_file", type=bool, default=True, help="Grab GNS3 Server configurations from file. Default: True")
    parser.add_argument("--gns3_config_path", type=str, default="~/.config/GNS3/2.2/gns3_server.conf", help="GNS3 Server config file, "+\
                        "Default: \"~/.config/GNS3/2.2/gns3_server.conf\"")

    parser.add_argument("--gns3_host", type=str, default="localhost", help="When not using config file. GNS3 server's hostname (Default is \"localhost\")")
    parser.add_argument("--gns3_port", type=int, default=3080, help="When not using config file. GNS3 server's port. Default: 3080")
    parser.add_argument("--gns3_auth", type=bool, default=True, help=" When not using config file. GNS3 server requires auth (Default True)")
    parser.add_argument("--gns3_username", type=str, default="admin", help="When not using config file. GNS3 server's admin username (Default is \"admin\")")
    parser.add_argument("--gns3_password", type=str, default="12345", help="When not using config file. GNS3 server's password (Default is \"12345\")")
    parser.add_argument("--env_vars", type=str, default='', help="Newline char (\\n) Env variables for node containers. E.g:  VAR_ONE=value1\\nVAR2=2\\nBLABLABLA=something. Default is empty str.")
    args = parser.parse_args()


    if args.use_gns3_config_file:
        args.server = Server(*read_local_gns3_config(args.gns3_config_path))
        args.gns3_host = args.server.addr
        args.gns3_port = args.server.port
        args.gns3_auth = args.server.auth
        args.gns3_username = args.server.user
        args.gns3_password = args.server.password
   

    gns3_server_connector = gfy.Gns3Connector(f"http://{args.gns3_host}:{args.gns3_port}", user=args.gns3_username, cred=args.gns3_password )

    args.project = resetProject(args)
    
    args.nodes_per_switch = (args.n_victims + args.n_attackers)//args.n_switches

    templates = get_all_templates(args.server)
    update_templates(args, templates)
    templates = get_all_templates(args.server)

    
    tree_topology(args, templates)
