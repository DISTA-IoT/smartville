from gns3util import *
import gns3fy as gfy


PROJECT_NAME = "sdn_project"

####################### DOCKER IMAGE NAMES:
CONTROLLER_IMG_NAME = "pox-controller"
SWITCH_IMG_NAME = "openvswitch"
VICTIM_IMG_NAME = "victim"
ATTACKER_IMG_NAME = "attacker"
NAT_IMG_NAME = "NAT"  # This is not a docker image, though...

########################### START COMMANDS:
# CONTROLLER_START_COMMAND="./pox.py samples.pretty_log smartController.switch"
CONTROLLER_START_COMMAND="sh" # DEV debugging controller...


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
    ip_pool = generateIPList(14, network, netmask)
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
        
        if idx > 3:
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
    create_docker_template_switch(server, SWITCH_IMG_NAME, str(SWITCH_IMG_NAME+":latest"),)


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

    server = Server(*read_local_gns3_config())
    gns3_server_connector = gfy.Gns3Connector("http://localhost:3080", user=server.user, cred=server.password)

    resetProject(PROJECT_NAME)

    templates = get_all_templates(server)
    update_templates()
    templates = get_all_templates(server)

    starTopology()
