
from gns3util import *
from GNS3_mount_switch import *
from GNS3_mount_topology import SWITCH_NAME
import gns3fy as gfy

def mountNAT(PROJECT_NAME,CONTROLLER_LABEL):
    server = Server(*read_local_gns3_config())
    project = get_project_by_name(server, PROJECT_NAME)

    gns3_server = gfy.Gns3Connector("http://localhost:3080")

    interfaces = []
    SWITCH_LABEL = "Edge-openvswitch"

    interfaces.extend(["eth1", "eth2"])
    mountSwitchDHCP(PROJECT_NAME, SWITCH_NAME,SWITCH_LABEL, interfaces)
    
    templates = get_all_templates(server)
    NAT_template_id = get_template_id_from_name(templates, "NAT")
    print("NAT TEMPLATE ID: ",NAT_template_id)

    # Create a new node
    nat_node = gfy.Node(
        project_id=project.id, 
        connector=gns3_server, 
        name="NAT", 
        template_id= NAT_template_id,
        x=0,
        y=-350)
    
    # Add the node to the project
    nat_node.create()

    print("NAT created")

    edge_switch_id = get_node_id_by_name(server,project,SWITCH_LABEL)
    controller_id = get_node_id_by_name(server,project,CONTROLLER_LABEL)
    nat_id = get_node_id_by_name(server,project,"NAT")

    print("switch id "+str(edge_switch_id))
    print("controller id "+str(controller_id))
    print("nat id "+str(nat_id))

    create_link(server, project,str(edge_switch_id),1,str(nat_id),0)
    create_link(server, project,str(edge_switch_id),2,str(controller_id),1)
    print(f"NAT: link to {SWITCH_LABEL} on created")
    print(f"{SWITCH_LABEL}: link to {CONTROLLER_LABEL} created")