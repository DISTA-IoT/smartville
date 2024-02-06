
from gns3util import *
from GNS3_mount_switch import *
import gns3fy as gfy

def mountNAT(PROJECT_NAME,CONTROLLER_LABEL):
    server = Server(*read_local_gns3_config())
    project = get_project_by_name(server, PROJECT_NAME)

    gns3_server = gfy.Gns3Connector("http://localhost:3080", user=server.user, cred=server.password)

    templates = get_all_templates(server)
    NAT_template_id = get_template_id_from_name(templates, "NAT")
    print("NAT TEMPLATE ID: ",NAT_template_id)


    #nat_node = create_node(server, project, 0, -100, NAT_template_id, "NAT")

    # Create a new node
    nat_node = gfy.Node(
        project_id=project.id, 
        connector=gns3_server, 
        name="NAT", 
        template_id= NAT_template_id,
        x=100,
        y=-350)

    # Add the node to the project
    nat_node.create()

    print("NAT created")

    controller_id = get_node_id_by_name(server,project,CONTROLLER_LABEL)
    nat_id = get_node_id_by_name(server,project,"NAT")

    print("controller id "+str(controller_id))
    print("nat id "+str(nat_id))

    create_link(server, project,str(controller_id),1,str(nat_id),0)
    print(f"NAT: link to {CONTROLLER_LABEL} on created")
