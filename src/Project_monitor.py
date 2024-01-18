from gns3fy import Gns3Connector, Project
from tabulate import tabulate

#indirizzo del server gns3
gns3_server="http://192.168.105.130:80"




#nome del progetto gns3
project_name="myProject_matteo_final"

#creazione di una connessione al server gns3
gns3=Gns3Connector(gns3_server)

#verifico se il progetto esiste gia
project=Project(name=project_name,connector=gns3)
project.get()

#ottengo informazioni sul progetto
project_summary=gns3.projects_summary(is_print=False)
print(
    tabulate(
        gns3.projects_summary(is_print=False),
        headers=[
            "Project Name",
            "Project ID",
            "Total Nodes",
            "Total Links",
            "Status",
        ],
    )
)
#ottengo informazioni sui nodi
nodes_summary=project.nodes_summary(is_print=False)
print(
    tabulate(nodes_summary,headers=["Node","Status","Console Port","ID"])
)
#ottengo informazioni sui collegamenti
links_summary=project.links_summary(is_print=False)
print(
    tabulate(links_summary,headers=["sdn-switch","switch-port","my-router","router-port","iot-device-n6","iot6-port","iot-device-n7","iot7-port","iot-device-n8","iot8-port"])
)
