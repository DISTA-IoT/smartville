from gns3util import *

def resetProject(PROJECT_NAME):
    #delete old project and create new one
    server = Server(*read_local_gns3_config())
    delete_project(server,PROJECT_NAME)
    print("Project deleted")
    create_project(server,PROJECT_NAME,1000,1000)
    print("New project created")
    project = get_project_by_name(server, PROJECT_NAME)
    open_project_if_closed(server, project)

