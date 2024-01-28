from gns3util import *
from GNS3_mount_controller import *
from GNS3_mount_switch import *
from GNS3_mount_host import *
from GNS3_project_handler import *

PROJECT_NAME="sdn_project"
#these are the names of the docker templates inside gns3
CONTROLLER_NAME="pox-controller"
SWITCH_NAME="Fixed Open vSwitch"
HOST_NAME="custom-host"


def main():
    #create project
    resetProject(PROJECT_NAME)
    starTopology()

#mount host1
def starTopology():

    #mount switch
    #args: sdn project name, template name, target switch label, ip/mask, gateway
    mountSwitch(PROJECT_NAME,SWITCH_NAME,"FixedOpenvSwitch-1","192.168.1.2/24","192.168.1.1")

    #mount controller
    mountController(PROJECT_NAME,CONTROLLER_NAME)

    #mount 10 hosts and link each one to a port of the switch
    gateway = "192.168.1.1"
    ip = "192.168.1.3/24"
    mountHost(PROJECT_NAME,HOST_NAME,"custom-host ("+ip+")","FixedOpenvSwitch-1",2,ip,gateway,-250,-200)
    ip = "192.168.1.4/24"
    mountHost(PROJECT_NAME,HOST_NAME,"custom-host ("+ip+")","FixedOpenvSwitch-1",3,ip,gateway,-250,-100)
    ip = "192.168.1.5/24"
    mountHost(PROJECT_NAME,HOST_NAME,"custom-host ("+ip+")","FixedOpenvSwitch-1",4,ip,gateway,-250,0)
    ip = "192.168.1.6/24"
    mountHost(PROJECT_NAME,HOST_NAME,"custom-host ("+ip+")","FixedOpenvSwitch-1",5,ip,gateway,-250,100)
    ip = "192.168.1.7/24"
    mountHost(PROJECT_NAME,HOST_NAME,"custom-host ("+ip+")","FixedOpenvSwitch-1",6,ip,gateway,-250,200)
    ip = "192.168.1.8/24"
    mountHost(PROJECT_NAME,HOST_NAME,"custom-host ("+ip+")","FixedOpenvSwitch-1",7,ip,gateway,250,-200)
    ip = "192.168.1.9/24"
    mountHost(PROJECT_NAME,HOST_NAME,"custom-host ("+ip+")","FixedOpenvSwitch-1",8,ip,gateway,250,-100)
    ip = "192.168.1.10/24"
    mountHost(PROJECT_NAME,HOST_NAME,"custom-host ("+ip+")","FixedOpenvSwitch-1",9,ip,gateway,250,0)
    ip = "192.168.1.11/24"
    mountHost(PROJECT_NAME,HOST_NAME,"custom-host ("+ip+")","FixedOpenvSwitch-1",10,ip,gateway,250,100)
    ip = "192.168.1.12/24"
    mountHost(PROJECT_NAME,HOST_NAME,"custom-host ("+ip+")","FixedOpenvSwitch-1",11,ip,gateway,250,200)

if __name__ == "__main__":
    main()