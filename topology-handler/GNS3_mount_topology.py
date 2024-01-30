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
ATTACKER_IMG_NAME="attacker"

def main():
    #create project
    resetProject(PROJECT_NAME)
    starTopology()

#return each time the previous ip +1 

def generateIPList(ip_nr,network,netmask):
    ip_pool = []
    for i in range(3,ip_nr+3):
        net_ip = network[:-1]
        ip = net_ip+str(i)+netmask
        ip_pool.append(ip)
        print("IP ADDED TO LIST: ",ip)
    return ip_pool


def starTopology():

    #mount switch
    #args: sdn project name, template name, target switch label, ip/mask, gateway
    mountSwitch(PROJECT_NAME,SWITCH_NAME,"FixedOpenvSwitch-1","192.168.1.2/24","192.168.1.1")

    #mount controller
    mountController(PROJECT_NAME,CONTROLLER_NAME)

    #mount 10 hosts and link each one to a port of the switch
    gateway = "192.168.1.1"
    switch_port = 3
    network = "192.168.1.0"
    netmask = "/24"
    #generate pool of ip addresses for specified network (es. 192.168.1.0)
    ip_pool = generateIPList(10,network,netmask)
    x = 300
    y = -200
    i = 1
    half = False

    for ip in ip_pool:
        curr_img_name = HOST_NAME
        HOST_LABEL = HOST_NAME+" ("+ip+")"

        if (i > (len(ip_pool))/2) and not half:
            half = True
            x = -300
            y = -200
        
        if half:
            curr_img_name = ATTACKER_IMG_NAME
            HOST_LABEL = ATTACKER_IMG_NAME+" ("+ip+")"
        
        mountHost(PROJECT_NAME,curr_img_name,HOST_LABEL,"FixedOpenvSwitch-1",switch_port,ip,gateway,x,y)
        i = i+1
        y = y+100
        switch_port = switch_port+1


if __name__ == "__main__":
    main()