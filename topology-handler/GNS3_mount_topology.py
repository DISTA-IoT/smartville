from gns3util import *
from GNS3_mount_controller import *
from GNS3_mount_switch import *
from GNS3_mount_host import *
from GNS3_project_handler import *

PROJECT_NAME="sdn_project"
# These are the names of the docker images
CONTROLLER_NAME="pox-controller"
SWITCH_NAME="openvswitch"
HOST_NAME="victim"
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
    mountSwitch(PROJECT_NAME,SWITCH_NAME,"openvswitch-1","192.168.1.2/24","192.168.1.1")

    # controller_start_command="./pox.py samples.pretty_log smartController.switch"
    controller_start_command="sh" # DEV debugging controller...

    #mount controller
    mountController(PROJECT_NAME,CONTROLLER_NAME,"openvswitch-1(192.168.1.2/24)",controller_start_command,"192.168.1.1/24","192.168.1.1")

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
    host_start_command = "sh"

    for idx, ip in enumerate(ip_pool):
        curr_img_name = HOST_NAME
        host_id = int(idx % (len(ip_pool)/2))
        HOST_LABEL = HOST_NAME+"-"+str(host_id)+"("+ip+")"

        if (i > (len(ip_pool))/2) and not half:
            half = True
            x = -300
            y = -200
        
        if half:
            curr_img_name = ATTACKER_IMG_NAME
            HOST_LABEL = ATTACKER_IMG_NAME+"-"+str(host_id)+"("+ip+")"
        
        mountHost(PROJECT_NAME,curr_img_name,HOST_LABEL,"openvswitch-1(192.168.1.2/24)",switch_port,ip,gateway,x,y, host_start_command)
        i = i+1
        y = y+100
        switch_port = switch_port+1


if __name__ == "__main__":
    main()