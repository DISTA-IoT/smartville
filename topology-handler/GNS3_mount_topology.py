from gns3util import *
from GNS3_mount_controller import *
from GNS3_mount_switch import *
from GNS3_mount_host import *
from GNS3_project_handler import *
from GNS3_mount_nat import *

PROJECT_NAME="sdn_project"
#these are the names of the docker images
CONTROLLER_NAME="pox-controller"
SWITCH_NAME="openvswitch"
HOST_NAME="custom-host"


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

    controller_start_command="./pox.py --verbose samples.pretty_log customScript.l3_learning_mod"
    #mount controller
    controller_name=mountController(PROJECT_NAME,CONTROLLER_NAME,"openvswitch-1",controller_start_command,"192.168.1.1/24","192.168.1.1")
    print("controllername ",controller_name)
    mountNAT(PROJECT_NAME,controller_name)

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

    for ip in ip_pool:
        if (i > (len(ip_pool))/2) and not half:
            half = True
            x = -300
            y = -200
        if half:
            host_start_command = "python3 icmp_flood.py"
        if not half:
            host_start_command = "python3 msg_send_sim.py"

        HOST_LABEL = "custom-host ("+ip+")"
        
        mountHost(PROJECT_NAME,HOST_NAME,HOST_LABEL,"openvswitch-1",switch_port,ip,gateway,x,y,host_start_command)
        i = i+1
        y = y+100
        switch_port = switch_port+1


if __name__ == "__main__":
    main()