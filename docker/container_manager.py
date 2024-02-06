import docker
import threading


# Connect to the Docker daemon
client = docker.from_env()

# List containers
containers = client.containers.list()

CONTROLLER_IMG_NAME = 'pox-controller:latest'
ATTACKER_IMG_NAME = 'attacker:latest'
VICTIM_IMG_NAME = 'victim:latest'


def switch_case(argument):
    switch_cases = {
        'victim-0': 'python3 replay.py doorlock 192.168.1.6 --repeat 10',
        'victim-1': 'python3 replay.py echo 192.168.1.5 --repeat 10',
        'victim-2': 'python3 replay.py hue 192.168.1.3 --repeat 10',
        'victim-3': 'python3 replay.py doorlock 192.168.1.4 --repeat 10',
        'victim-4': 'python3 replay.py echo 192.168.1.8 --repeat 10',
        'attacker-0': 'python3 replay.py muhstik 192.168.1.3 --repeat 10',
        'attacker-1': 'python3 replay.py okiru  192.168.1.4 --repeat 10',
        'attacker-2': 'python3 replay.py h_scan 192.168.1.5 --repeat 10',
        'attacker-3': 'python3 replay.py cc_heartbeat 192.168.1.6 --repeat 10',
        'attacker-4': 'python3 replay.py generic_ddos 192.168.1.7 --repeat 10',
        'default': 'echo hello'
    }
    
    return switch_cases.get(argument, switch_cases['default'])


# Print container information including IP addresses
for container in containers:
    # Inspect the container to get its details, including its network settings
    container_info = client.api.inspect_container(container.id)
    
    # Extract the IP address of the container from its network settings
    container_info_str = container_info['Config']['Hostname']
    container_img_name = container_info_str.split('(')[0]
    container_ip = container_info_str.split('(')[-1][:-1]

    print("Container Image:", container_img_name)
    print("Container IP:", container_ip)
    # Get the proper command
    command_to_run = switch_case(container_img_name)
    # print(f"Running command: {command_to_run}")

    # Execute the command inside the container
    command_output = container.exec_run(command_to_run)

    # Print the output of the command
    print(command_output.output.decode('utf-8'))
