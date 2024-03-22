# This file is part of the "Smartville" project.
# Copyright (c) 2024 University of Insubria
# Licensed under the Apache License 2.0.
# SPDX-License-Identifier: Apache-2.0
# For the full text of the license, visit:
# https://www.apache.org/licenses/LICENSE-2.0

# Smartville is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# Apache License 2.0 for more details.

# You should have received a copy of the Apache License 2.0
# along with Smartville. If not, see <https://www.apache.org/licenses/LICENSE-2.0>.

# Additional licensing information for third-party dependencies
# used in this file can be found in the accompanying `NOTICE` file.
import docker
import threading
import time
import subprocess
import argparse
import json
import random

TERMINAL_ISSUER_PATH = './terminal_issuer.sh'  # MAKE IT EXECUTABLE WITH chmod +x terminal_issuer.sh
BROWSER_PATH = None


KNOWN_TRAFFIC_NODES = []
TRAINING_ZDA_NODES = []
TEST_ZDA_NODES = []  



containers_dict = {}

start_zookeeper_command = "zookeeper-server-start.sh pox/smartController/zookeeper.properties"
start_kafka_command = "kafka-server-start.sh pox/smartController/kafka_server.properties"
start_prometheus_command = "prometheus --config.file=pox/smartController/prometheus.yml --storage.tsdb.path=pox/smartController/PrometheusLogs/"
start_grafana_command = "grafana-server -homepath /usr/share/grafana"
start_training_command = "./pox.py samples.pretty_log smartController.smartController"


# Function to continuously print output of a command
def print_output(container, command, thread_name):
    # Execute the command in the container and stream the output
    return_tuple = container.exec_run(command, stream=True, tty=True, stdin=True)
    for line in return_tuple[1]:
        print(thread_name+": "+line.decode().strip())  # Print the output line by line


def launch_detached_command(command):
    # Run the command on a new pseudo TTY
    try:
        # Run the command and capture the output
        output = subprocess.check_output(command, stderr=subprocess.STDOUT)
        print(output.decode('utf-8'))  # Decode the output bytes to UTF-8 and print it
    except subprocess.CalledProcessError as e:
        # Handle errors if the command exits with a non-zero status
        print("Error:", e)


def launch_prometheus(controller_container):
    print(run_command_in_container(controller_container, "python3 pox/smartController/set_prometheus.py"))
    time.sleep(1)
    output_thread = threading.Thread(
        target=print_output, 
        args=(controller_container, start_prometheus_command, 'PROMETHEUS'))
    output_thread.start()


def launch_prometheus_detached(controller_container):
    print(run_command_in_container(controller_container, "python3 pox/smartController/set_prometheus.py"))
    time.sleep(1)
    # Build the command to execute your Bash script with its arguments
    command = [TERMINAL_ISSUER_PATH, f"{controller_container.id}:PROMETHEUS:{start_prometheus_command}"]
    launch_detached_command(command)


def launch_grafana(controller_container):
    output_thread = threading.Thread(
        target=print_output, 
        args=(controller_container, start_grafana_command, 'GRAFANA'))
    output_thread.start()


def launch_grafana_detached(controller_container):
    # Build the command to execute your Bash script with its arguments
    command = [TERMINAL_ISSUER_PATH, f"{controller_container.id}:GRAFANA:{start_grafana_command}"]
    launch_detached_command(command)
    print('Waiting for Grafana to start...')
    time.sleep(10)
    # print('Linking Grafana to Prometheus...')
    # print(run_command_in_container(controller_container, "python3 pox/smartController/link_grafana_to_prometheus.py"))
    time.sleep(1)


def launch_zookeeper(controller_container):
    output_thread = threading.Thread(
        target=print_output, 
        args=(controller_container, start_zookeeper_command, 'ZOOKEEPER'))
    output_thread.start()


def launch_zookeeper_detached(controller_container):
    # Build the command to execute your Bash script with its arguments
    command = [TERMINAL_ISSUER_PATH, f"{controller_container.id}:ZOOKEEPER:{start_zookeeper_command}"]
    launch_detached_command(command)


def launch_kafka(controller_container):
    output_thread = threading.Thread(
        target=print_output, 
        args=(controller_container, start_kafka_command, 'KAFKA'))
    output_thread.start()


def launch_kafka_detached(controller_container):
    delete_kafka_logs(controller_container)
    time.sleep(2)
    delete_kafka_logs(controller_container)
    time.sleep(1)
    # Build the command to execute your Bash script with its arguments
    command = [TERMINAL_ISSUER_PATH, f"{controller_container.id}:KAFKA:{start_kafka_command}"]
    launch_detached_command(command)


def start_training(controller_container):
    training_command = start_training_command
    print(f"Training command: {training_command}")
    print(f"Now launching training")
    command = [TERMINAL_ISSUER_PATH, f"{controller_container.id}:TRAINING:{training_command}"]
    launch_detached_command(command)


def launch_brower_consoles(controller_container):
    ifconfig_output = run_command_in_container(
        controller_container, 
        "ifconfig")
    accessible_ip = ifconfig_output.split('eth1')[1].split('inet ')[1].split(' ')[0]
    # url = "http://"+accessible_ip+":9090"  # Prometheus
    # subprocess.call([BROWSER_PATH, url])
    url = "http://"+accessible_ip+":3000"  # Grafana
    subprocess.call([BROWSER_PATH, url])


def delete_kafka_logs(controller_container):
    print('Deleting Kafka logs...')
    return run_command_in_container(
        controller_container, 
        "rm -rf /opt/kafka/logs")


def launch_controller_processes(controller_container):
    launch_zookeeper_detached(controller_container)
    print('Zookeeper launched on controller! please wait...')
    time.sleep(1)
    launch_prometheus_detached(controller_container)
    print('Prometheus launched on controller! please wait...')
    time.sleep(1)
    launch_grafana_detached(controller_container)
    print('Grafana launched on controller! please wait...')
    time.sleep(1)
    launch_kafka_detached(controller_container)
    print('Kafka launched on controller! please wait...')
    time.sleep(1)
    print('Launching Grafanfa dashboard on host...')
    launch_brower_consoles(controller_container)


def launch_metrics():
    for container_name, container_obj in containers_dict.items():
        if container_name.startswith('victim'):
            # Build the command to execute your Bash script with its arguments
            command = [TERMINAL_ISSUER_PATH, f"{container_obj.id}:{container_name}-METRICS:python3 producer.py"]
            launch_detached_command(command)


def run_command_in_container(container, command):
    # Run the command in the container shell to obtain the PID
    exec_result = container.exec_run(f"sh -c '{command} & echo $!'")
    pid = exec_result.output.decode("utf-8").strip()
    return pid


def launch_traffic(from_file=False):
    args = []

    if from_file:
        # Read dictionary from a file in JSON format
        # Modify this file to adjust it to your topology and desired pattern replay dynamics.
        with open('preset_traffic.json', 'r') as file:
            replay_dictionary = json.load(file)
        print("Replay configuration from file.")
        for container in containers:
            container_info = client.api.inspect_container(container.id)
            # Extract the IP address of the container from its network settings
            container_info_str = container_info['Config']['Hostname']
            container_img_name = container_info_str.split('(')[0]
            container_ip = container_info_str.split('(')[-1][:-1]
            container_ip = container_ip.split('/')[0]
            # print(container_img_name)
            # print("Container IP:", container_ip)
            if 'attacker' in container_img_name or 'victim' in container_img_name:
                # Get the proper command
                command_to_run = replay_dictionary[container_img_name] 
                print(f"{container_img_name} ({container_ip}) will launch {command_to_run}")
                args.append(f"{container.id}:{container_info_str}:{command_to_run}")
    else: 
        print('Random traffic configuration.')
        victims = [] 
        attackers = [] 
        attacks = ['cc_heartbeat', 'generic_ddos', 'h_scan', 'hakai',  'torii', 'mirai', 'gafgyt', 'hajime', 'okiru', 'muhstik'] 
        bening_patterns =['echo', 'doorlock', 'hue'] 
        for container in containers:
            container_info = client.api.inspect_container(container.id)
            # Extract the IP address of the container from its network settings
            container_info_str = container_info['Config']['Hostname']
            container_img_name = container_info_str.split('(')[0]
            container_ip = container_info_str.split('(')[-1][:-1]
            container_ip = container_ip.split('/')[0]
            
            if 'attacker' in container_img_name:
                attackers.append((container.id,container_img_name, container_ip, container_info_str)) 
            elif 'victim' in container_img_name:
                victims.append((container.id,container_img_name, container_ip, container_info_str)) 
            
        victim_ips =[ip for (_, _, ip, _) in victims]

        for attacker_tuple in attackers:
            # Get the proper command
            command_to_run = f"python3 replay.py {random.choice(attacks)} {random.choice(victim_ips)} --repeat 10"
            print(f"{attacker_tuple[1]} ({attacker_tuple[2]}) will launch {command_to_run}")
            args.append(f"{attacker_tuple[0]}:{attacker_tuple[3]}:{command_to_run}")

        for victim_tuple in victims:
            # Get the proper command
            curr_dests = list(set(victim_ips) - set([victim_tuple[2]]))
            command_to_run = f"python3 replay.py {random.choice(bening_patterns)} {random.choice(curr_dests)} --repeat 10"
            print(f"{victim_tuple[1]} ({victim_tuple[2]}) will launch {command_to_run}")
            args.append(f"{victim_tuple[0]}:{victim_tuple[3]}:{command_to_run}")

    
    print('Now launching traffic:')
    # Build the command to execute your Bash script with its arguments
    command = [TERMINAL_ISSUER_PATH] + args

    # Run the command
    try:
        # Run the command and capture the output
        output = subprocess.check_output(command, stderr=subprocess.STDOUT)
        print(output.decode('utf-8'))  # Decode the output bytes to UTF-8 and print it
    except subprocess.CalledProcessError as e:
        # Handle errors if the command exits with a non-zero status
        print("Error:", e)
    

if __name__ == "__main__":


    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Container Manager script")
    parser.add_argument(
        "--browser", 
        help="Browser Exetubale Path (Default is /usr/bin/firefox, change to your favourite)", 
        default="/usr/bin/firefox")
    
    args = parser.parse_args()

    
    BROWSER_PATH = args.browser


    # Connect to the Docker daemon
    client = docker.from_env()
    # List containers
    containers = client.containers.list()

    for container in client.containers.list():

        container_info = client.api.inspect_container(container.id)
        # Extract the IP address of the container from its network settings
        container_info_str = container_info['Config']['Hostname']
        container_img_name = container_info_str.split('(')[0]

        containers_dict[container_img_name] = container

    user_input = input("SmartVille Container Maganer \n" +\
                       "Plase input a character and type enter. \n" +\
                       "'c' to launch controller services. This will: \n"+\
                       " |---1. launch zookeeper service,   ('zoo' option) \n"+\
                       " |---2. config and launch prometheus service,   ('pro' option) \n"+\
                       " |---3. config and launch grafana service ,     ('gra' option) \n"+\
                       " |---4. delete kafka logs,                      ('dkl' option) \n"+\
                       " |---5. config and launch kafka,                ('kaf' option) \n"+\
                       " |---6. launch grafana dashboard on browser,    ('dash' option) \n"+\
                       " ________________________________________________________________\n"+\
                       "\n"+\
                       "'s' to send all traffic patterns from nodes.\n"+\
                       " ________________________________________________________________"+\
                       "\n"+\
                       "'t' to initiate training at controller.\n"+\
                       " ________________________________________________________________"+\
                       "\n"+\
                       "'m' to send node features from all nodes, \n"+\
                       " ________________________________________________________________"+\
                       "\n"+\
                       "'q' to quit. \n"+\
                       " ________________________________________________________________\n"+\
                       " Your input: ")
                   
    
    if user_input == 's':
        launch_traffic()
    elif user_input == 'm':
        launch_metrics()
    elif user_input == 'c':
        launch_controller_processes(containers_dict['pox-controller-1'])
    elif user_input == 't':
        start_training(containers_dict['pox-controller-1'])
    elif user_input == 'pro':
        print(launch_prometheus_detached(containers_dict['pox-controller-1']))
    elif user_input == 'gra':
        print(launch_grafana_detached(containers_dict['pox-controller-1']))
    elif user_input == 'dash':
        print(launch_brower_consoles(containers_dict['pox-controller-1']))
    elif user_input == 'zoo':
        print(launch_zookeeper_detached(containers_dict['pox-controller-1']))
    elif user_input == 'kaf':
        print(launch_kafka_detached(containers_dict['pox-controller-1']))
    elif user_input == 'dkl':
        print(delete_kafka_logs(containers_dict['pox-controller-1']))
    elif user_input == 'q':
        print('Bye Bye!')
        exit
    else:
        print=('Invalid Option!')