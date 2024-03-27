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

TERMINAL_ISSUER_PATH = './terminal_issuer.sh'  # MAKE IT EXECUTABLE WITH chmod +x terminal_issuer.sh
BROWSER_PATH = None


CURRICULUM=None # For training and labelling purposes. Must be set according to the labelling in the controller. 
KNOWN_TRAFFIC_NODES = []
TRAINING_ZDA_NODES = []
TEST_ZDA_NODES = []  



containers_dict = {}

start_zookeeper_command = "zookeeper-server-start.sh pox/smartController/zookeeper.properties"
start_kafka_command = "kafka-server-start.sh pox/smartController/kafka_server.properties"
start_prometheus_command = "prometheus --config.file=pox/smartController/prometheus.yml --storage.tsdb.path=pox/smartController/PrometheusLogs/"
start_grafana_command = "grafana-server -homepath /usr/share/grafana"


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


def pattern_replay_commands(argument):
    """
    Modify this function to adjust it to your topolgy and desired pattern replay dynamics.
    """
    switch_cases = {
        # Known Bening:
        'victim-0': 'python3 replay.py doorlock 192.168.1.6 --repeat 10',
        'victim-1': 'python3 replay.py echo 192.168.1.5 --repeat 10',
        'victim-2': 'python3 replay.py hue 192.168.1.3 --repeat 10',
        'victim-3': 'python3 replay.py doorlock 192.168.1.4 --repeat 10',
        'attacker-4': 'python3 replay.py cc_heartbeat 192.168.1.3 --repeat 10',
        'attacker-5': 'python3 replay.py generic_ddos  192.168.1.4 --repeat 10',
        'attacker-6': 'python3 replay.py h_scan 192.168.1.5 --repeat 10',
        'attacker-7': 'python3 replay.py hakai 192.168.1.6 --repeat 10',
        'attacker-8': 'python3 replay.py torii 192.168.1.4 --repeat 10',
        'attacker-9': 'python3 replay.py mirai 192.168.1.5 --repeat 10',
        'attacker-10': 'python3 replay.py gafgyt 192.168.1.3 --repeat 10',
        'attacker-11': 'python3 replay.py hajime 192.168.1.6 --repeat 10',
        'attacker-12': 'python3 replay.py okiru 192.168.1.5 --repeat 10',
        'attacker-13': 'python3 replay.py muhstik 192.168.1.3 --repeat 10',
        'default': None
    }
    return switch_cases.get(argument, switch_cases['default'])


def send_known_traffic():
    args = []

    for container_img_name in KNOWN_TRAFFIC_NODES:
        container = containers_dict[container_img_name]

        command_to_run = pattern_replay_commands(container_img_name)
        if command_to_run:
            args.append(f"{container.id}:{container_img_name}:{command_to_run}")

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


def send_training_zdas():
    
    args = []

    for container_img_name in TRAINING_ZDA_NODES:

        curr_container = containers_dict[container_img_name]
        command_to_run = pattern_replay_commands(container_img_name)
        args.append(f"{curr_container.id}:{container_img_name} (Training ZdA):{command_to_run}")
    
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


def send_test_zdas():

    args = []

    for container_img_name in TEST_ZDA_NODES:

        curr_container = containers_dict[container_img_name]
        command_to_run = pattern_replay_commands(container_img_name)
        args.append(f"{curr_container.id}:{container_img_name} (Test ZdA):{command_to_run}")
    
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


def launch_traffic():

    args = []

    for container in containers:
        container_info = client.api.inspect_container(container.id)
        # Extract the IP address of the container from its network settings
        container_info_str = container_info['Config']['Hostname']
        container_img_name = container_info_str.split('(')[0]
        # container_ip = container_info_str.split('(')[-1][:-1]
        # print(container_img_name)
        # print("Container IP:", container_ip)
        # Get the proper command
        command_to_run = pattern_replay_commands(container_img_name)
        if command_to_run != 'echo hello':
            args.append(f"{container.id}:{container_info_str}:{command_to_run}")

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


def send_all_traffic():
    send_known_traffic()
    send_training_zdas()
    send_test_zdas()
    

if __name__ == "__main__":


    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Container Manager script")
    parser.add_argument(
        "--curriculum", 
        help="The curriculum to divide known, train unknown and test unkown patterns for ZdA Attack detection  (Default is 1)", 
        default=1)
    parser.add_argument(
        "--browser", 
        help="Browser Exetubale Path (Default is /usr/bin/firefox, change to your favourite)", 
        default="/usr/bin/firefox")
    
    args = parser.parse_args()

    CURRICULUM = args.curriculum
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

    
    if CURRICULUM == 0:

        KNOWN_TRAFFIC_NODES = [
        'victim-0',
        'victim-1',
        'victim-2',
        'victim-3',
        'attacker-4',
        'attacker-5',
        'attacker-6']  

        TRAINING_ZDA_NODES = [
        'attacker-7',
        'attacker-8',
        'attacker-9',
        'attacker-10']  

        TEST_ZDA_NODES = [
        'attacker-11',
        'attacker-12',
        'attacker-13']  

    elif CURRICULUM == 1:

        KNOWN_TRAFFIC_NODES = [
        'victim-0',
        'victim-1',
        'victim-2',
        'victim-3',
        'attacker-7',
        'attacker-8',
        'attacker-12']  

        TRAINING_ZDA_NODES = [
        'attacker-4',
        'attacker-5',
        'attacker-9',
        'attacker-10']  


        TEST_ZDA_NODES = [
        'attacker-11',
        'attacker-6',
        'attacker-13']  

    elif CURRICULUM == 2:

        KNOWN_TRAFFIC_NODES = [
        'victim-0',
        'victim-1',
        'victim-2',
        'victim-3',
        'attacker-13',
        'attacker-11',
        'attacker-9']  

        TRAINING_ZDA_NODES = [
        'attacker-4',
        'attacker-5',
        'attacker-12',
        'attacker-6']  

        TEST_ZDA_NODES = [
        'attacker-7',
        'attacker-10',
        'attacker-8']  


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
                       "'s' to send all traffic patterns from nodes, This will: \n"+\
                       " |---1. send known traffic according to curriculum,   ('known' option) \n"+\
                       " |---2. send training zdas,                           ('zda1' option) \n"+\
                       " |---3. send test zdas,                               ('zda2' option) \n"+\
                       " ________________________________________________________________"+\
                       "\n"+\
                       "'m' to send node features from all nodes, \n"+\
                       " ________________________________________________________________"+\
                       "\n"+\
                       "'q' to quit. \n"+\
                       " ________________________________________________________________\n"+\
                       " Your input: ")
                   
    
    if user_input == 's':
        send_all_traffic()
    elif user_input == 'known':
        send_known_traffic()
    elif user_input == 'zda1':
        send_training_zdas()
    if user_input == 'zda2':
        send_test_zdas()
    elif user_input == 'm':
        launch_metrics()
    elif user_input == 'c':
        launch_controller_processes(containers_dict['pox-controller-1'])
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
        exit