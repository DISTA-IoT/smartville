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
import json
import random
import yaml
import http.server
import socketserver
import json

config_dict = {
    'base_params': {
        'container_manager_replay_from_file': True,
        'browser_path': '/usr/bin/firefox',
        'terminal_issuer_path': './terminal_issuer.sh'
    },
    'intrusion_detection': {
        'eval': False,
        'device': 'cpu',
        'seed': 777,
        'ai_debug': True,
        'multi_class': True,
        'use_packet_feats': True,
        'packet_buffer_len': 1,
        'flow_buff_len': 10,
        'node_features': False,
        'metric_buffer_len': 10,
        'inference_freq_secs': 60,
        'grafana_user': 'admin',
        'grafana_password': 'admin',
        'max_kafka_conn_retries': 5,
        'curriculum': 1,
        'wb_tracking': False,
        'wb_project_name': 'SmartVille',
        'wb_run_name': 'My new run',
        'FLOWSTATS_FREQ_SECS': 5,
        'flow_idle_timeout': 10,
        'arp_timeout': 120,
        'max_buffered_packets': 5,
        'max_buffering_secs' : 5,
        'arp_req_exp_secs': 4
    }
}

containers_dict = {}
containers_ips = {}
TRAFFIC_DICT ={}
TERMINAL_ISSUER_PATH = None

start_zookeeper_command = "zookeeper-server-start.sh pox/smartController/zookeeper.properties"
start_kafka_command = "kafka-server-start.sh pox/smartController/kafka_server.properties"
start_prometheus_command = "prometheus --config.file=pox/smartController/prometheus.yml --storage.tsdb.path=pox/smartController/PrometheusLogs/"
start_grafana_command = "grafana-server -homepath /usr/share/grafana"
start_training_command = "./pox.py samples.pretty_log smartController.smartController"


def read_config(file_path):
        
    try:
        # Read configuration from YAML file
        with open(file_path, 'r') as file:
            file_confg_dict = yaml.safe_load(file)
        
        # Update default configuration with values from the file
        if file_confg_dict:
            for key in config_dict.keys():
                if key in file_confg_dict:
                    config_dict[key].update(file_confg_dict[key])
        return config_dict

    except FileNotFoundError:
        print(f"Error: Configuration file '{file_path}' not found.")
        return config_dict

    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return config_dict


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



def get_cmd_line_args(config_dict):
    args_str = ""
    for key, value in config_dict.items():
        args_str += f" --{key}={value}"
    return args_str


def start_training(controller_container):

    training_args = get_cmd_line_args(config_dict['intrusion_detection'])
    training_command = f"{start_training_command} {training_args}"
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
    # subprocess.Popen([config_dict['base_params']['browser_path'], url])
    url = "http://"+accessible_ip+":3000"  # Grafana
    subprocess.Popen([config_dict['base_params']['browser_path'], url])
    time.sleep(5)
    print('\nBrowser launched, press enter to continue\n')

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


def launch_traffic():
    response_str = ""
    args =[] 
    for container_key, container_obj in containers_dict.items():
        if 'attacker' in container_key or 'victim' in container_key:
            # Get the proper command
            command_to_run = TRAFFIC_DICT[container_key] 
            print(f"{container_key} ({container_obj.name}) will launch {command_to_run}")
            args.append(f"{container_obj.id}:{container_key}:{command_to_run}")
    print('Now launching traffic:')
    """
    for args_line in args:
        container_key, command_to_run = args_line.split(':')[1:3]
        response_str += launch_traffic_single(container_key, command_to_run) + "\n"
    return response_str
    """
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
    for args_line in args:
        container_key, command_to_run = args_line.split(':')[1:3]
        response_str += args_line.split(':')[1] + ' ' + args_line.split(':')[2]  + "\n"
    return response_str


def launch_traffic_single(container_key, command_to_run):
    container_obj = containers_dict[container_key]
    pattern, target_ip = command_to_run.split(' ')[2:4]

    exec_result = container_obj.exec_run(
                f"sh -c '{command_to_run} & echo $!'", 
                detach=True)
    if exec_result.exit_code == 0:
        log_str = f"Started {pattern} traffic from {container_key} ({container_obj.name}) to {target_ip}"
        
    else:
        log_str = f"Failed to start pattern {pattern} in {container_key} ({container_obj.name}). Exit code: {exec_result.exit_code}"
        if exec_result.output:
            print(f"Error output: {exec_result.output.decode('utf-8')}")
    
    print(log_str)
    return log_str


def verify_traffic(restart=False):
    """Verify if the pattern is actually running in the container"""
    
    for container_key, container_obj in containers_dict.items():
        if 'openvswitch' not in container_key and 'controller' not in container_key:
            # Find and kill the Python process running the replay script
            exec_result = container_obj.exec_run(
                cmd=['sh', '-c', "pgrep -f 'python3 replay.py'"],
                detach=False
            )
            answer = str(exec_result.output, 'utf-8').split('\n')
            if answer[1]  != '':
               # print(f"{container_key} ({container_obj.name}) is replaying traffic in process {answer[0]}")
                print('')
            else:
                print(f"{container_key} ({container_obj.name}) is not replaying any traffic!")
                if restart:
                    print('will now restart its traffic!')
                    launch_traffic_single(container_key, TRAFFIC_DICT[container_key])
        

def stop_traffic():
    for container_key, container_obj in containers_dict.items():
        
        try:
            # Find and kill the Python process running the replay script
            exec_result = container_obj.exec_run(
                cmd=['sh', '-c', "pkill -f 'python3 replay.py'"],
                detach=True
            )
            
            print(f"Stopped eventual python3 replay.py process from {container_key}")
            
        except Exception as e:
            print(f"Error stopping pattern  from {container_key}: {str(e)}")
        

def refresh_containers():
    global containers_dict
    global containers_ips
    # Connect to the Docker daemon
    client = docker.from_env()

    for container in client.containers.list():

        container_info = client.api.inspect_container(container.id)
        # Extract the IP address of the container from its network settings
        container_info_str = container_info['Config']['Hostname']
        container_img_name = container_info_str.split('(')[0]
        container_ip = container_info_str.split('(')[-1][:-1].split('/')[0]
        print(f'{container_img_name} is {container.name} with ip {container_ip}')
        containers_dict[container_img_name] = container
        containers_ips[container_img_name] = container_ip 
    print('\n\n\n')
            

def init_traffic_stuff():
    global TRAFFIC_DICT
    from_file = config_dict['base_params']['container_manager_replay_from_file']  
    if from_file:
        print('traffic will be replayed from file')
        # Read dictionary from a file in JSON format
        # Modify this file to adjust it to your topology and desired pattern replay dynamics.
        with open('preset_traffic_tiger.json', 'r') as file:
            TRAFFIC_DICT = json.load(file)
    else: 
        attacks = ['cc_heartbeat', 'generic_ddos', 'h_scan', 'hakai',  'torii', 'mirai', 'gafgyt', 'hajime', 'okiru', 'muhstik'] 
        bening_patterns =['echo', 'doorlock', 'hue']
        victim_ips = [item[1] for item in containers_ips.items() if 'victim' in item[0]]

        for container_key in containers_dict:
            if 'attacker' in container_key:
                TRAFFIC_DICT[container_key] = f"python3 replay.py {random.choice(attacks)} {random.choice(victim_ips)} --repeat 10"
            elif 'victim' in container_key:
                des_ips = list(set(victim_ips)  - set([containers_ips[container_key]]))
                TRAFFIC_DICT[container_key] = f"python3 replay.py {random.choice(bening_patterns)} {random.choice(des_ips)} --repeat 10" 


def create_init_labels_dict():
    init_labels_dict = {}
    for container_key, ip_addr in containers_ips.items():
        curr_label = ''
        if 'controller' not in container_key and 'switch' not in container_key:
            curr_label = TRAFFIC_DICT[container_key].split(' ')[2]
            if 'victim' in container_key:
                curr_label += ' (Bening)'
            init_labels_dict[ip_addr] = curr_label
    return init_labels_dict


if __name__ == "__main__":
    print("\n________________________________________________________________\n\n"+\
          "               SMARTVILLE Container Manager \n" +\
          "________________________________________________________________\n"+\
          "\n"+\
          "IMPORTANT:  - Parameters are read from the smartville.yaml file at project's root dir. \n" +\
          "            - Re-launch this script each time you change container status (through node restart). \n\n\n")
    # Read configuration from YAML file
    config_file_path = "../smartville.yaml"
    config_dict = read_config(config_file_path)
    # setting global vars for commodity:
    TERMINAL_ISSUER_PATH = config_dict['base_params']['terminal_issuer_path'] 
    refresh_containers() 
    init_traffic_stuff()
    
        
    PORT = 7777

    class MyRequestHandler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/refresh_containers':
                refresh_containers()
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = {'message': 'Containers refreshed!'}
                self.wfile.write(json.dumps(response).encode())
            elif self.path == '/launch_traffic':
                response_str = launch_traffic()
                self.send_response(200)
                self.send_header('Content-type', 'text/plain')  
                self.end_headers()
                self.wfile.write(response_str.encode()) 
            elif self.path == '/stop_traffic':
                stop_traffic()
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = {'message': 'Traffic stopped!'}
                self.wfile.write(json.dumps(response).encode())
            elif self.path == '/fix_traffic':
                verify_traffic(restart=True)
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = {'message': 'Traffic fixed!'}
                self.wfile.write(json.dumps(response).encode())
            elif self.path == '/labels':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response_obj = create_init_labels_dict()
                self.wfile.write(json.dumps(response_obj).encode())
            elif self.path == '/init_prices':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                with open('../cti_init_prices.json', 'r') as file:
                    response_obj = json.load(file)
                self.wfile.write(json.dumps(response_obj).encode())
            else:
                super().do_GET()

# Create a TCPServer instance with SO_REUSEADDR option
class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True  # This allows the server to reuse the address

with ReusableTCPServer(("", PORT), MyRequestHandler) as httpd:
    print(f"Serving at port {PORT}")
    httpd.serve_forever()