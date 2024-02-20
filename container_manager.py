import docker
import threading
import time
import subprocess

TERMINAL_ISSUER_PATH = './terminal_issuer.sh'  # MAKE IT EXECUTABLE WITH chmod +x terminal_issuer.sh

BROWSER_PATH = '/usr/bin/brave-browser'  # Change to your commodity browser
CONTROLLER_IMG_NAME = 'pox-controller:latest'
ATTACKER_IMG_NAME = 'attacker:latest'
VICTIM_IMG_NAME = 'victim:latest'

CURRICULUM=0
KNWON_TRAFFIC_NODES = []
TRAINING_ZDA_NODES = []
TEST_ZDA_NODES = []  

process_ids = {}
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
    # Build the command to execute your Bash script with its arguments
    command = [TERMINAL_ISSUER_PATH, f"{controller_container.id}:KAFKA:{start_kafka_command}"]
    launch_detached_command(command)


def launch_brower_consoles(controller_container):
    ifconfig_output = run_command_in_container(
        controller_container, 
        "ifconfig")
    accessible_ip = ifconfig_output.split('eth1')[1].split('inet ')[1].split(' ')[0]
    url = "http://"+accessible_ip+":9090"  # Prometheus
    subprocess.call([BROWSER_PATH, url])
    url = "http://"+accessible_ip+":3000"  # Grafana
    subprocess.call([BROWSER_PATH, url])


def delete_kafka_logs(controller_container):
    return run_command_in_container(
        controller_container, 
        "rm -rf /opt/kafka/logs")


def launch_controller_processes(controller_container):
    launch_prometheus_detached(controller_container)
    print('Prometheus launched on controller! please wait...')
    time.sleep(1)
    launch_grafana_detached(controller_container)
    print('Grafana launched on controller! please wait...')
    time.sleep(1)
    launch_zookeeper_detached(controller_container)
    print('Zookeeper launched on controller! please wait...')
    time.sleep(1)
    launch_kafka_detached(controller_container)
    print('Kafka launched on controller! please wait...')


def launch_metrics():
    for i in range(0,5):
        curr_container = containers_dict['victim-'+str(i)]
        # Build the command to execute your Bash script with its arguments
        command = [TERMINAL_ISSUER_PATH, f"{curr_container.id}:victim-{i}-METRICS:python3 producer.py"]
        launch_detached_command(command)


def run_command_in_container(container, command):
    # Run the command in the container shell to obtain the PID
    exec_result = container.exec_run(f"sh -c '{command} & echo $!'")
    pid = exec_result.output.decode("utf-8").strip()
    return pid


def kill_attacks():
    print('Killing attacks...')
    for container, pid in process_ids.items():
        container_info = client.api.inspect_container(container.id)
        container.exec_run(f"kill {pid}")
        print(f"Killed process {pid} in {container_info['Config']['Hostname']}")


def switch_case(argument):
    switch_cases = {
        # Known Bening:
        'victim-0': 'python3 replay.py doorlock 192.168.1.6 --repeat 10',
        'victim-1': 'python3 replay.py echo 192.168.1.5 --repeat 10',
        'victim-2': 'python3 replay.py hue 192.168.1.3 --repeat 10',
        'victim-3': 'python3 replay.py doorlock 192.168.1.4 --repeat 10',
        # Known Attacks:
        'attacker-4': 'python3 replay.py cc_heartbeat 192.168.1.3 --repeat 10',
        'attacker-5': 'python3 replay.py generic_ddos  192.168.1.4 --repeat 10',
        'attacker-6': 'python3 replay.py h_scan 192.168.1.5 --repeat 10',
        # Training ZdAs:
        'attacker-7': 'python3 replay.py hakai 192.168.1.6 --repeat 10',
        'attacker-8': 'python3 replay.py torii 192.168.1.4 --repeat 10',
        'attacker-9': 'python3 replay.py mirai 192.168.1.5 --repeat 10',
        'attacker-10': 'python3 replay.py gafgyt 192.168.1.3 --repeat 10',
        # Test ZdAs:
        'attacker-11': 'python3 replay.py hajime 192.168.1.6 --repeat 10',
        'attacker-12': 'python3 replay.py okiru 192.168.1.5 --repeat 10',
        'attacker-13': 'python3 replay.py muhstik 192.168.1.3 --repeat 10',

        'default': 'echo hello'
    }
    return switch_cases.get(argument, switch_cases['default'])


def send_known_traffic():
    args = []

    for container_img_name in KNWON_TRAFFIC_NODES:
        container = containers_dict[container_img_name]

        command_to_run = switch_case(container_img_name)
        if command_to_run != 'echo hello':
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
        command_to_run = switch_case(container_img_name)
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
        command_to_run = switch_case(container_img_name)
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
        command_to_run = switch_case(container_img_name)
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


def stop_nodes():
    """
    Have no time? Do it by hand on GNS3, it is much faster...
    """
    for container_img_name, container in containers_dict.items():
        if 'attacker' in container_img_name or 'victim' in container_img_name:
            print(f'gracefully stoping {container_img_name}')
            container.stop()


def start_nodes():

    for container_img_name, container in containers_dict.items():
        if 'attacker' in container_img_name or 'victim' in container_img_name:
            print(f'starting {container_img_name}')
            container.start()


if __name__ == "__main__":

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

        KNWON_TRAFFIC_NODES = [
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

        KNWON_TRAFFIC_NODES = [
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

        KNWON_TRAFFIC_NODES = [
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


    user_input = input("Press '1' to send known traffic, \n" +\
                       "'2' to send training zdas \n" +\
                       "'3' to send test zdas \n" +\
                        "'4' to send metrics, \n" +\
                        "'5' to launch controller services, \n"+\
                        "or 'q' to quit: ")
    
    if user_input == '1':
        send_known_traffic()
    if user_input == '2':
        send_training_zdas()
    if user_input == '3':
        send_test_zdas()
    elif user_input == '4':
        launch_metrics()
    elif user_input == '5':
        launch_controller_processes(containers_dict['pox-controller-1'])
    elif user_input == 'pro':
        print(launch_prometheus_detached(containers_dict['pox-controller-1']))
    elif user_input == 'gra':
        print(launch_grafana(containers_dict['pox-controller-1']))
    elif user_input == 'url':
        print(launch_brower_consoles(containers_dict['pox-controller-1']))
    elif user_input == 'zoo':
        print(launch_zookeeper_detached(containers_dict['pox-controller-1']))
    elif user_input == 'kaf':
        print(launch_kafka(containers_dict['pox-controller-1']))
    elif user_input == 'dkl':
        print(delete_kafka_logs(containers_dict['pox-controller-1']))
    elif user_input == 'stop':
        stop_nodes()
    elif user_input == 'start':
        start_nodes()