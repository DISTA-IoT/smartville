import docker
import time

CONTROLLER_IMG_NAME = 'pox-controller:latest'
ATTACKER_IMG_NAME = 'attacker:latest'
VICTIM_IMG_NAME = 'victim:latest'

process_ids = {}
containers_dict = {}


def launch_prometheus(controller_container):
    return run_command_in_container(
        controller_container, 
        "prometheus --config.file=pox/smartController/prometheus.yml --storage.tsdb.path=pox/smartController/PrometheusLogs/")


def launch_grafana(controller_container):
    return run_command_in_container(
        controller_container, 
        "grafana-server -homepath /usr/share/grafana")


def launch_zookeeper(controller_container):
    return run_command_in_container(
        controller_container, 
        "zookeeper-server-start.sh pox/smartController/zookeeper.properties")


def launch_kafka(controller_container):
    return run_command_in_container(
        controller_container, 
        "kafka-server-start.sh pox/smartController/kafka_server.properties")

def delete_kafka_logs(controller_container):
    return run_command_in_container(
        controller_container, 
        "rm -rf /opt/kafka/logs")

def launch_controller_processes(controller_container):
    if launch_prometheus(controller_container):
        print('Prometheus launched on controller!')
        time.sleep(1)
        if launch_grafana(controller_container):
            print('Grafana launched on controller!')
            time.sleep(1)
            if launch_zookeeper(controller_container):
                print('Zookeeper launched on controller!')
                time.sleep(1)
                if launch_kafka(controller_container):
                    print('Kafka launched on controller!')


def launch_producers():
    for i in range(0,5):
        curr_container = containers_dict['victim-'+str(i)]
        run_command_in_container(curr_container, "python3 producer.py")
        curr_container = containers_dict['attacker-'+str(i)]
        run_command_in_container(curr_container, "python3 producer.py")


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


def launch_attacks():
    print('Launching attacks:')
    for container in containers:
        container_info = client.api.inspect_container(container.id)
        
        # Extract the IP address of the container from its network settings
        container_info_str = container_info['Config']['Hostname']
        container_img_name = container_info_str.split('(')[0]
        # container_ip = container_info_str.split('(')[-1][:-1]

        print(container_img_name)
        # print("Container IP:", container_ip)
        # Get the proper command
        command_to_run = switch_case(container_img_name)
        # Execute the command inside the container
        pid = run_command_in_container(container=container, command=command_to_run)
        print(f'Process id {pid}')
        process_ids[container] = pid


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

    while True:
        user_input = input("Press '1' to launch attacks, " +\
                            "'2' to stop attacks, " +\
                            "'3' to launch controller services, "+\
                            "'4' to launch producers, "+\
                            "or 'q' to quit: ")
        
        if user_input == '1':
            launch_attacks()
        elif user_input == '2':
            kill_attacks()
        elif user_input == '3':
            launch_controller_processes(containers_dict['pox-controller-1'])
        elif user_input == '4':
            launch_producers()
        elif user_input == 'pro':
            print(launch_prometheus(containers_dict['pox-controller-1']))
        elif user_input == 'gra':
            print(launch_grafana(containers_dict['pox-controller-1']))
        elif user_input == 'zoo':
            print(launch_zookeeper(containers_dict['pox-controller-1']))
        elif user_input == 'kaf':
            print(launch_kafka(containers_dict['pox-controller-1']))
        elif user_input == 'dkl':
            print(delete_kafka_logs(containers_dict['pox-controller-1']))
        elif user_input.lower() == 'q':
            print("Quitting...")
            break
        else:
            print("Invalid input. Please try again.")