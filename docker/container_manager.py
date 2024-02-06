import docker

CONTROLLER_IMG_NAME = 'pox-controller:latest'
ATTACKER_IMG_NAME = 'attacker:latest'
VICTIM_IMG_NAME = 'victim:latest'

process_ids = {}


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

    while True:
        user_input = input("Press '1' to launch attacks, '2' to stop attacks, or 'q' to quit: ")
        
        if user_input == '1':
            launch_attacks()
        elif user_input == '2':
            kill_attacks()
        elif user_input.lower() == 'q':
            print("Quitting...")
            break
        else:
            print("Invalid input. Please try again.")