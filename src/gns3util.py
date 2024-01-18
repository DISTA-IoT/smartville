"""Functions to automate network topology creation/manipulation etc. using the GNS3 API."""

import base64
import configparser
import hashlib
import ipaddress
import json
import os
import re
import resource
import time
import warnings

from collections import namedtuple
from telnetlib import Telnet
from typing import Any, List, Dict, Optional, Pattern

import requests


Server = namedtuple("Server", ("addr", "port", "auth", "user", "password"))
Project = namedtuple("Project", ("name", "id", "grid_unit"))
Item = namedtuple("Item", ("name", "id"))
Position = namedtuple("Position", ("x", "y"))

def get_node_status(hostname, port, project, node_id):
    """
    Restituisce lo stato di un nodo in GNS3.

    Args:
        server: L'oggetto `Gns3Server` che rappresenta il server GNS3.
        project: Il nome del progetto in cui si trova il nodo.
        node_id: L'ID univoco del nodo.

    Returns:
        Lo stato del nodo, come stringa.
    """

    # Crea la richiesta GET.
    get_node_status_url = f"http://{hostname}:{port}/v2/projects/{project}/nodes/{node_id}/status"

    # Invia la richiesta.
    response = requests.get(get_node_status_url)

    # Controlla lo stato della risposta.
    if response.status_code != 200:
        raise Exception(f"Errore nel recupero dello stato del nodo: {response.status_code}")

    # Estrai lo stato dal corpo della risposta.
    node_status = response.json()["status"]

    return node_status


def md5sum_file(fname: str) -> str:
    """Get file MD5 checksum."""
    # TODO update in chunks.
    with open(fname, "rb") as f:
        data = f.read()
    return hashlib.md5(data).hexdigest()


def make_grid(num: int, cols: int):
    """Make grid."""
    xi, yi = 0, 0
    for i in range(1, num + 1):
        yield (xi, yi)
        xi += 1
        if i % cols == 0:
            yi += 1
            xi = 0


def check_resources() -> None:
    """Check some system resources."""
    nofile_soft, nofile_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if nofile_soft <= 1024:
        msg = (
            f"The maximum number of open file descriptors for the current process is set to {nofile_soft}.\n"
            "This limit might not be enough to run multiple devices in GNS3 (approx more than 150 docker devices, may vary).\n"
            "To increase the limit, edit '/etc/security/limits.conf' and append: \n"
            "*                hard    nofile          65536\n"
            "*                soft    nofile          65536\n"
        )
        warnings.warn(msg, RuntimeWarning)


def check_local_gns3_config() -> bool:
    """Checks for the GNS3 server config."""
    config = configparser.ConfigParser()
    with open(os.path.expanduser("~/.config/GNS3/2.2/gns3_server.conf")) as f:
        config.read_file(f)
    if not "Qemu" in config.keys():
        warnings.warn("Qemu settings are not configured. Enable KVM for better performance.", RuntimeWarning)
        return False
    kvm = config["Qemu"].get("enable_kvm")
    if kvm is None:
        warnings.warn("'enable_kvm' key not defined. Enable KVM for better performance.", RuntimeWarning)
        return False
    if kvm == "false":
        warnings.warn("'enable_kvm' set to false. Enable KVM for better performance.", RuntimeWarning)
        return False
    print(f"KVM is set to {kvm}")
    return True


def read_local_gns3_config():
    """Return some GNS3 configuration values."""
    config = configparser.ConfigParser()
    with open(os.path.expanduser("~/.config/GNS3/2.2/gns3_server.conf")) as f:
        config.read_file(f)
    return config["Server"].get("host"), config["Server"].getint("port"), config["Server"].getboolean("auth"), config["Server"].get("user"), config["Server"].get("password")


def check_server_version(server: Server) -> str:
    """Check GNS3 server version."""
    req = requests.get(f"http://{server.addr}:{server.port}/v2/version", auth=(server.user, server.password))
    req.raise_for_status()
    return req.json()["version"]



def get_all_projects(server: Server) -> List[Dict[str, Any]]:
    """Get all the projects in the GNS3 server."""
    req = requests.get(f"http://{server.addr}:{server.port}/v2/projects", auth=(server.user, server.password))
    req.raise_for_status()
    return req.json()


def get_project_by_name(server: Server, name: str) -> Optional[Project]:
    """Get GNS3 project by name."""
    projects = get_all_projects(server)
    filtered_project = list(filter(lambda x: x["name"] == name, projects))

    if not filtered_project:
        return None

    filtered_project = filtered_project[0]
    return Project(name=filtered_project["name"], id=filtered_project["project_id"],
                   grid_unit=int(filtered_project["grid_size"]))
def create_docker_template(server: Server, name: str, image: str, environment: str = '') -> Optional[Dict[str, Any]]:
    """Create a new GNS3 docker template.

    'environment' should be the empty string '' or a string with newline separated key=value pairs,
    e.g. environment = 'VAR_ONE=value1\nVAR2=2\nBLABLABLA=something'
    """
    defaults = {'adapters': 1,
                'builtin': False,
                'category': 'guest',
                'compute_id': 'local',
                'console_auto_start': False,
                'console_http_path': '/',
                'console_http_port': 80,
                'console_resolution': '1024x768',
                'console_type': 'telnet',
                'custom_adapters': [],
                'default_name_format': '{name}-{0}',
                'extra_hosts': '',
                'extra_volumes': [],
                'start_command': '',
                'symbol': ':/symbols/docker_guest.svg',
                'template_type': 'docker',
                'usage': ''}

    defaults["name"] = name
    defaults["image"] = image
    defaults["environment"] = environment






    req = requests.post(f"http://{server.addr}:{server.port}/v2/templates", data=json.dumps(defaults), auth=(server.user, server.password))
    req.raise_for_status()
    return req.json()

def create_docker_template_switch(server: Server, name: str, image: str, environment: str = '') -> Optional[Dict[str, Any]]:
    """Create a new GNS3 docker template.

    'environment' should be the empty string '' or a string with newline separated key=value pairs,
    e.g. environment = 'VAR_ONE=value1\nVAR2=2\nBLABLABLA=something'
    """
    defaults = {'adapters': 6,
                'builtin': False,
                'category': 'guest',
                'compute_id': 'local',
                'console_auto_start': False,
                'console_http_path': '/',
                'console_http_port': 80,
                'console_resolution': '1024x768',
                'console_type': 'telnet',
                'custom_adapters': [],
                'default_name_format': '{name}-{0}',
                'extra_hosts': '',
                'extra_volumes': [],
                'start_command': '',
                'symbol': ':/symbols/multilayer_switch.svg',
                'template_type': 'docker',
                'usage': ''
                }

    defaults["name"] = name
    defaults["image"] = image
    defaults["environment"] = environment


    req = requests.post(f"http://{server.addr}:{server.port}/v2/templates", data=json.dumps(defaults), auth=(server.user, server.password))
    req.raise_for_status()
    return req.json()

def create_docker_template_router(server: Server, name: str, image: str, environment: str = '') -> Optional[Dict[str, Any]]:
    """Create a new GNS3 docker template.

    'environment' should be the empty string '' or a string with newline separated key=value pairs,
    e.g. environment = 'VAR_ONE=value1\nVAR2=2\nBLABLABLA=something'
    """
    defaults = {'adapters': 3,
                'builtin': False,
                'category': 'guest',
                'compute_id': 'local',
                'console_auto_start': False,
                'console_http_path': '/',
                'console_http_port': 80,
                'console_resolution': '1024x768',
                'console_type': 'telnet',
                'custom_adapters': [],
                'default_name_format': '{name}-{0}',
                'extra_hosts': '',
                'extra_volumes': [],
                'start_command': '',
                'symbol': ':/symbols/router.svg',
                'template_type': 'docker',
                'usage': ''
                }

    defaults["name"] = name
    defaults["image"] = image
    defaults["environment"] = environment


    req = requests.post(f"http://{server.addr}:{server.port}/v2/templates", data=json.dumps(defaults), auth=(server.user, server.password))
    req.raise_for_status()
    return req.json()



def environment_dict_to_string(env: dict):
    """Environment variable dictionary to string."""
    res = []
    for k, v in env.items():
        res.append(f"{k}={v}")
    return "\n".join(res)


def extrahosts_dict_to_string(hosts: dict):
    """GNS3 extra_hosts."""
    res = []
    for k, v in hosts.items():
        res.append(f"{k}:{v}")
    return "\n".join(res)


def environment_string_to_dict(env: str):
    """Environment variable string to dictionary."""
    return { pair.split("=", 1)[0]: pair.split("=", 1)[1] for pair in env.split("\n") }


def get_docker_node_environment(server: Server, project: Project, node_id: str):
    """Get GNS3 docker node environment variables."""
    req = requests.get(f"http://{server.addr}:{server.port}/v2/projects/{project.id}/nodes/{node_id}", auth=(server.user, server.password))
    req.raise_for_status()
    return req.json()["properties"]["environment"]


def update_docker_node_environment(server: Server, project: Project, node_id: str, env: str):
    """Update GNS3 docker node environment variables."""
    payload = {"environment": env}
    req = requests.put(f"http://{server.addr}:{server.port}/v2/compute/projects/{project.id}/docker/nodes/{node_id}", data=json.dumps(payload), auth=(server.user, server.password))
    req.raise_for_status()
    return req.json()


def update_docker_node_extrahosts(server: Server, project: Project, node_id: str, hosts: str):
    """Update GNS3 docker node extra_hosts."""
    payload = {"extra_hosts": hosts}
    req = requests.put(f"http://{server.addr}:{server.port}/v2/compute/projects/{project.id}/docker/nodes/{node_id}", data=json.dumps(payload), auth=(server.user, server.password))
    req.raise_for_status()
    return req.json()


def create_project(server: Server, name: str, height: int, width: int, zoom: Optional[int] = 40):
    """Create GNS3 project."""
    # http://api.gns3.net/en/2.2/api/v2/controller/project/projects.html
    # Coordinate 0,0 is located in the center of the project
    payload_project = {"name": name, "show_grid": True, "scene_height": int(height), "scene_width": int(width), "zoom": int(zoom)}
    req = requests.post(f"http://{server.addr}:{server.port}/v2/projects", data=json.dumps(payload_project), auth=(server.user, server.password))
    try:
     req.raise_for_status()
     req = req.json()
     return Project(name=req["name"], id=req["project_id"], grid_unit=int(req["grid_size"]))
    except requests.exceptions.HTTPError as err:
     print(f"HTTP Error: {err}")
     print(f"Response: {req.text}")
    raise


def open_project_if_closed(server: Server, project: Project):
    """If the GNS3 project is closed, open it."""
    req = requests.get(f"http://{server.addr}:{server.port}/v2/projects/{project.id}", auth=(server.user, server.password))
    req.raise_for_status()
    if req.json()["status"] == "opened":
        print(f"Project {project.name} is already open.")
        return
    req = requests.post(f"http://{server.addr}:{server.port}/v2/projects/{project.id}/open", auth=(server.user, server.password))
    req.raise_for_status()
    print(f"Project {project.name} {req.json()['status']}.")
    assert req.json()["status"] == "opened"


def get_all_templates(server: Server) -> List[Dict[str, Any]]:
    """Get all the defined GNS3 templates."""
    req = requests.get(f"http://{server.addr}:{server.port}/v2/templates", auth=(server.user, server.password))
    req.raise_for_status()
    return req.json()


def get_static_interface_config_file(iface: str, address: str, netmask: str, gateway: str, nameserver: Optional[str] = None) -> str:
    """Configuration file for a static network interface."""
    if nameserver is None:
        nameserver = gateway
    return (
        "# autogenerated\n"
        f"# Static config for {iface}\n"
        f"auto {iface}\n"
        f"iface {iface} inet static\n"
        f"\taddress {address}\n"
        f"\tnetmask {netmask}\n"
        f"\tgateway {gateway}\n"
        f"\tup echo nameserver {nameserver} > /etc/resolv.conf\n"
    )


def get_template_from_id(server: Server, template_id: str) -> Dict[str, Any]:
    """Get templete description from template ID."""
    req = requests.get(f"http://{server.addr}:{server.port}/v2/templates/{template_id}", auth=(server.user, server.password))
    req.raise_for_status()
    return req.json()


def get_template_id_from_name(templates: List[Dict[str, Any]], name: str) -> Optional[str]:
    """Get GNS3 template ID from the template name."""
    for template in templates:
        if template["name"] == name:
            return template["template_id"]
    return None
def get_template_id_from_name2(template_name: str, templates: List[Dict[str, Any]]) -> Optional[str]:
    """Get GNS3 template ID from the template name."""
    for template in templates:
        if template["name"] == template_name:
            return template["template_id"]
    return None

def get_all_nodes(server: Server, project: Project) -> List[Dict[str, Any]]:
    """Get all nodes in a GNS3 project."""
    req = requests.get(f"http://{server.addr}:{server.port}/v2/projects/{project.id}/nodes", auth=(server.user, server.password))
    req.raise_for_status()
    return req.json()


def get_nodes_id_by_name_regexp(server: Server, project: Project, name_regexp: Pattern) -> Optional[List[Item]]:
    """Get the list of all node IDs that match a node name regular expression."""
    nodes = get_all_nodes(server, project)
    nodes_filtered = list(filter(lambda n: name_regexp.match(n["name"]), nodes))
    return [Item(n["name"], n["node_id"]) for n in nodes_filtered]


def get_node_telnet_host_port(server: Server, project: Project, node_id: str) -> tuple:
    """Get the telnet hostname and port of a node."""
    req = requests.get(f"http://{server.addr}:{server.port}/v2/projects/{project.id}/nodes/{node_id}", auth=(server.user, server.password))
    req.raise_for_status()
    # TODO include checks for console type
    assert req.json()["console_type"] == "telnet"
    if req.json()["console_host"] in ("0.0.0.0", "::"):
        host = server.addr
    else:
        host = req.json()["console_host"]
    return (host, req.json()["console"])


def get_node_docker_container_id(server: Server, project: Project, node_id: str) -> str:
    """Get the Docker container id."""
    req = requests.get(f"http://{server.addr}:{server.port}/v2/projects/{project.id}/nodes/{node_id}", auth=(server.user, server.password))
    req.raise_for_status()
    assert req.json()["node_type"] == "docker"
    return req.json()["properties"]["container_id"]


def get_links_id_from_node_connected_to_name_regexp(server: Server, project: Project, node_id: str, name_regexp: Pattern) -> Optional[List[Item]]:
    """Get all the link IDs from node node_id connected to other nodes with names that match name_regexp regular expression."""
    req = requests.get(f"http://{server.addr}:{server.port}/v2/projects/{project.id}/nodes/{node_id}", auth=(server.user, server.password))
    req.raise_for_status()
    node_name = req.json()["name"]

    req = requests.get(f"http://{server.addr}:{server.port}/v2/projects/{project.id}/nodes/{node_id}/links", auth=(server.user, server.password))
    req.raise_for_status()
    links = req.json()
    relevant_nodes = get_nodes_id_by_name_regexp(server, project, name_regexp)

    def is_link_relevant(link: Dict) -> Optional[Item]:
        for c in link["nodes"]: # two ends of the link
            for rn in relevant_nodes:
                if c["node_id"] == rn.id:
                    return rn
        return None

    links_filtered: List[Item]= []
    for link in links:
        linked_node = is_link_relevant(link)
        if linked_node:
            links_filtered.append(Item(f"{linked_node.name} <--> {node_name}", link["link_id"]))

    return links_filtered

def create_node(server: Server, project: Project, start_x: int, start_y: int, node_template_id: str, node_name: Optional[str] = None):
    """Create selected node at coordinates start_x, start_y."""
    payload = {"x": int(start_x), "y": int(start_y)}
    if node_name:
        # GNS3 is not updating the name...
        payload["name"] = node_name
    req = requests.post(f"http://{server.addr}:{server.port}/v2/projects/{project.id}/templates/{node_template_id}", data=json.dumps(payload), auth=(server.user, server.password))
    req.raise_for_status()
    return req.json()






def start_node(server: Server, project: Project, node_id: str) -> None:
    """Start selected node."""
    req = requests.post(f"http://{server.addr}:{server.port}/v2/projects/{project.id}/nodes/{node_id}/start", data={}, auth=(server.user, server.password))
    req.raise_for_status()


def get_node_id_by_name(server: Server, project: Project, node_name: str) -> Optional[str]:
    """Get the ID of the node with the specified name."""
    nodes = get_all_nodes(server, project)

    for node in nodes:
        if node["name"] == node_name:
            return node["node_id"]

    return None




def stop_node(server: Server, project: Project, node_id: str) -> None:
    """Stop selected node."""
    req = requests.post(f"http://{server.addr}:{server.port}/v2/projects/{project.id}/nodes/{node_id}/stop", data={}, auth=(server.user, server.password))
    req.raise_for_status()


def delete_node(server: Server, project: Project, node_id: str) -> None:
    """Delete selected node."""
    # check if node is running?
    req = requests.delete(f"http://{server.addr}:{server.port}/v2/projects/{project.id}/nodes/{node_id}", auth=(server.user, server.password))
    req.raise_for_status()


def start_node_by_name(server: Server, project: Project, node_name: str) -> None:
    """Start the selected node by name."""
    node_id = get_node_id_by_name(server, project, node_name)

    if node_id:
        req = requests.post(f"http://{server.addr}:{server.port}/v2/projects/{project.id}/nodes/{node_id}/start",
                            data={}, auth=(server.user, server.password))
        req.raise_for_status()
        print(f"Node '{node_name}' started successfully.")
    else:
        print(f"Node '{node_name}' not found.")

def create_link(server: Server, project: Project, node1_id: str, node1_port: int, node2_id: str, node2_port: int):
    """Create link between two nodes."""
    try:
        payload = {"nodes":[{"node_id": node1_id, "adapter_number": node1_port, "port_number": 0},
                            {"node_id": node2_id, "adapter_number": node2_port, "port_number": 0}]}
        req = requests.post(f"http://{server.addr}:{server.port}/v2/projects/{project.id}/links", data=json.dumps(payload), auth=(server.user, server.password))
        req.raise_for_status()
        # TODO rename link node labels
        return req.json()

    except requests.exceptions.HTTPError as err:
        print(f"HTTP Error: {err}")
        print(f"Response: {req.text}")
        # Puoi rilanciare l'eccezione per propagarla al chiamante se lo desideri
        raise

    except Exception as e:
        # Gestisci altre eccezioni se necessario
        print(f"Error: {e}")
        raise

import re

def create_link_by_name(server: Server, project: Project, node1_name: str, node1_port: int, node2_name: str, node2_port: int):
    """Create link between two nodes using node names."""
    try:
        # Ottenere gli ID dei nodi corrispondenti ai nomi forniti
        node1_id = get_nodes_id_by_name_regexp(server, project, re.compile(f"^{re.escape(node1_name)}$"))
        node2_id = get_nodes_id_by_name_regexp(server, project, re.compile(f"^{re.escape(node2_name)}$"))

        if not node1_id or not node2_id:
            raise ValueError("Node names not found.")

        payload = {"nodes":[{"node_id": node1_id[0].id, "adapter_number": node1_port, "port_number": 0},
                            {"node_id": node2_id[0].id, "adapter_number": node2_port, "port_number": 0}]}
        req = requests.post(f"http://{server.addr}:{server.port}/v2/projects/{project.id}/links", data=json.dumps(payload), auth=(server.user, server.password))
        req.raise_for_status()
        # TODO rename link node labels
        return req.json()
    except Exception as e:
        print(f"Error creating link: {e}")
        return None




def set_node_network_interfaces(server: Server, project: Project, node_id: str, iface_name: str, ip_iface: ipaddress.IPv4Interface, gateway: str, nameserver: Optional[str] = None) -> None:
    """Configure the /etc/network/interfaces file for the node."""
    if ip_iface.netmask == ipaddress.IPv4Address("255.255.255.255"):
        warnings.warn(f"Interface netmask is set to {ip_iface.netmask}", RuntimeWarning)
    payload = get_static_interface_config_file(iface_name, str(ip_iface.ip), str(ip_iface.netmask), gateway, nameserver)
    req = requests.post(f"http://{server.addr}:{server.port}/v2/projects/{project.id}/nodes/{node_id}/files/etc/network/interfaces", data=payload, auth=(server.user, server.password))
    req.raise_for_status()


def create_cluster_of_nodes(server: Server, project: Project, num_devices: int, start_x: int, start_y: int, nodes_per_row: int,
                            switch_template_id: str, node_template_id: str, upstream_switch_id: Optional[str], upstream_switch_port: Optional[int],
                            node_start_ip_iface: ipaddress.IPv4Interface, gateway: str, nameserver: str, spacing: Optional[float] = 2):
    """Create cluster of nodes.

          R  <--- gateway (must exist in the topology).
          |
          S  <--- upstream switch (must exist in the topology).
         /
        S  <----- cluster switch, based on switch_template_id. At coordinates (start_x, start_y).
        |
    n n n n n    |  num_devices number of            first ip address = node_start_ip_iface.ip
    n n n n n  <-|  nodes, based on                  last ip address = node_start_ip_iface.ip + num_devices - 1
    n n n n n    |  node_template_id.
    """
    assert num_devices > 0
    assert nodes_per_row > 0
    assert get_template_from_id(server, switch_template_id)["adapters"] >= (num_devices - (1 if upstream_switch_id else 0))
    if not spacing:
        spacing = 2

    # create cluster switch
    cluster_switch = create_node(server, project, start_x, start_y, switch_template_id)
    print(f"Creating node {cluster_switch['name']}")
    # create device grid
    coord_first = Position(start_x - project.grid_unit * spacing * (nodes_per_row - 1) // 2, start_y + project.grid_unit * spacing)
    devices = []

    for dx, dy in make_grid(num_devices, nodes_per_row):
        device = create_node(server, project, coord_first.x + project.grid_unit * spacing * dx, coord_first.y + project.grid_unit * spacing * dy, node_template_id)
        devices.append(device)
        print(f"Creating node {device['name']}")
        time.sleep(0.1)

    coord_last = Position(devices[-1]["x"], devices[-1]["y"])

    # links
    if upstream_switch_id:
        create_link(server, project, cluster_switch["node_id"], 0, upstream_switch_id, upstream_switch_port)
        print(f"Creating link {cluster_switch['name']} <--> {upstream_switch_id}")
    for i, device in enumerate(devices, start=1):
        create_link(server, project, device["node_id"], 0, cluster_switch["node_id"], i)
        print(f"Creating link {device['name']} <--> {cluster_switch['name']}")
        time.sleep(0.1)

    # configure devices
    for i, device in enumerate(devices, start=0):
        device_ip_iface = ipaddress.IPv4Interface(f"{node_start_ip_iface.ip + i}/{node_start_ip_iface.netmask}")
        set_node_network_interfaces(server, project, device["node_id"], "eth0", device_ip_iface, gateway, nameserver)
        print(f"Configuring {device['name']} addr: {device_ip_iface.ip}/{device_ip_iface.netmask} gw: {gateway} ns: {nameserver}")

    # decoration
    payload = {"x": int(start_x + project.grid_unit * spacing), "y": int(start_y - 15),
               "svg": f"<svg><text font-family=\"monospace\" font-size=\"12\">Start addr: {node_start_ip_iface.ip}/{node_start_ip_iface.netmask}</text></svg>"}
    req = requests.post(f"http://{server.addr}:{server.port}/v2/projects/{project.id}/drawings", data=json.dumps(payload), auth=(server.user, server.password))
    req.raise_for_status()

    payload = {"x": int(start_x + project.grid_unit * spacing), "y": int(start_y),
               "svg": f"<svg><text font-family=\"monospace\" font-size=\"12\">End addr  : {device_ip_iface.ip}/{device_ip_iface.netmask}</text></svg>"}
    req = requests.post(f"http://{server.addr}:{server.port}/v2/projects/{project.id}/drawings", data=json.dumps(payload), auth=(server.user, server.password))
    req.raise_for_status()

    payload = {"x": int(start_x + project.grid_unit * spacing), "y": int(start_y + 15),
               "svg": f"<svg><text font-family=\"monospace\" font-size=\"12\">Gateway   : {gateway}</text></svg>"}
    req = requests.post(f"http://{server.addr}:{server.port}/v2/projects/{project.id}/drawings", data=json.dumps(payload), auth=(server.user, server.password))
    req.raise_for_status()

    payload = {"x": int(start_x + project.grid_unit * spacing), "y": int(start_y + 30),
               "svg": f"<svg><text font-family=\"monospace\" font-size=\"12\">Nameserver: {nameserver}</text></svg>"}
    req = requests.post(f"http://{server.addr}:{server.port}/v2/projects/{project.id}/drawings", data=json.dumps(payload), auth=(server.user, server.password))
    req.raise_for_status()

    return (cluster_switch, devices, coord_first, coord_last)


def start_capture(server, project, link_ids):
    """Start packet capture (wireshark) in the selected link_ids."""
    for link in link_ids:
        req = requests.post(f"http://{server.addr}:{server.port}/v2/projects/{project.id}/links/{link}/start_capture", data={}, auth=(server.user, server.password))
        req.raise_for_status()
        result = req.json()
        print(f"Capturing {result['capturing']}, {result['capture_file_name']}")
        time.sleep(0.3)


def stop_capture(server, project, link_ids):
    """Stop packet capture in the selected link_ids."""
    for link in link_ids:
        req = requests.post(f"http://{server.addr}:{server.port}/v2/projects/{project.id}/links/{link}/stop_capture", data={}, auth=(server.user, server.password))
        req.raise_for_status()
        result = req.json()
        print(f"Capturing {result['capturing']}, {result['capture_file_name']}")
        time.sleep(0.3)


def start_all_nodes_by_name_regexp(server: Server, project: Project, node_pattern: Pattern, sleeptime: float = 0.1) -> None:
    """Start all nodes that match a name regexp."""
    nodes = get_nodes_id_by_name_regexp(server, project, node_pattern)
    if nodes:
        print(f"found {len(nodes)} nodes")
        for node in nodes:
            print(f"Starting {node.name}... ", end="", flush=True)
            start_node(server, project, node.id)
            print("OK")
            time.sleep(sleeptime)


def stop_all_nodes_by_name_regexp(server: Server, project: Project, node_pattern: Pattern, sleeptime: float = 0.1) -> None:
    """Stop all nodes that match a name regexp."""
    nodes = get_nodes_id_by_name_regexp(server, project, node_pattern)
    if nodes:
        print(f"found {len(nodes)} nodes")
        for node in nodes:
            print(f"Stopping {node.name}... ", end="", flush=True)
            stop_node(server, project, node.id)
            print("OK")
            time.sleep(sleeptime)


def start_all_switches(server: Server, project: Project, switches_pattern : Pattern=re.compile("openvswitch.*", re.IGNORECASE), sleeptime: float = 1.0) -> None:
    """Start all network switch nodes (OpenvSwitch switches)."""
    start_all_nodes_by_name_regexp(server, project, switches_pattern, sleeptime)


def start_all_routers(server: Server, project: Project, routers_pattern : Pattern=re.compile("vyos.*", re.IGNORECASE), sleeptime: float = 60.0) -> None:
    """Start all router nodes (VyOS routers)."""
    start_all_nodes_by_name_regexp(server, project, routers_pattern, sleeptime)


def start_all_iot(server: Server, project: Project, iot_pattern : Pattern=re.compile("iotsim-.*", re.IGNORECASE)) -> None:
    """Start all iotsim-* docker nodes."""
    start_all_nodes_by_name_regexp(server, project, iot_pattern)


def stop_all_switches(server: Server, project: Project, switches_pattern : Pattern=re.compile("openvswitch.*", re.IGNORECASE)) -> None:
    """Stop all network switch nodes (OpenvSwitch switches)."""
    stop_all_nodes_by_name_regexp(server, project, switches_pattern)


def stop_all_routers(server: Server, project: Project, routers_pattern : Pattern=re.compile("vyos.*", re.IGNORECASE)) -> None:
    """Stop all router nodes (VyOS routers)."""
    stop_all_nodes_by_name_regexp(server, project, routers_pattern)


def start_capture_all_iot_links(server, project, switches_pattern: Pattern=re.compile("openvswitch.*", re.IGNORECASE), iot_pattern: Pattern=re.compile("mqtt-device.*|coap-device.*", re.IGNORECASE)) -> None:
    """Start packet capture on each IoT device."""
    switches = get_nodes_id_by_name_regexp(server, project, switches_pattern)
    if switches:
        print(f"found {len(switches)} switches")
        for sw in switches:
            print(f"Finding links in switch {sw.name}... ", end="", flush=True)
            links = get_links_id_from_node_connected_to_name_regexp(server, project, sw.id, iot_pattern)
            if links:
                print(f"{len(links)} found")
                for lk in links:
                    print(f"\t Starting capture in link {lk.name}... ", end="", flush=True)
                    start_capture(server, project, [lk.id])
                    print("OK")
            else:
                print("0 links, skipping.")
        time.sleep(0.3)


def stop_capture_all_iot_links(server, project, switches_pattern: Pattern=re.compile("openvswitch.*", re.IGNORECASE), iot_pattern: Pattern=re.compile("mqtt-device.*|coap-device.*", re.IGNORECASE)) -> None:
    """Stop packet capture on each IoT device."""
    switches = get_nodes_id_by_name_regexp(server, project, switches_pattern)
    if switches:
        print(f"found {len(switches)} switches")
        for sw in switches:
            print(f"Finding links in switch {sw.name}... ", end="", flush=True)
            links = get_links_id_from_node_connected_to_name_regexp(server, project, sw.id, iot_pattern)
            if links:
                print(f"{len(links)} found")
                for lk in links:
                    print(f"\t Stopping capture in link {lk.name}... ", end="", flush=True)
                    stop_capture(server, project, [lk.id])
                    print("OK")
            else:
                print("0 links, skipping.")
        time.sleep(0.3)


def check_ipaddrs(server: Server, project: Project):
    """Check for duplicated addresses in the project."""
    nodes = get_all_nodes(server, project)
    found_addrs = {}
    for node in nodes:
        req = requests.get(f"http://{server.addr}:{server.port}/v2/projects/{project.id}/nodes/{node['node_id']}/files/etc/network/interfaces", auth=(server.user, server.password))
        if not req.ok:
            print(f"Ignoring  {node['name']}:\t{req.status_code} {req.reason} /etc/network/interfaces")
            continue
        # ignore comments
        ifaces = "\n".join(filter(lambda l: not l.strip().startswith("#"), req.text.split("\n")))
        match = re.search(r"address\s+(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})", ifaces)
        if match:
            addr = match.group(1)
            found_addrs[addr] = found_addrs.get(addr, 0) + 1
            print(f"Searching {node['name']}:\t{addr}")
        else:
            print(f"Searching {node['name']}:\tNo matches")

    duplicates = {k: v for k, v in found_addrs.items() if v > 1}
    if duplicates:
        raise ValueError(f"Duplicated ip addresses found: {duplicates}")


def install_vyos_image_on_node(node_id: str, hostname: str, telnet_port: int, pre_exec : Optional[str] = None) -> None:
    """Perform VyOS installation steps.

    pre_exec example:
    pre_exec = "konsole -e telnet localhost 5000"
    """
    if pre_exec:
        import subprocess
        import shlex
        pre_proc = subprocess.Popen(shlex.split(pre_exec))
    with Telnet(hostname, telnet_port) as tn:
        out = tn.read_until(b"vyos login:")
        print(out.decode("utf-8").split("\n")[-1])

        tn.write(b"vyos\n")
        out = tn.expect([b"Password:"], timeout=10)
        print(out[2].decode("utf-8"))

        tn.write(b"vyos\n")
        out = tn.expect([b"vyos@vyos:~\$"], timeout=10)
        print(out[0])
        print(out[2].decode("utf-8"))

        tn.write(b"install image\n")
        out = tn.expect([b"Would you like to continue\? \(Yes/No\)"], timeout=10)
        print(out[0])
        print(out[2].decode("utf-8"))

        tn.write(b"Yes\n")
        out = tn.expect([b"Partition \(Auto/Parted/Skip\)"], timeout=10)
        print(out[0])
        print(out[2].decode("utf-8"))

        tn.write(b"Auto\n")
        out = tn.expect([b"Install the image on"], timeout=10)
        print(out[0])
        print(out[2].decode("utf-8"))

        tn.write(b"\n")
        out = tn.expect([b"Continue\? \(Yes/No\)"], timeout=10)
        print(out[0])
        print(out[2].decode("utf-8"))

        tn.write(b"Yes\n")
        out = tn.expect([b"How big of a root partition should I create"], timeout=30)
        print(out[0])
        print(out[2].decode("utf-8"))

        tn.write(b"\n")
        out = tn.expect([b"What would you like to name this image"], timeout=30)
        print(out[0])
        print(out[2].decode("utf-8"))

        tn.write(b"\n")
        out = tn.expect([b"Which one should I copy to"], timeout=30)
        print(out[0])
        print(out[2].decode("utf-8"))

        tn.write(b"\n")
        out = tn.expect([b"Enter password for user 'vyos':"], timeout=10)
        print(out[0])
        print(out[2].decode("utf-8"))

        tn.write(b"vyos\n")
        out = tn.expect([b"Retype password for user 'vyos':"], timeout=10)
        print(out[0])
        print(out[2].decode("utf-8"))

        tn.write(b"vyos\n")
        out = tn.expect([b"Which drive should GRUB modify the boot partition on"], timeout=10)
        print(out[0])
        print(out[2].decode("utf-8"))

        tn.write(b"\n")
        out = tn.expect([b"vyos@vyos:~\$"], timeout=30)
        print(out[0])
        print(out[2].decode("utf-8"))

        time.sleep(2)
        tn.write(b"poweroff\n")
        out = tn.expect([b"Are you sure you want to poweroff this system"], timeout=10)
        print(out[0])
        print(out[2].decode("utf-8"))

        tn.write(b"y\n")
        time.sleep(2)

    if pre_exec:
        pre_proc.kill()


def configure_vyos_image_on_node(node_id: str, hostname: str, telnet_port: int, path_script: str, pre_exec: Optional[str] = None) -> None:
    """Configure VyOS router.

    pre_exec example:
    pre_exec = "konsole -e telnet localhost 5000"
    """
    if pre_exec:
        import subprocess
        import shlex
        pre_proc = subprocess.Popen(shlex.split(pre_exec))

    local_checksum = md5sum_file(path_script)

    with open(path_script, "rb") as f:
        config = base64.b64encode(f.read())

    with Telnet(hostname, telnet_port) as tn:
        out = tn.read_until(b"vyos login:")
        print(out.decode("utf-8").split("\n")[-1])

        tn.write(b"vyos\n")
        out = tn.expect([b"Password:"], timeout=10)
        print(out[2].decode("utf-8"))

        tn.write(b"vyos\n")
        out = tn.expect([b"vyos@vyos:~\$"], timeout=10)
        print(out[0])
        print(out[2].decode("utf-8"))

        payload = b"echo '" + config + b"' >> config.b64\n"
        tn.write(payload)
        out = tn.expect([b"vyos@vyos:~\$"], timeout=10)
        print(out[0])
        print(out[2].decode("utf-8"))

        tn.write(b"base64 --decode config.b64 > config.sh\n")
        out = tn.expect([b"vyos@vyos:~\$"], timeout=10)
        print(out[0])
        print(out[2].decode("utf-8"))

        tn.write(b"md5sum config.sh\n")
        out = tn.expect([re.compile(r"[0-9a-f]{32}  config.sh".encode("utf-8"))], 5)
        if out[0] == -1:
            warnings.warn("Error generating file MD5 checksum.", RuntimeWarning)
            return
        uploaded_checksum = out[1].group().decode("utf-8").split()[0]

        if uploaded_checksum != local_checksum:
            warnings.warn("Checksums do not match.", RuntimeWarning)
        else:
            print("Checksums match.")

        tn.write(b"chmod +x config.sh\n")
        out = tn.expect([b"vyos@vyos:~\$"], timeout=10)
        print(out[0])
        print(out[2].decode("utf-8"))

        tn.write(b"./config.sh\n")
        out = tn.expect([b"Done"], timeout=60)
        print(out[0])
        print(out[2].decode("utf-8"))
        out = tn.expect([b"vyos@vyos:~\$"], timeout=10)
        print(out[0])
        print(out[2].decode("utf-8"))

        tn.write(b"poweroff\n")
        out = tn.expect([b"Are you sure you want to poweroff this system"], timeout=10)
        print(out[0])
        print(out[2].decode("utf-8"))

        tn.write(b"y\n")
        time.sleep(2)

    if pre_exec:
        pre_proc.kill()
