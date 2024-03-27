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

"""
THIS SCRIPT OVERWRITES THE PROMETHEUS SERVER ADDRESS 
SPECIFICALLY, THIS CODE USES THE DYNAMIC ADDRESS GIVEN TO THE CONTAINER
YOU SHOULD NOT NEED TO RUN THIS SCRIPT, BECAUSE THE PROMEHTEUS SERVER RUNS ON "LOCALHOST" IN THE OUT-OF-THE-BOX CONFIGURATION

HOWEVER, IF YOU NEED TO RUN PROMETHEUS ON ANOTHER HOST, OR IF YOU USE SOME OTHER CONFIGURATION SETUP, THIS SCRIPT COULD BE USEFUL
THIS SCRIPT SCHOULD BE EXECUTED BEFORE RUNNING PROMETHEUS AND GRAFANA. ADJUST YOUR IP ADRRESSES, AND TRANSPORT PORTS, ACCORDING TO YOUR NEEDS. 
"""

import netifaces as ni 
import os

def get_source_ip_address():
    try:
        ip = ni.ifaddresses('eth1')[ni.AF_INET][0]['addr']
        return ip
    except ValueError:
        return "Interface not found"


def generate_prometheus_config(ip):
    config = f"""\
global:
  scrape_interval:     5s
  evaluation_interval: 5s

  external_labels:
    monitor: 'example'

alerting:
  alertmanagers:
  - static_configs:
    - targets: ['{ip}:9093']

rule_files:

scrape_configs:
  - job_name: 'prometheus'

    scrape_interval: 5s
    scrape_timeout: 5s

    static_configs:
      - targets: ['{ip}:9090']

  - job_name: 'system_metrics'

    static_configs:
      - targets: ['{ip}:8000']"""

    # Get the directory path of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the file path relative to the script directory
    file_path = os.path.join(script_dir, 'prometheus.yml')

    # Write to the file
    with open(file_path, 'w') as f:
        f.write(config)

if __name__ == "__main__":
    dynamic_IP = get_source_ip_address()
    print(f'Generating Prometheus config file with ip {dynamic_IP}')
    generate_prometheus_config(dynamic_IP)
    print('Done!')
