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
THIS SCRIPT OVERWRITES THE PROMETHEUS SERVER ADDRESS TO THE DYNAMIC ADDRESS GIVEN TO THE CONTAINER
IT MUST BE RUN EACH TIME THE CONTROLLER CONTAINER IS REBOOT.
THE OVERWRITTING TAKES PLACE IN THE PROMETHEUS CONFIG FILE, FOR THIS REASON, THIS SCRIPT MUST BE RUN BEFORE LAUNCHING PROMETHEUS.
"""
import netifaces as ni 


def get_source_ip_address():
    try:
        ip = ni.ifaddresses('eth1')[ni.AF_INET][0]['addr']
        return ip
    except ValueError:
        return "Interface not found"


def replace_string_in_file(file_name, old_string, new_string):
    try:
        # Read the content of the file
        with open(file_name, 'r') as file:
            file_content = file.read()

        # Replace the old string with the new string
        modified_content = file_content.replace(old_string, new_string)

        # Write the modified content back to the file
        with open(file_name, 'w') as file:
            file.write(modified_content)

        print(f"String '{old_string}' replaced with '{new_string}' in {file_name}")
    except FileNotFoundError:
        print(f"Error: File '{file_name}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")



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
  - job_name: 'system_metrics'

    static_configs:
      - targets: ['{ip}:8000']"""

    with open('prometheus.yml', 'w') as f:
        f.write(config)



if __name__ == "__main__":
    # file_name = "pox/smartController/prometheus.yml"
    # old_string = "localhost"
    dynamic_IP = get_source_ip_address()
    # replace_string_in_file(file_name, old_string, dynamic_IP)
    generate_prometheus_config(dynamic_IP)