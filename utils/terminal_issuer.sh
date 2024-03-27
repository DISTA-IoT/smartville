#!/bin/bash
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
# THIS SCRIPT WONT WORK UNLESS YOU MAKE IT EXECUTABLE WITH chmod +x terminal_issuer.sh
 
# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker to use this script."
    exit 1
fi

# Check if at least one argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <container1:title1:command1> [<container2:title2:command2> ...]"
    exit 1
fi


# Parse the arguments into an array
containers_and_commands=("$@")

# Iterate through the list of containers and commands
for arg in "${containers_and_commands[@]}"; do



    # Extract container, title, and command from the argument
    container=$(echo "$arg" | cut -d ':' -f 1)
    title=$(echo "$arg" | cut -d ':' -f 2)
    command=$(echo "$arg" | cut -d ':' -f 3-)

    # Launch a new terminal window and execute the command in the container
    gnome-terminal --title "$title" -- docker exec -it "$container" bash -c "$command; bash"


done