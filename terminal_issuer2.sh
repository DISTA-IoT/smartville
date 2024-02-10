#!/bin/bash

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

# Construct the command for gnome-terminal
command="gnome-terminal"

# Iterate through the list of containers, titles, and commands
for arg in "$@"; do
    # Extract container, title, and command from the argument
    container=$(echo "$arg" | cut -d ':' -f 1)
    title=$(echo "$arg" | cut -d ':' -f 2)
    command_arg=$(echo "$arg" | cut -d ':' -f 3-)

    # Append the tab specification to the gnome-terminal command
    command+=" --tab --title=\"$title\" -- bash -c 'docker exec -it \"$container\" bash -c \"$command_arg; bash\"' "
done

# Launch gnome-terminal with the constructed command
eval "$command"
