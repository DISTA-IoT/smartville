#!/bin/bash

# Create directories
mkdir -p "$HOME/.config/GNS3/2.2/"

# Copy configuration file
cp "gns3_server.conf" "$HOME/.config/GNS3/2.2/"

echo "Default server config file generated"