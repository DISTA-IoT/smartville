#This script has to be run in case the GNS3 installation does not generate automatically the "gns3_server.conf" file.
#It creates the default configration file in the default directory for linux "$HOME/.config/GNS3/2.2/". 

#!/bin/bash

# Create directories
mkdir -p "$HOME/.config/GNS3/2.2/"

# Copy configuration file
cp "gns3_server.conf" "$HOME/.config/GNS3/2.2/"

echo "Default server config file generated"