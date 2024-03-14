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
import subprocess
import time

host = "www.google.com"  # Rimpiazza col nome di un sito a piacere

while True:
    start_time = time.time()
    # Faccio correre il comando del ping
    result = subprocess.run(["ping", "-c", "1", host], capture_output=True, text=True)

    # Estraggo l'output
    output = result.stdout

    print(output)

    lines = output.split('\n')

    round_trip_time = None

    for line in lines:
        if "time=" in line:
            time_index = line.find("time=")
            time_str = line[time_index + 5:].split()[0]
            round_trip_time = float(time_str)
            break

    if round_trip_time is not None:
        print(f"Round-trip time: {round_trip_time} ms")
    else:
        print("Unable to retrieve round-trip time.")

    # Attendo 1 centesimo di secondo per poi fare correre di nuovo il programma
    time.sleep(0.01)
