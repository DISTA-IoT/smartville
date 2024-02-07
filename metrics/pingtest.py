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
