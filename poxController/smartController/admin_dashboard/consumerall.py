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
from confluent_kafka import KafkaException
from confluent_kafka.admin import AdminClient
# from prometheus_client import start_http_server, Gauge
# import dashgenerator
# import graphgenerator
# import graphsort
import consumerthread
import time
import socket
import argparse
import sys

def server_exist(bootstrap_servers):

    # Controlla se la stinga è del formato host:porta

    if ':' not in bootstrap_servers:
        print("Errore: la stringa deve contenere il formato host:porta")
        return False

    split_values = bootstrap_servers.split(':')

    if len(split_values) != 2 :
        print("Errore: la stringa deve contenere solo due valori (host:porta)")
        return False
        
    host, port = bootstrap_servers.split(':')

    if not port.isdigit():
        print(f"Errore: il valore della porta '{port}' non è un numero valido.")
        return False

    try:
        # Attempt to create a socket connection to the Kafka broker
        with socket.create_connection((host, port), timeout=2):
            print(f"Server {host}:{port} raggiungibile.")
            return True
    except (socket.error, socket.timeout) as e:
        print(f"Server {host}:{port} non raggiungibile")
        return False


if __name__ == "__main__":

    if len(sys.argv) > 1:
        print("Usage: python3 producer.py")
        sys.exit(1)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Metrics producer script")
    # parser.add_argument("some_arg", help="some_argument for the script")

    args = parser.parse_args()
    # some_arg = args.some_arg

    kafka_check = False

    while not kafka_check:

        bootstrap_servers = "192.168.1.1:9092"

        if server_exist(bootstrap_servers):
            try:
                #consumer_conf = {'bootstrap.servers': bootstrap_servers, 'group.id': 'my-group'}
                conf = {'bootstrap.servers': bootstrap_servers}
                #consumer = Consumer(consumer_conf)
                admin = AdminClient(conf)
                topics = admin.list_topics(timeout=5)
                kafka_check = True

            except KafkaException as e:
                print(f"Errore nella connessione")
                admin = None
                continue
        else:
            print(f"Kafka server {bootstrap_servers} non trovato.")


    topicslist = []

    """
    # Avvio server Prometheus
    start_http_server(8000)
    # Definizione metriche inserite su Prometheus
    cpu_metric = Gauge('CPU_percentage', 'Metrica CPU percentuale', ['label_name'])
    ram_metric = Gauge('RAM_GB', 'Metrica RAM', ['label_name'])
    ping_metric = Gauge('Latenza_ms', 'Metrica latenza del segnale', ['label_name'])
    incoming_traffic_metric = Gauge('Incoming_network_KB', 'Metrica traffico in entrata', ['label_name'])
    outcoming_traffic_metric = Gauge('Outcoming_network_KB', 'Metrica traffico in uscita', ['label_name'])
    """

    threads = []
    working_threads_count = 0
    sortcount = 0
        
    try:

        """
        # Chiamata al metodo di inserimento dashboards su Grafana
        # questo compito viene svolto nella classe dashgenerator
        dash = dashgenerator.dash_generator()
        grafana_cred = dash.dash_gen()
        """

        while True:

            updated_topicslist = []
            updated_topics = admin.list_topics()

            # Inserimento topics in una lista di topics aggiornata
            for topic, details in updated_topics.topics.items():
                if topic != '__consumer_offsets':
                    updated_topicslist.append(topic)

            topicslist_set = set(topicslist)
            updated_topicslist_set = set(updated_topicslist)

            # Creazione di una lista contenente i nuovi topics inseriti
            new_topicslist_set = updated_topicslist_set - topicslist_set
            new_topicslist = list(new_topicslist_set)

            # Tramite questo passaggio, la lista di topics vecchia prende il posto della lista di topics aggiornata
            # Ciò permetterà poi di essere confrontata in un prossimo ciclo con la futura lista dei topics
            # aggiornata, in modo tale da verificare ogni volta se sono stati inseriti nuovi topics 
            topicslist = updated_topicslist

            time.sleep(5)

            # Per ciascun topic nella nuova lista di topics, viene avviato un thread dedicato alla lettura
            # delle metriche in esso contenuto
            for topic in new_topicslist:

                print(f"Thread per il topic {topic} in avviamento")

                """
                # Chiamata al metodo di inserimento grafici su Grafana per ciascun topic/utente
                # questo compito viene svolto nella classe graphgenerator
                graph = graphgenerator.graph_generator()
                # usr_exist = graph.graph_check()
                # if (usr_exist):
                graph.graph_gen(topic, grafana_cred)
                
                # Avvio thread dedicato alla lettura metriche contenute nel topic passato in ingresso 
                # I compiti del thread vengono definiti nella classe consumerthread in cui vengono passate
                # in ingresso tutte quante le metriche da aggiornare su prometheus, in modo tale da evitare
                # di dichiarare le metriche nel sistema di archiviazione ogni volta che viene avviato un
                # nuovo thread, e quindi, rischiare di causare dei conflitti dovuti all'accesso concorrente
                thread = consumerthread.consumer_thread(bootstrap_servers, topic, cpu_metric, ram_metric, ping_metric, incoming_traffic_metric, outcoming_traffic_metric)
                """

                thread = consumerthread.consumer_thread(bootstrap_servers, topic)
                
                threads.append(thread)
                thread.start()

            """
            # ogni minuto viene riorganizzata la dashboard in maniera tale che viene riorganizzata
            # in modo tale che i pannelli che presentano valori più alti finiscono in cima
            if (sortcount>=12):
                print(f"Riordino i grafici")
                graphs = graphsort.graph_sort()
                graphs.graph_sort_all(grafana_cred)
                sortcount = 0
            """

            sortcount +=1

    # In caso di interruzione del programma madre, tutti quanti i threads vengono terminati tramite 
    # il metodo stop_threads
            
    except KeyboardInterrupt:

        for thread in threads:
            if (thread.is_alive()):
                thread.stop_threads()
                working_threads_count += 1

        print(f"{working_threads_count} threads terminati")