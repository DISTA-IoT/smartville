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
from prometheus_client import start_http_server, Gauge
from prometheus_api_client import PrometheusConnect
from smartController.consumer_thread import ConsumerThread
from smartController.dashgenerator import DashGenerator
from smartController.graphgenerator import GraphGenerator
from grafana_api.grafana_face import GrafanaFace

from confluent_kafka import KafkaException
from confluent_kafka.admin import AdminClient
import time
import socket
from collections import deque
import threading
import netifaces as ni

RAM = 'RAM'
CPU = 'CPU'
IN_TRAFFIC = 'IN_TRAFFIC'
OUT_TRAFFIC = 'OUT_TRAFFIC'
DELAY = 'DELAY'


def server_exist(bootstrap_servers):

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


class MetricsLogger: 

    def __init__(
            self, 
            server_addr = "192.168.1.1:9092",
            max_conn_retries = 5,
            metric_buffer_len = 10,
            grafana_user="admin", 
            grafana_pass="admin"):

        self.grafana_user = grafana_user
        self.grafana_pass = grafana_pass
        self.server_addr = server_addr
        self.topics = None
        self.topic_list = []
        self.threads = []
        self.working_threads_count = 0
        self.sortcount = 0
        self.kafka_admin_client = None
        self.max_conn_retries = max_conn_retries  # max Kafkfa connection retries.
        # self.metrics_dict = {}
        self.metric_buffer_len = metric_buffer_len
        self.accessible_ip = ni.ifaddresses('eth1')[ni.AF_INET][0]['addr']
        self.grafana_connection = GrafanaFace(
                auth=(self.grafana_user, self.grafana_pass), 
                host=self.accessible_ip+':3000')

        if self.init_kafka_connection(): 
            
            self.init_prometheus_server()
            
            self.dash_generator = DashGenerator(self.grafana_connection)

            self.graph_generator = GraphGenerator(
                grafana_connection=self.grafana_connection,
                prometheus_connection=self.prometheus_connection)
            
            self.consumer_thread_manager = threading.Thread(
                target=self.start_consuming, 
                args=())
            
            try:
                self.consumer_thread_manager.start()
            except KeyboardInterrupt:
                for thread in self.threads:
                    if (thread.is_alive()):
                        thread.stop_threads()
                        working_threads_count += 1
                print(f" Closed {working_threads_count} threads")

        else: print('MetricsLogger not attached!')
        

    def init_kafka_connection(self):
        retries = 0
        while retries < self.max_conn_retries: 
            if server_exist(self.server_addr):
                try:
                    conf = {'bootstrap.servers': self.server_addr}
                    self.kafka_admin_client = AdminClient(conf)
                    self.topics = self.kafka_admin_client.list_topics(timeout=5)
                    return True
                except KafkaException as e:
                    print(f"Kafka connection error {e}")
                    self.kafka_admin_client = None
                    return False
            else:
                print(f"Could not find Kafka server at {self.server_addr}")
                retries += 1
        return False
    

    def init_prometheus_server(self):
        start_http_server(port=8000, addr=self.accessible_ip)

        # Definizione metriche inserite su Prometheus
        self.cpu_metric = Gauge('CPU_percentage', 'Metrica CPU percentuale', ['label_name'])
        self.ram_metric = Gauge('RAM_GB', 'Metrica RAM', ['label_name'])
        self.ping_metric = Gauge('Latenza_ms', 'Metrica latenza del segnale', ['label_name'])
        self.incoming_traffic_metric = Gauge('Incoming_network_KB', 'Metrica traffico in entrata', ['label_name'])
        self.outcoming_traffic_metric = Gauge('Outcoming_network_KB', 'Metrica traffico in uscita', ['label_name'])
        
        # prometheus_connection will permit the graph generator 
        # organize graphs...  
        self.prometheus_connection = PrometheusConnect('http://'+self.accessible_ip+':9090/24):9090')


    def start_consuming(self):

        while True:
            updated_topic_list = []
            curr_topics_dict = self.kafka_admin_client.list_topics().topics

            # Inserimento topics in una lista di topics aggiornata
            for topic_name in curr_topics_dict.keys():
                if topic_name != '__consumer_offsets':
                    updated_topic_list.append(topic_name)

            # Creazione di una lista contenente i nuovi topics inseriti
            to_add_topic_list = list(set(updated_topic_list) - set(self.topic_list))

            # La lista di topics aggiornata prende il posto della lista di topics vecchia
            self.topic_list = updated_topic_list

            time.sleep(5)

            # Per ciascun topic nuovo, viene avviato un thread dedicato alla lettura delle metriche
            for topic_name in to_add_topic_list:

                self.graph_generator.generate_all_graphs(topic_name)

                """
                self.metrics_dict[topic_name] = {
                    CPU: deque(maxlen=self.metric_buffer_len), 
                    DELAY: deque(maxlen=self.metric_buffer_len), 
                    IN_TRAFFIC: deque(maxlen=self.metric_buffer_len), 
                    OUT_TRAFFIC: deque(maxlen=self.metric_buffer_len),
                    RAM: deque(maxlen=self.metric_buffer_len) }
                """

                print(f"Consumer Thread for topic {topic_name} commencing")
                thread = ConsumerThread(
                    self.server_addr, 
                    topic_name,
                    curr_topics_dict[topic_name],
                    self.cpu_metric,
                    self.ram_metric,
                    self.ping_metric,
                    self.incoming_traffic_metric,
                    self.outcoming_traffic_metric)

                self.threads.append(thread)
                thread.start()


            if (self.sortcount>=12):     # Ogni minuto (5 secs * 12)
                print(f"Organizing dashboard priorities...")
                self.graph_generator.sort_all_graphs()
                self.sortcount = 0

            self.sortcount +=1

    """
    def  update_cpu_metric(self, value, topic):
        self.metrics_dict[topic][CPU].append(value)

    def update_ram_metric(self, value, topic):
        self.metrics_dict[topic][RAM].append(value)

    def update_ping_metric(self, value, topic):
        self.metrics_dict[topic][DELAY].append(value)

    def update_incoming_traffic_metric(self, value, topic):
        self.metrics_dict[topic][IN_TRAFFIC].append(value)

    def update_outcoming_traffic_metric(self, value, topic):
        self.metrics_dict[topic][OUT_TRAFFIC].append(value)
    """