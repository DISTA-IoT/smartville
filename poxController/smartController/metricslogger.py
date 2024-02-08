from prometheus_client import start_http_server, Gauge
from smartController.simple_consumer_thread import SimpleConsumerThread
from confluent_kafka import KafkaException
from confluent_kafka.admin import AdminClient
import time
import socket
from collections import deque

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
            metric_buffer_len = 40):

        self.server_addr = server_addr
        self.topics = None
        self.topicslist = []
        self.threads = []
        self.working_threads_count = 0
        self.sortcount = 0
        self.admin = None
        self.max_conn_retries = max_conn_retries #max Kafkfa connection retries.
        self.metrics_dict = {}
        self.metric_buffer_len = metric_buffer_len
        

        if self.init_connection(): 
            try:
                self.init_prometheus_server()
                self.start_consuming()
            except KeyboardInterrupt:
                for thread in self.threads:
                    if (thread.is_alive()):
                        thread.stop_threads()
                        working_threads_count += 1
                print(f" Closed {working_threads_count} threads")

        else: print('MetricsLogger not attached!')
        

    def init_connection(self):
        retries = 0
        while retries < self.max_conn_retries: 
            if server_exist(self.server_addr):
                try:
                    #consumer_conf = {'bootstrap.servers': bootstrap_servers, 'group.id': 'my-group'}
                    conf = {'bootstrap.servers': self.server_addr}
                    #consumer = Consumer(consumer_conf)
                    self.admin = AdminClient(conf)
                    self.topics = self.admin.list_topics(timeout=5)
                    return True
                except KafkaException as e:
                    print(f"Kafka connection error {e}")
                    self.admin = None
                    return False
            else:
                print(f"Could not find Kafka server at {self.server_addr}")
                retries += 1
        return False
    

    def init_prometheus_server(self):
        start_http_server(8000)
        # Definizione metriche inserite su Prometheus
        self.cpu_metric = Gauge('CPU_percentage', 'Metrica CPU percentuale', ['label_name'])
        self.ram_metric = Gauge('RAM_GB', 'Metrica RAM', ['label_name'])
        self.ping_metric = Gauge('Latenza_ms', 'Metrica latenza del segnale', ['label_name'])
        self.incoming_traffic_metric = Gauge('Incoming_network_KB', 'Metrica traffico in entrata', ['label_name'])
        self.outcoming_traffic_metric = Gauge('Outcoming_network_KB', 'Metrica traffico in uscita', ['label_name'])
        



    def start_consuming(self):

        while True:
            updated_topicslist = []
            updated_topics = self.admin.list_topics()

            # Inserimento topics in una lista di topics aggiornata
            for topic, _ in updated_topics.topics.items():
                if topic != '__consumer_offsets':
                    updated_topicslist.append(topic)

            topicslist_set = set(self.topicslist)
            updated_topicslist_set = set(updated_topicslist)

            # Creazione di una lista contenente i nuovi topics inseriti
            new_topicslist_set = updated_topicslist_set - topicslist_set
            new_topicslist = list(new_topicslist_set)

            # Tramite questo passaggio, la lista di topics aggiornata prende il posto della lista di topics vecchia
            # Ciò permetterà poi di essere confrontata in un prossimo ciclo con la futura lista dei topics
            # aggiornata, in modo tale da verificare ogni volta se sono stati inseriti nuovi topics 
            self.topicslist = updated_topicslist

            time.sleep(5)

            # Per ciascun topic nella nuova lista di topics, viene avviato un thread dedicato alla lettura
            # delle metriche in esso contenuto
            for topic in new_topicslist:
                self.metrics_dict[topic] = {
                    CPU: deque(maxlen=self.metric_buffer_len), 
                    DELAY: deque(maxlen=self.metric_buffer_len), 
                    IN_TRAFFIC: deque(maxlen=self.metric_buffer_len), 
                    OUT_TRAFFIC: deque(maxlen=self.metric_buffer_len),
                    RAM: deque(maxlen=self.metric_buffer_len) }

                print(f"Consumer Thread for topic {topic} commencing")
                thread = SimpleConsumerThread(self.server_addr, topic, self)
                self.threads.append(thread)
                thread.start()

            self.sortcount +=1


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