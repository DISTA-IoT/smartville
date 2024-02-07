from confluent_kafka import Consumer, KafkaException
from confluent_kafka.admin import AdminClient
import threading
import math

class SimpleConsumerThread(threading.Thread):

    def __init__(self, bootstrap_servers, topic, metricslogger):
        threading.Thread.__init__(self)
        self.metricslogger = metricslogger
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.exit_signal = threading.Event()


    # Definizione metodo di arresto di un thread
    def stop_threads(self):
        print("Stopping threads...")
        self.exit_signal.set()
    

    def run(self):

        # Definizione metodi di connessione al server Kafka
        consumer_conf = {'bootstrap.servers': self.bootstrap_servers, 'group.id': 'my-group'}
        conf = {'bootstrap.servers': self.bootstrap_servers}

        consumer = Consumer(consumer_conf)
        admin_client = AdminClient(conf)

        delete_ok = False
        
        end_message = str(math.nan).encode('utf-8')
        stopper = 0       

        topics = admin_client.list_topics()

        # Estrazione numero delle partizioni dal topic
        extract_partitions = topics.topics[self.topic].partitions
        num_partitions = len(extract_partitions)

        # Controllo se il topic a cui ci si deve iscrivere sia corretto: viene considerata una condizione
        # sufficiente se il numero di partizioni all'interno del topic sono esattamente 5
        if (num_partitions==5):

            consumer.subscribe([self.topic])

            try:

                while not self.exit_signal.is_set():

                    # Il messaggio viene atteso per massimo 3 secondi
                    msg = consumer.poll(timeout=3)

                    #print(f'Got message: {msg.value().decode("utf-8")} from partition {msg.partition()}')

                    # Nel caso non arrivasse alcun messaggio, vengono inseriti valori nulli per le 
                    # rispettive metriche

                    if msg is None :
                        self.metricslogger.update_cpu_metric(end_message, self.topic)
                        self.metricslogger.update_ram_metric(end_message, self.topic)
                        self.metricslogger.update_ping_metric(end_message, self.topic)
                        self.metricslogger.update_incoming_traffic_metric(end_message, self.topic)
                        self.metricslogger.update_outcoming_traffic_metric(end_message, self.topic)

                        stopper += 1

                        # Nel caso non si ricevesse alcun messaggio per 40*3 = 120 secondi, viene interrotto
                        # il ciclo
                        if (stopper <40):
                            continue
                        else:
                            delete_ok = True
                            break

                    if msg.error():
                        if msg.error().code() == KafkaException._PARTITION_EOF:
                            print(f'Partizione {msg.partition()} terminata')
                            continue
                        else:
                            print(f'Errore: {msg.error()}')
                            break

                    # Lettura metriche dal server: ciascuna metrica ha la propria partizione dedicata
                    # Ciascuna metrica verrà monitorata tramite Prometheus e inserita nel suo sistema di
                    # archiviazione
                    if (msg.partition()==0):
                        print(f'Valore CPU letto da {self.topic}: {msg.value().decode("utf-8")}')
                        self.metricslogger.update_cpu_metric(float(msg.value()), self.topic)
                        stopper = 0

                    elif (msg.partition()==1):
                        print(f'Valore RAM letto da {self.topic}: {msg.value().decode("utf-8")}')
                        self.metricslogger.update_ram_metric(float(msg.value()), self.topic)
                        stopper = 0

                    elif (msg.partition()==2):
                        print(f'Valore latenza letto da {self.topic}: {msg.value().decode("utf-8")}')
                        self.metricslogger.update_ping_metric(float(msg.value()), self.topic)
                        stopper = 0

                    elif (msg.partition()==3):
                        print(f'Valore traffico rete in entrata letto da {self.topic}: {msg.value().decode("utf-8")}')
                        self.metricslogger.update_incoming_traffic_metric(float(msg.value()), self.topic)
                        stopper = 0

                    elif (msg.partition()==4):
                        print(f'Valore traffico rete in uscita letto da {self.topic}: {msg.value().decode("utf-8")}')
                        self.metricslogger.update_outcoming_traffic_metric(float(msg.value()), self.topic)
                        stopper = 0

            # Quando viene interrotto il ciclo, se ciò è dovuto alla mancanza di ricezione dei messaggi
            # per un periodo troppo lungo stabilito dalla variabile booleana "delete_ok", il thread viene 
            # arrestato e il topic dedicatogli viene eliminato in maniera tale da fare pulizia all'interno
            # del server kafka         
            finally:

                if delete_ok:
                    admin_client.delete_topics([self.topic])

                print(f"{self.topic}: thread arrestato")

        else:

            admin_client.delete_topics([self.topic])
            print(f"Utente {self.topic} registrato non correttamente")
