il server kafka deve essere lanciato su un'immagine docker
per farlo, dopo aver navigato nell directory, bisogna digitare su bash la stringa: 
"docker-compose up -d"

per costruire l'immagine del produttore:
"docker build -t producer ."

il file eseguibile producer.py deve essere lanciato su un'immagine docker tramite la stringa:
"docker run --network username_kafka-net -d -it producer"
rimpiazzare username con il nome del proprio utente docker

Se si vogliono lanciare una moltitudine di produttori, bisogna inserire nel file run_producer.sh il nome della propria rete e il numero di produttori che si vogliono lanciare.
Dunque si procede con:
"chmod +x run_producer.sh"
"./run_producer.sh"


il consumatore corre invece sulla macchina host
per far correre il programma bisogna prima accettarsi di aver installato Prometheus e Grafana
per prometheus utilizzare la stringa:
"wget https://github.com/prometheus/prometheus/releases/download/v2.30.3/prometheus-2.30.3.linux-amd64.tar.gz
tar -xzf prometheus-2.30.3.linux-amd64.tar.gz"
Si muovono i file binari prometheus nelle cartelle specifiche
"sudo mv prometheus-2.30.3.linux-amd64/prometheus /usr/local/bin/"
"sudo mv prometheus-2.30.3.linux-amd64/promtool /usr/local/bin/"

E dunque, si definiscono i job di scraping di prometheus sul file yml
"sudo nano /etc/prometheus/prometheus.yml"
Per le impostazioni job di scraping di prometheus:
- Fissare scrape ed evaluation interval a 5 secondi
- Nella voce scrape_config, impostare "system_metrics" in job_name e ["localhost:8000"] in targets

Dunque si fa correre il server prometheus:
"sudo /usr/local/prometheus/prometheus --config.file=/etc/prometheus/prometheus.yml --storage.tsdb.path=/var/lib/prometheus/"

Mentre per installare grafana:
"sudo apt-get install -y software-properties-common"
"sudo add-apt-repository "deb https://packages.grafana.com/oss/deb stable main" "
"sudo apt-get update"
"sudo apt-get install -y grafana"

Dunque si fa correre il server grafana, navigando prima nella sua directory:
"cd /usr/share/grafana"
"sudo ./bin/grafana-server web"

Per installare le librerie necessarie per eseguire il programma produttore/consumatore:
"pip3 install confluent_kafka"
"pip3 install psutil"
"pip3 install prometheus-client"
"pip3 install prometheus-api-client"
"pip3 intall grafana_api"
Nel caso il programma consumatore non riuscisse a connettersi all'api di Grafana:
"pip3 install git+https://github.com/mikayel/grafana_api.git"

Per fare correre il programma consumatore:
"python3 consumerall.py"
Per visuallizzare i grafici su sistema locale:
"http://localhost:3000"


