from grafana_api.grafana_face import GrafanaFace


def get_source_ip_address(interface=IFACE_NAME):
    try:
        ip = ni.ifaddresses(interface)[ni.AF_INET][0]['addr']
        return ip
    except ValueError:
        return "Interface not found"

class dash_generator:
    """
    Classe dedicata all'inserimento delle varie dashboard su Grafana per la visualizzazione 
    interattiva. Ciò viene svolto tramite la libreria che permette la connessione all'host grafana mediante
    la sua chiave api messa a disposizione
    """

    def __init__(self, grafana_user="admin", grafana_pass="admin"):
        self.grafana_user = grafana_user
        self.grafana_pass = grafana_pass
        self.dash_gen()


    def dash_gen(self):
        grafana_check = False       
        while not grafana_check:
            
            grafana = GrafanaFace(
                auth=(self.grafana_user, self.grafana_pass), 
                host='localhost:3000')

            new_dash = self.dash_check('CPU_data')

            if (new_dash):
                print("Dashboard non esistente, creazione in corso...")
                grafana_check = self.dash_gen_single_first('CPU_data','CPU (%)')
            else:
                grafana_check = True

        # Controllo se le vari grafici dedicati all'utente sono già presenti, in tal caso non vengono inseriti
        # nuovamente, ciò avviene tramite il metodo "dash_check".
        # L'inserimento di una dashboard avviene tramite la chiamata ai vari metodi "dash_gen", ai quali vengono
        # passate in ingresso le stringhe relative al nome della dashboard e l'UID che le identifica univocamente

        new_dash = self.dash_check('RAM_data')

        if (new_dash):
            print("Dashboard non esistente, creazione in corso...")
            self.dash_gen_single('RAM_data','RAM (GB)')

        new_dash = self.dash_check('PING_data')

        if (new_dash):
            print("Dashboard non esistente, creazione in corso...")
            self.dash_gen_single('PING_data','Latenza (ms)')

        new_dash = self.dash_check('INT_data')

        if (new_dash):
            print("Dashboard non esistente, creazione in corso...")
            self.dash_gen_single('INT_data','Traffico rete in entrata (KBps)')

        new_dash = self.dash_check('ONT_data')

        if (new_dash):
            print("Dashboard non esistente, creazione in corso...")
            self.dash_gen_single('ONT_data','Traffico rete in uscita (KBps)')

        grafana_cred = [self.grafana_user, self.grafana_pass]
        return grafana_cred

    # Definizione metodo di controllo esistenza dashboard
    
    def dash_check(self, dash_UID):

        grafana = GrafanaFace(auth=(self.grafana_user, self.grafana_pass), host='localhost:3000')

        try:

        # Tentativo di connessione alla dashboard tramite l'UID

            dashboard = grafana.dashboard.get_dashboard(dash_UID)
            return False
        except Exception as e:

            # Nel caso la connessione non andasse a buon fine, allora significa che la dashboard 
            # non è esistente 

            return True

    # Creazione di una nuova dashboard
    
    def dash_gen_single_first(self, dash_UID, dash_name):

        # Connessione all'host Grafana con le relative credenziali

        grafana = GrafanaFace(auth=(self.grafana_user, self.grafana_pass), host='localhost:3000')

        DASHBOARD_UID = 'CPU_data'

        # Configurazione dashboard tramite la definizione del suo JSON model

        dashboard_config = {
            "dashboard": {
                "uid": dash_UID,
                "title": dash_name,
                "panels": [],
                "refresh": "5s",
                "time": {
                    "from": "now-15m",
                    "to": "now"
                },
            },
            "overwrite": False
        }

        # Creazione effettiva dashboard 

        try:
            created_dashboard = grafana.dashboard.update_dashboard(dashboard_config)
            print(f"Dashboard con UID '{dash_UID}' creata con successo!")
            return True

        except Exception as e:
            print(f"Errore: ricontrolla username o password")
            return False

    
    def dash_gen_single(self, dash_UID, dash_name):

        # Connessione all'host Grafana con le relative credenziali

        grafana = GrafanaFace(auth=(self.grafana_user, self.grafana_pass), host='localhost:3000')

        DASHBOARD_UID = 'CPU_data'

        # Configurazione dashboard tramite la definizione del suo JSON model

        dashboard_config = {
            "dashboard": {
                "uid": dash_UID,
                "title": dash_name,
                "panels": [],
                "refresh": "5s",
                "time": {
                    "from": "now-15m",
                    "to": "now"
                },
            },
            "overwrite": False
        }

        # Creazione effettiva dashboard 

        created_dashboard = grafana.dashboard.update_dashboard(dashboard_config)
        print(f"Dashboard con UID '{dash_UID}' creata con successo!")