from grafana_api.grafana_face import GrafanaFace
import netifaces as ni

IFACE_NAME = 'eth1'


def get_source_ip_address(interface=IFACE_NAME):
    try:
        ip = ni.ifaddresses(interface)[ni.AF_INET][0]['addr']
        return ip
    except ValueError:
        return "Interface not found"

class DashGenerator:
    """
    Classe dedicata all'inserimento delle varie dashboard su Grafana per la visualizzazione 
    interattiva. Ciò viene svolto tramite la libreria che permette la connessione all'host grafana mediante
    la sua chiave api messa a disposizione
    """

    def __init__(self, grafana_user="admin", grafana_pass="admin"):
        self.grafana_user = grafana_user
        self.grafana_pass = grafana_pass
        self.accesible_ip = get_source_ip_address()
        self.grafana_object = GrafanaFace(
                auth=(self.grafana_user, self.grafana_pass), 
                host=self.accesible_ip+':3000')
        self.generate_all_dashes()


    def generate_all_dashes(self):
        while True:
            if not self.dashboard_exists('CPU_data'):
                print("Dashboard non esistente, creazione in corso...")
                self.generate_single_dash_with_return('CPU_data','CPU (%)')
            else:
                break

        """
        Controllo se i vari grafici dedicati all'utente sono già presenti, in tal caso non vengono inseriti
        nuovamente, ciò avviene tramite il metodo dashboard_exists.
        L'inserimento di una dashboard avviene tramite la chiamata ai vari metodi "dash_gen", ai quali vengono
        passate in ingresso le stringhe relative al nome della dashboard e l'UID che le identifica univocamente
        """
        if not self.dashboard_exists('RAM_data'):
            print("Dashboard RAM_data non esistente, creazione in corso...")
            self.generate_single_dash('RAM_data','RAM (GB)')

        if not self.dashboard_exists('PING_data'):
            print("Dashboard PING_data non esistente, creazione in corso...")
            self.generate_single_dash('PING_data','Latenza (ms)')

        if not self.dashboard_exists('INT_data'):
            print("Dashboard INT_data non esistente, creazione in corso...")
            self.generate_single_dash('INT_data','Traffico rete in entrata (KBps)')

        if not self.dashboard_exists('ONT_data'):
            print("Dashboard ONT_data non esistente, creazione in corso...")
            self.generate_single_dash('ONT_data','Traffico rete in uscita (KBps)')

    
    def dashboard_exists(self, dash_UID):
        """
        Metodo di controllo esistenza dashboard
        """
        try:
            # Tentativo di connessione alla dashboard tramite l'UID
            self.grafana_object.dashboard.get_dashboard(dash_UID)
            return True
        except Exception:
            # Nel caso la connessione non andasse a buon fine, allora significa che la dashboard 
            # non è esistente 
            return False


    def generate_single_dash_with_return(self, dash_UID, dash_name):
        """ 
        Creazione di una nuova dashboard
        """
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
            self.grafana_object.dashboard.update_dashboard(dashboard_config)
            print(f"Dashboard con UID '{dash_UID}' creata con successo!")
            return True

        except Exception as e:
            print(f"Errore: {e} ricontrolla username o password")
            return False

    
    def generate_single_dash(self, dash_UID, dash_name):

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
        self.grafana_object.dashboard.update_dashboard(dashboard_config)
        print(f"Dashboard con UID '{dash_UID}' creata con successo!")