from grafana_api.grafana_face import GrafanaFace

num_panels = 0

# Questa classe è dedicata all'inserimento dei vari grafici su Grafana per la visualizzazione 
# interattiva. Ciò viene svolto tramite la libreria che permette la connessione all'host grafana mediante
# la sua chiave api messa a disposizione

class graph_generator:

    def graph_gen(self, name, grafana_cred):

        # Passaggio della variabile contenente il nome del nuovo utente per cui devono essere creati i vari
        # grafici

        self.name = name
        self.grafana_user = grafana_cred[0]
        self.grafana_pass = grafana_cred[1]

        # Controllo se i vari grafici sono già presenti nelle rispettive, in tal caso non vengono inseriti
        # nuovamente, ciò avviene tramite il metodo "graph_check".
        # L'inserimento di una grafico avviene tramite la chiamata ai vari metodi "graph_gen", ai quali vengono
        # passate in ingresso le stringhe relative al nome dei grafici, l'UID che li identifica univocamente
        # e il colore che questi assumeranno

        new_graph = self.graph_check('CPU_data')

        if (new_graph):
            self.graph_gen_single('CPU_data','CPU_percentage','semi-dark-yellow')

        new_graph = self.graph_check('RAM_data')

        if (new_graph):
            self.graph_gen_single('RAM_data','RAM_GB','#315b2b')

        new_graph = self.graph_check('PING_data')

        if (new_graph):
            self.graph_gen_single('PING_data','Latenza_ms','#00e674')

        new_graph = self.graph_check('INT_data')

        if (new_graph):
            self.graph_gen_single('INT_data','Incoming_network_KB','#00bcff')

        new_graph = self.graph_check('ONT_data')

        if (new_graph):
            self.graph_gen_single('ONT_data','Outcoming_network_KB','#0037ff')

    # Definizione metodo di controllo esistenza grafici
    
    def graph_check(self, dash_UID):

        grafana = GrafanaFace(auth=(self.grafana_user, self.grafana_pass), host='localhost:3000')

        # Ottenimento della dashboard esistente

        dashboard = grafana.dashboard.get_dashboard(dash_UID)

        # Estrazione di tutti i pannelli nella dashboard

        panel_titles = [panel['title'] for panel in dashboard['dashboard']['panels']]

        global num_panels
        num_panels = len(panel_titles)

        # Definizione del nome il pannello che si sta cercando

        table_title = self.name  

        # Controllo se il pannello è all'interno della lista, nel caso lo fosse, viene restituita la 
        # variabile vera

        table_exists = table_title in panel_titles

        if table_exists:
            #print(f"Il grafico '{table_title}' esiste.")
            return False
        else:
            #print(f"Il grafico '{table_title}' non esiste.")
            return True



    def graph_gen_single(self, dash_UID, metric, colortab):

        # Connessione all'host Grafana con le relative credenziali

        grafana = GrafanaFace(auth=(self.grafana_user, self.grafana_pass), host='localhost:3000')

        # Configurazione dashboard tramite la definizione del suo JSON model

        panel_config = {
            "type": "timeseries",
            "title": f"{self.name}",
            "uid": f"{self.name}",
            "datasource": "Prometheus",
            "transparent": True,
            "fieldConfig": {
                "defaults": {
                    "color": {
                        "fixedColor": colortab,
                        "mode": "fixed"
                    },
                    "custom": {
                        "axisCenteredZero": False,
                        "axisColorMode": "text",
                        "axisLabel": "",
                        "axisPlacement": "auto",
                        "barAlignment": 0,
                        "drawStyle": "line",
                        "fillOpacity": 50,
                        "gradientMode": "none",
                        "hideFrom": {
                            "legend": False,
                            "tooltip": False,
                            "viz": False
                        },
                        "insertNulls": False,
                        "lineInterpolation": "linear",
                        "lineWidth": 5,
                        "pointSize": 5,
                        "scaleDistribution": {
                            "type": "linear"
                        },
                        "showPoints": "auto",
                        "spanNulls": False,
                        "stacking": {
                            "group": "A",
                            "mode": "none"
                        },
                        "thresholdsStyle": {
                            "mode": "off"
                        }
                    },
                    "mappings": [],
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {
                                "color": "green",
                                "value": None
                            },
                            {
                                "color": "red",
                                "value": 80
                            }
                        ]
                    }
                },
                "overrides": []
            },
            "gridPos": {
                "h": 6,
                "w": 8,
                "x": (num_panels%3)*8, # la posizione dipende dai pannelli presenti
                "y": 0
            },
            "id": None,
            "options": {
                "legend": {
                    "calcs": [],
                    "displayMode": "list",
                    "placement": "bottom",
                    "showLegend": False
                },
                "tooltip": {
                    "mode": "single",
                    "sort": "none"
                }
            },
            "targets": [
                {
                    "datasource": "prometheus",
                    "disableTextWrap": False,
                    "editorMode": "builder",
                    "expr": f'{metric}{{label_name="{self.name}"}}',
                    "fullMetaSearch": False,
                    "includeNullMetadata": True,
                    "instant": False,
                    "legendFormat": "__auto",
                    "range": True,
                    "refId": "A",
                    "useBackend": False
                }
            ]
        }

        dashboard = grafana.dashboard.get_dashboard(dash_UID)

        if dashboard is not None:

            # Aggiunge il pannello alla relativa dashboard

            dashboard['dashboard']['panels'].append(panel_config)
            updated_dashboard = grafana.dashboard.update_dashboard(dashboard)
            #print(f"Grafico '{self.name}' aggiunto alla dashboard '{dash_UID}' con successo!")

        else:
            print(f"Dashboard con UID '{dash_UID}' not presente.")


    



        
