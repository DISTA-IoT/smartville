from prometheus_api_client.exceptions import PrometheusApiClientException

class GraphGenerator:
    """
    Questa classe è dedicata all'inserimento dei vari grafici su Grafana per la visualizzazione 
    interattiva. Ciò viene svolto tramite la libreria che permette la connessione all'host grafana mediante
    la sua chiave api messa a disposizione
    """
    def __init__(self, grafana_connection, prometheus_connection):
        self.num_panels = 0
        self.grafana_connection = grafana_connection
        self.prometheus_connection = prometheus_connection

    
    def generate_all_graphs(self, panel_title):
        """
        Controllo se i vari grafici sono già presenti nelle rispettive, 
        in tal caso non vengono inseriti nuoovamente.
        """

        if not self.graph_exists('CPU_data', panel_title):
            self.generate_graph('CPU_data', panel_title, 'CPU_percentage','semi-dark-yellow')

        if not self.graph_exists('RAM_data', panel_title):
            self.generate_graph('RAM_data', panel_title, 'RAM_GB','#315b2b')

        if not self.graph_exists('PING_data', panel_title):
            self.generate_graph('PING_data', panel_title, 'Latenza_ms','#00e674')

        if not self.graph_exists('INT_data', panel_title):
            self.generate_graph('INT_data', panel_title, 'Incoming_network_KB','#00bcff')

        if not self.graph_exists('ONT_data', panel_title):
            self.generate_graph('ONT_data', panel_title, 'Outcoming_network_KB','#0037ff')

    
    def graph_exists(self, dash_UID, panel_title):
        """
        Controllo esistenza grafici
        Controllo se il pannello è all'interno della lista, nel caso lo fosse, 
        viene restituito False
        """
        # Ottenimento della dashboard esistente
        dashboard = self.grafana_connection.dashboard.get_dashboard(dash_UID)
        # Estrazione di tutti i pannelli nella dashboard
        panel_titles = [panel['title'] for panel in dashboard['dashboard']['panels']]
        self.num_panels = len(panel_titles)
        
        return panel_title in panel_titles


    def generate_graph(self, dash_UID, panel_title, metric, colortab):
        """
        Inserimento di un grafico 
        params: 
            - stringhe relative al nome dei grafici, 
            - il nome del pannello,
            - l'UID che li identifica univocamente
            - il colore che questi assumeranno.
        """
        # Configurazione dashboard tramite la definizione del suo JSON model
        # print(f'generando il grafico per {panel_title}')
        panel_config = {
            "type": "timeseries",
            "title": f"{panel_title}",
            "uid": f"{panel_title}",
            "datasource": "prometheus",
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
                "x": (self.num_panels%3)*8, # la posizione dipende dai pannelli presenti
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
                    "expr": f'{metric}{{label_name="{panel_title}"}}',
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

        dashboard = self.grafana_connection.dashboard.get_dashboard(dash_UID)
        if dashboard is not None:
            # Aggiungi il pannello alla relativa dashboard
            dashboard['dashboard']['panels'].append(panel_config)
            self.grafana_connection.dashboard.update_dashboard(dashboard)
            print(f"Grafico '{panel_title}' aggiunto alla dashboard '{dash_UID}' con successo!!!!")
        else:
            print(f"Dashboard con UID '{dash_UID}' not presente.")

    
    def sort_all_graphs(self):
        """
        Riorganizzala dashboard in maniera tale che viene riorganizzata
        in modo tale che i pannelli che presentano valori più alti finiscono in cima
        """
        self.sort_single_graph('CPU_data')
        self.sort_single_graph('RAM_data')
        self.sort_single_graph('PING_data')
        self.sort_single_graph('INT_data')
        self.sort_single_graph('ONT_data')


    def sort_single_graph(self, dash_UID):

        dashboard_config = self.grafana_connection.dashboard.get_dashboard(dash_UID)
        # Ottengo tutti quanti i pannelli appartenenti alla Dashboard
        panels = dashboard_config['dashboard']['panels']

        for panel in panels:
            targets = panel.get('targets', [])
            if targets:
                target = targets[0]                
                prometheus_expr = target.get('expr')
                # Ottengo tutte le informazioni associate al pannello dell'ultimo minuto
                prometheus_expr_range = f"{prometheus_expr}[1m]"
                # Ottengo la lista contenente tutti quante le informazioni
                try:
                    metric_data_range = self.prometheus_connection.custom_query(query=prometheus_expr_range)                    
                except PrometheusApiClientException as e:
                    print(f"Error while querying prometheus! Query {prometheus_expr_range} | Error Message: {e}")
                    return
                values = []
                sum_value = 0

                if metric_data_range:
                    # Estraggo i valori effettivi
                    values_range = metric_data_range[0].get('values', [])
                    values = [float(point[1]) if point and len(point) > 1 and point[1] not in ('+Inf', '-Inf', 'NaN') else 0 for point in values_range] 
                    # Sommo i valori              
                    sum_value = sum(values) 
                # Associo la somma a ciascun pannello
                panel['sum_value'] = sum_value


        # Ordina i pannelli in base al valore somma che presentano
        sorted_panels = sorted(panels, key=lambda x: x.get('sum_value', 0), reverse=True)

        for i, panel in enumerate(sorted_panels):
            panel['gridPos'] = {
                "h": 6,
                "w": 8,
                "x": (i % 3)*8,
                "y": 0
            }

        # Aggiorna la dashboard con la configurazione stabilita
        dashboard_config['dashboard']['panels'] = sorted_panels
        self.grafana_connection.dashboard.update_dashboard(dashboard_config)

