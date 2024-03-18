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
from grafana_api.grafana_face import GrafanaFace
from prometheus_api_client import PrometheusConnect
from datetime import datetime, timedelta
import math

class graph_sort:

    # Metodo che invoca tutti quanti i metodi di riordinamento per ciascuna dashboard

    def graph_sort_all(self, grafana_cred):

        self.grafana_user = grafana_cred[0]
        self.grafana_pass = grafana_cred[1]

        self.graph_sort_one('CPU_data')
        self.graph_sort_one('RAM_data')
        self.graph_sort_one('PING_data')
        self.graph_sort_one('INT_data')
        self.graph_sort_one('ONT_data')

    def graph_sort_one(self, dash_UID):

        grafana = GrafanaFace(auth=(self.grafana_user, self.grafana_pass), host='localhost:3000')

        dashboard_config = grafana.dashboard.get_dashboard(dash_UID)

        # Ottengo tutti quanti i pannelli appartenenti alla Dashboard
        panels = dashboard_config['dashboard']['panels']

        for panel in panels:

            targets = panel.get('targets', [])

            if targets:

                target = targets[0]

                prometheus = PrometheusConnect('http://localhost:9090')
                
                prometheus_expr = target.get('expr')

                # Ottengo tutte le informazioni associate al pannello dell'ultimo minuto
                prometheus_expr_range = f"{prometheus_expr}[1m]"

                # Ottengo la lista contenente tutti quante le informazioni
                metric_data_range = prometheus.custom_query(query=prometheus_expr_range)

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

        updated_dashboard = grafana.dashboard.update_dashboard(dashboard_config)

