from grafana_api.grafana_face import GrafanaFace
import netifaces as ni 


def get_source_ip_address():
    try:
        ip = ni.ifaddresses('eth1')[ni.AF_INET][0]['addr']
        return ip
    except ValueError:
        return "Interface not found"


def delete_datasources_by_name(grafana_connection, datasource_name):
    print('Trying to delete previous datasource named prometheus')
    try:
        response = grafana_connection.datasource.delete_datasource_by_name(datasource_name)
        print(response['message'])
    except Exception as grafana_error:
        print('Cannot delete datasource: ', grafana_error) 

def add_datasource(grafana_connection, datasource_config):
    response = grafana_connection.datasource.create_datasource(datasource_config)
    if response['message'] == 'Datasource added':
        print('Data source added successfully.')
    else:
        print(f'Failed to add data source. Error message: {response["message"]}')


if __name__ == "__main__":
    dynamic_IP = get_source_ip_address()

    grafana_url = f'{dynamic_IP}:3000'  # URL where Grafana is hosted
    prometheus_url = f'http://{dynamic_IP}:9090/24):9090'
    datasource_config = {
        "name": "prometheus",
        "type": "prometheus",
        "url": prometheus_url,  # URL of the Prometheus server
        "access": "direct",  # Access mode (proxy or direct)
        "basicAuth": True,  # Whether to use basic authentication
        "isDefault": True  # Whether to set this data source as default
        # Add other configuration options as needed
    }


    grafana_connection = GrafanaFace(auth=('admin', 'admin'), host=grafana_url)

    delete_datasources_by_name(grafana_connection, "prometheus")

    add_datasource(grafana_connection, datasource_config)