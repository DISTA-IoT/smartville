from grafana_api.grafana_face import GrafanaFace
import netifaces as ni



IFACE_NAME = 'eth1'


def get_source_ip_address(interface=IFACE_NAME):
    try:
        ip = ni.ifaddresses(interface)[ni.AF_INET][0]['addr']
        return ip
    except ValueError:
        return "Interface not found"
    

accesible_ip = get_source_ip_address()

print(accesible_ip)

a = GrafanaFace(
                auth=("admin", "admin"), 
                host='http://192.168.122.170:3000')
print(a)