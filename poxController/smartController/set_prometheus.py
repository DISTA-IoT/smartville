
import netifaces as ni 


def get_source_ip_address():
    try:
        ip = ni.ifaddresses('eth1')[ni.AF_INET][0]['addr']
        return ip
    except ValueError:
        return "Interface not found"


def replace_string_in_file(file_name, old_string, new_string):
    try:
        # Read the content of the file
        with open(file_name, 'r') as file:
            file_content = file.read()

        # Replace the old string with the new string
        modified_content = file_content.replace(old_string, new_string)

        # Write the modified content back to the file
        with open(file_name, 'w') as file:
            file.write(modified_content)

        print(f"String '{old_string}' replaced with '{new_string}' in {file_name}")
    except FileNotFoundError:
        print(f"Error: File '{file_name}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    file_name = "pox/smartController/prometheus.yml"
    old_string = "localhost"
    new_string = get_source_ip_address()
    
    replace_string_in_file(file_name, old_string, new_string)