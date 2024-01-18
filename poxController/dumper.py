from pox.core import core
# implemented in l3_learning_mod


log = core.getLogger()

class PacketLogger(object):
    def __init__(self):
        core.openflow.addListeners(self)

    def _handle_PacketIn(self, event):
        packet = event.parsed
        log.info("Received packet:")
        log.info("  Parsed packet object: %s", packet)
        log.info("  Attributes of parsed packet:")
        
        # Iterate over the attributes of event.parsed
        for attr_name in dir(packet):
            if not attr_name.startswith("__"):  # Skip internal attributes
                attr_value = getattr(packet, attr_name)
                log.info("    %s: %s", attr_name, attr_value)

def launch():
    core.registerNew(PacketLogger)
