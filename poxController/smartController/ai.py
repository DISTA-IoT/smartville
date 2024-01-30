from pox.core import core
import random
import pox.openflow.libopenflow_01 as of
from pox.customScript.l3_learning_mod import openflow_connection

log = core.getLogger()
def ai_placeholder(packetlist,port):
  log.info(f"AI: RECEIVED {len(packetlist)} PACKETS FOR PORT: {port}")
  choose = random.choice([True, False])
  log.info(f"AI: I HAVE CHOSEN: {choose} FOR PORT: {port}")
  if choose:
    mitigate_attack(port)

#do some actions to mitigate the attack
def mitigate_attack(port):
  block_traffic(port)

def block_traffic(port):
  # Creating a flow rule to drop all packets coming in on the specified port
  msg = of.ofp_flow_mod()
  msg.match.in_port = port  # Replace with the desired input port
  msg.idle_timeout = 0  # Set to 0 for no idle timeout
  msg.hard_timeout = 0  # Set to 0 for no hard timeout
  openflow_connection.send(msg)
  log.info(f"SWITCH FLOW MOD SENT - BLOCKED PORT {port}")