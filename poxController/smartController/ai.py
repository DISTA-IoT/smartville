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