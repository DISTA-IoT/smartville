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
from collections import defaultdict

"""
LABELLING Strategy:

SmartVille implements a labeling strategy by assuming that each node performs one attack. 
By doing so, IP addresses are labelled using the name of the attack (in the multi-class classification setting)
Or simply "attack" in the binary classification setting.
"""
# IpV4 attackers (for training purposes) Also victim response flows are considered infected
CLASS_LABELS= defaultdict(lambda: "Bening") # class "bening" is default and is reserved for leggittimate traffic. 
ZDA_LABELS = defaultdict(lambda: False) 
TEST_ZDA_LABELS = defaultdict(lambda: False) 


#########################################################################################################
######################## TIGER CURRICULA (DEFAULT) ###################################################### 
#########################################################################################################
 

####################### Known Benign Traffic: 

# Victim-1
CLASS_LABELS["192.168.1.4"] = "Echo (Bening)"

####################### Known attacks: 

# attacker-7
CLASS_LABELS["192.168.1.10"] = "Hakai"

# attacker-8
CLASS_LABELS["192.168.1.11"] = "Torii"

####################### ZdAs group 1:
# Must use the G1 keyword!

# attacker-12
CLASS_LABELS["192.168.1.15"] = "Okiru (ZdA G1)"
ZDA_LABELS["192.168.1.15"] = True

# attacker-4
CLASS_LABELS["192.168.1.7"] = "CC_HeartBeat (ZdA G1)"
ZDA_LABELS["192.168.1.7"] = True

# attacker-5
CLASS_LABELS["192.168.1.8"] = "Gen_DDoS (ZdA G1)"
ZDA_LABELS["192.168.1.8"] = True


####################### ZdAs group 2:
# Must use the G2 keyword!

# attacker-9
CLASS_LABELS["192.168.1.12"] = "Mirai (ZdA G2)"
ZDA_LABELS["192.168.1.12"] = True
TEST_ZDA_LABELS["192.168.1.12"] = True

# attacker-10
CLASS_LABELS["192.168.1.13"] = "Gafgyt (ZdA G2)"
ZDA_LABELS["192.168.1.13"] = True
TEST_ZDA_LABELS["192.168.1.13"] = True

# attacker-11
CLASS_LABELS["192.168.1.14"] = "Hajime (ZdA G2)"
ZDA_LABELS["192.168.1.14"] = True
TEST_ZDA_LABELS["192.168.1.14"] = True

# attacker-6
CLASS_LABELS["192.168.1.9"] = "H_Scan (ZdA G2)"
ZDA_LABELS["192.168.1.9"] = True
TEST_ZDA_LABELS["192.168.1.9"] = True

# attacker-13
CLASS_LABELS["192.168.1.16"] = "Muhstik (ZdA G2)"
ZDA_LABELS["192.168.1.16"] = True
TEST_ZDA_LABELS["192.168.1.16"] = True


####################### Unknown benign: 
# Must use the G2 keyword!

# Victim-2
CLASS_LABELS["192.168.1.5"] = "Hue (Bening G2) "
ZDA_LABELS["192.168.1.5"] = True
TEST_ZDA_LABELS["192.168.1.5"] = True

# Victim-3
CLASS_LABELS["192.168.1.6"] = "Doorlock (Bening G2)"
ZDA_LABELS["192.168.1.6"] = True
TEST_ZDA_LABELS["192.168.1.6"] = True


# Victim-0
CLASS_LABELS["192.168.1.3"] = "Doorlock (Bening G2)"
ZDA_LABELS["192.168.1.3"] = True
TEST_ZDA_LABELS["192.168.1.3"] = True





#########################################################################################################
######################## ALTERNATE CURRICULUM 0 ######################################################### 
#########################################################################################################
 
# IpV4 attackers (for training purposes) Also victim response flows are considered infected
AC0_TRAINING_LABELS_DICT= defaultdict(lambda: "Bening") # class "bening" is default and is reserved for leggittimate traffic. 
AC0_ZDA_DICT = defaultdict(lambda: False) 
AC0_TEST_ZDA_DICT = defaultdict(lambda: False) 

####################### Benign Traffic: 
"""
# Victim-0
AC0_TRAINING_LABELS_DICT["192.168.1.3"] = "Doorlock (Bening)"

# Victim-1
AC0_TRAINING_LABELS_DICT["192.168.1.4"] = "Echo (Bening)"
AC0_ZDA_DICT["192.168.1.4"] = True
AC0_TEST_ZDA_DICT["192.168.1.4"] = True

# Victim-2
AC0_TRAINING_LABELS_DICT["192.168.1.5"] = "Hue (Bening)"
AC0_ZDA_DICT["192.168.1.5"] = True

# Victim-3
AC0_TRAINING_LABELS_DICT["192.168.1.6"] = "Doorlock (Bening)"
"""

####################### Known attacks: 

# attacker-4
AC0_TRAINING_LABELS_DICT["192.168.1.7"] = "CC_HeartBeat"

# attacker-5
AC0_TRAINING_LABELS_DICT["192.168.1.8"] = "Gen_DDoS"

# attacker-6
AC0_TRAINING_LABELS_DICT["192.168.1.9"] = "H_Scan"


####################### ZdAs group 1:
# attacker-7
AC0_TRAINING_LABELS_DICT["192.168.1.10"] = "Hakai (ZdA G1)"
AC0_ZDA_DICT["192.168.1.10"] = True

# attacker-8
AC0_TRAINING_LABELS_DICT["192.168.1.11"] = "Torii (ZdA G1)"
AC0_ZDA_DICT["192.168.1.11"] = True

# attacker-9
AC0_TRAINING_LABELS_DICT["192.168.1.12"] = "Mirai (ZdA G1)"
AC0_ZDA_DICT["192.168.1.12"] = True

# attacker-10
AC0_TRAINING_LABELS_DICT["192.168.1.13"] = "Gafgyt (ZdA G1)"
AC0_ZDA_DICT["192.168.1.13"] = True

####################### ZdAs group 2:

# attacker-11
AC0_TRAINING_LABELS_DICT["192.168.1.14"] = "Hajime (ZdA G2)"
AC0_ZDA_DICT["192.168.1.14"] = True
AC0_TEST_ZDA_DICT["192.168.1.14"] = True

# attacker-12
AC0_TRAINING_LABELS_DICT["192.168.1.15"] = "Okiru (ZdA G2)"
AC0_ZDA_DICT["192.168.1.15"] = True
AC0_TEST_ZDA_DICT["192.168.1.15"] = True

# attacker-13
AC0_TRAINING_LABELS_DICT["192.168.1.16"] = "Muhstik (ZdA G2)"
AC0_ZDA_DICT["192.168.1.16"] = True
AC0_TEST_ZDA_DICT["192.168.1.16"] = True



#########################################################################################################
######################## ALTERNATE CURRICULUM 1 ######################################################### 
#########################################################################################################
 
# IpV4 attackers (for training purposes) Also victim response flows are considered infected
AC1_TRAINING_LABELS_DICT= defaultdict(lambda: "Bening") # class "bening" is default and is reserved for leggittimate traffic. 
AC1_ZDA_DICT = defaultdict(lambda: False) 
AC1_TEST_ZDA_DICT = defaultdict(lambda: False) 

####################### Benign Traffic: 
"""
# Victim-0
AC1_TRAINING_LABELS_DICT["192.168.1.3"] = "Doorlock (Bening)"
AC1_ZDA_DICT["192.168.1.3"] = True
AC1_TEST_ZDA_DICT["192.168.1.3"] = True

# Victim-1
AC1_TRAINING_LABELS_DICT["192.168.1.4"] = "Echo (Bening)"

# Victim-2
AC1_TRAINING_LABELS_DICT["192.168.1.5"] = "Hue (Bening)"
AC1_ZDA_DICT["192.168.1.5"] = True

# Victim-3
AC1_TRAINING_LABELS_DICT["192.168.1.6"] = "Doorlock (Bening)"
AC1_ZDA_DICT["192.168.1.6"] = True
AC1_TEST_ZDA_DICT["192.168.1.6"] = True
"""

####################### Known attacks: 

# attacker-7
AC1_TRAINING_LABELS_DICT["192.168.1.10"] = "Hakai"

# attacker-8
AC1_TRAINING_LABELS_DICT["192.168.1.11"] = "Torii"

# attacker-12
AC1_TRAINING_LABELS_DICT["192.168.1.15"] = "Okiru"


####################### ZdAs group 1:

# attacker-4
AC1_TRAINING_LABELS_DICT["192.168.1.7"] = "CC_HeartBeat (ZdA G1)"
AC1_ZDA_DICT["192.168.1.7"] = True

# attacker-5
AC1_TRAINING_LABELS_DICT["192.168.1.8"] = "Gen_DDoS (ZdA G1)"
AC1_ZDA_DICT["192.168.1.8"] = True

# attacker-9
AC1_TRAINING_LABELS_DICT["192.168.1.12"] = "Mirai (ZdA G1)"
AC1_ZDA_DICT["192.168.1.12"] = True

# attacker-10
AC1_TRAINING_LABELS_DICT["192.168.1.13"] = "Gafgyt (ZdA G1)"
AC1_ZDA_DICT["192.168.1.13"] = True

####################### ZdAs group 2:

# attacker-11
AC1_TRAINING_LABELS_DICT["192.168.1.14"] = "Hajime (ZdA G2)"
AC1_ZDA_DICT["192.168.1.14"] = True
AC1_TEST_ZDA_DICT["192.168.1.14"] = True

# attacker-6
AC1_TRAINING_LABELS_DICT["192.168.1.9"] = "H_Scan (ZdA G2)"
AC1_ZDA_DICT["192.168.1.9"] = True
AC1_TEST_ZDA_DICT["192.168.1.9"] = True

# attacker-13
AC1_TRAINING_LABELS_DICT["192.168.1.16"] = "Muhstik (ZdA G2)"
AC1_ZDA_DICT["192.168.1.16"] = True
AC1_TEST_ZDA_DICT["192.168.1.16"] = True



#########################################################################################################
######################## ALTERNATE CURRICULUM 2 ######################################################### 
######################################################################################################### 

# IpV4 attackers (for training purposes) Also victim response flows are considered infected
AC2_TRAINING_LABELS_DICT= defaultdict(lambda: "Bening") # class "bening" is default and is reserved for leggittimate traffic. 
AC2_ZDA_DICT = defaultdict(lambda: False) 
AC2_TEST_ZDA_DICT = defaultdict(lambda: False) 


####################### Benign Traffic: 
"""
# Victim-0
AC2_TRAINING_LABELS_DICT["192.168.1.3"] = "Doorlock (Bening)"

# Victim-1
AC2_TRAINING_LABELS_DICT["192.168.1.4"] = "Echo (Bening)"
AC2_ZDA_DICT["192.168.1.4"] = True

# Victim-2
AC2_TRAINING_LABELS_DICT["192.168.1.5"] = "Hue (Bening)"
AC2_ZDA_DICT["192.168.1.5"] = True
AC2_TEST_ZDA_DICT["192.168.1.5"] = True

# Victim-3
AC2_TRAINING_LABELS_DICT["192.168.1.6"] = "Doorlock (Bening)"
"""

####################### Known attacks: 

# attacker-13
AC2_TRAINING_LABELS_DICT["192.168.1.16"] = "Muhstik"

# attacker-11
AC2_TRAINING_LABELS_DICT["192.168.1.14"] = "Hajime"

# attacker-9
AC2_TRAINING_LABELS_DICT["192.168.1.12"] = "Mirai"



####################### ZdAs group 1:

# attacker-4
AC2_TRAINING_LABELS_DICT["192.168.1.7"] = "CC_HeartBeat (ZdA G1)"
AC2_ZDA_DICT["192.168.1.7"] = True

# attacker-5
AC2_TRAINING_LABELS_DICT["192.168.1.8"] = "Gen_DDoS (ZdA G1)"
AC2_ZDA_DICT["192.168.1.8"] = True

# attacker-12
AC2_TRAINING_LABELS_DICT["192.168.1.15"] = "Okiru(ZdA G1)"
AC2_ZDA_DICT["192.168.1.15"] = True

# attacker-6
AC2_TRAINING_LABELS_DICT["192.168.1.9"] = "H_Scan (ZdA G1)"
AC2_ZDA_DICT["192.168.1.9"] = True

####################### ZdAs group 2:

# attacker-7
AC2_TRAINING_LABELS_DICT["192.168.1.10"] = "Hakai (ZdA G2)"
AC2_ZDA_DICT["192.168.1.10"] = True
AC2_TEST_ZDA_DICT["192.168.1.10"] = True

# attacker-10
AC2_TRAINING_LABELS_DICT["192.168.1.13"] = "Gafgyt (ZdA G2)"
AC2_ZDA_DICT["192.168.1.13"] = True
AC2_TEST_ZDA_DICT["192.168.1.13"] = True

# attacker-8
AC2_TRAINING_LABELS_DICT["192.168.1.11"] = "Torii (ZdA G2)"
AC2_ZDA_DICT["192.168.1.11"] = True
AC2_TEST_ZDA_DICT["192.168.1.11"] = True




#########################################################################################################
############# CUSTOM CURRICULUM (Used for custom/random traffic patterns fill as desired.)  ########### 
######################################################################################################### 

# IpV4 attackers (for training purposes) Also victim response flows are considered infected
CUSTOM_TRAINING_LABELS_DICT= defaultdict(lambda: "Bening") # class "bening" is default and is reserved for leggittimate traffic. 
CUSTOM_ZDA_DICT = defaultdict(lambda: False) 
CUSTOM_TEST_ZDA_DICT = defaultdict(lambda: False) 

####################### Known attacks: 

CUSTOM_TRAINING_LABELS_DICT["192.168.1.6"] = "Gen_DDoS"

####################### ZdAs group 1:

CUSTOM_TRAINING_LABELS_DICT["192.168.1.5"] = "Mirai (ZdA G1)"
CUSTOM_ZDA_DICT["192.168.1.5"] = True


####################### ZdAs group 2:

CUSTOM_TRAINING_LABELS_DICT["192.168.1.10"] = "Hakai (ZdA G2)"
CUSTOM_ZDA_DICT["192.168.1.10"] = True
CUSTOM_TEST_ZDA_DICT["192.168.1.10"] = True
