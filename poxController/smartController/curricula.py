from collections import defaultdict


#########################################################################################################
######################## ALTERNATE CURRICULUM 0 ######################################################### 
#########################################################################################################
 
# IpV4 attackers (for training purposes) Also victim response flows are considered infected
AC0_TRAINING_LABELS_DICT= defaultdict(lambda: "Bening") # class "bening" is default and is reserved for leggittimate traffic. 
AC0_ZDA_DICT = defaultdict(lambda: False) 
AC0_TEST_ZDA_DICT = defaultdict(lambda: False) 

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