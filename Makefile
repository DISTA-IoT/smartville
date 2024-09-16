# Makefile

# Load the .env file if it exists
ifneq (,$(wildcard .env))
    include .env
    export
endif

.PHONY: all build-controller build-attacker build-victim

all: build-controller build-attacker build-victim build-openvswitch

build-controller:
	docker build --build-arg GIT_USERNAME=$(GIT_USERNAME) --build-arg GIT_TOKEN=$(GIT_TOKEN) --build-arg WANDB_API_KEY=$(WANDB_API_KEY) -t pox-controller -f poxController/controller.Dockerfile poxController/.

build-attacker:
	docker build -t attacker -f attacker.Dockerfile AttackerNode/.

build-victim:
	docker build -t victim -f victim.Dockerfile VictimNode/.

build-botmaster:
	docker build --build-arg GIT_USERNAME=$(GIT_USERNAME) --build-arg GIT_TOKEN=$(GIT_TOKEN) -t botmaster -f BotMasterNode/botmaster.Dockerfile BotMasterNode/.

build-openvswitch:
	docker build -t openvswitch -f openvswitch.Dockerfile openSwitch/.
