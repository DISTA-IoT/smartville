.PHONY: all build-controller build-attacker build-victim

all: build-controller build-attacker build-victim

build-controller:
	docker build -t pox-controller -f controller.Dockerfile poxController/.

build-attacker:
	docker build -t attacker -f attacker.Dockerfile AttackerNode/.

build-victim:
	docker build -t victim -f victim.Dockerfile VictimNode/.
