BUILD_CMD = docker build

all:customHost/Dockerfile \
	poxController/Dockerfile \

customHost: customHost/Dockerfile
	$(BUILD_CMD) --file $< --tag custom-host customHost
	@touch buildstatus

poxController: poxController/Dockerfile
	$(BUILD_CMD) --file $< --tag pox-controller poxController
	@touch buildstatus