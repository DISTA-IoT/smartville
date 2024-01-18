BUILD_CMD = docker build

all:customHost/Dockerfile \
	poxController/Dockerfile \
	myRouter2/Dockerfile \
    mySDNswitch2/Dockerfile \
    IoT_device_number6/Dockerfile \
    IoT_device_number7/Dockerfile \
    IoT_device_number8/Dockerfile\
    MQTT_broker/Dockerfile

customHost: customHost/Dockerfile
	$(BUILD_CMD) --file $< --tag custom-host customHost
	@touch buildstatus

poxController: poxController/Dockerfile
	$(BUILD_CMD) --file $< --tag pox-controller poxController
	@touch buildstatus

router: myRouter2/Dockerfile
	$(BUILD_CMD) --file $< --tag my-router2-mt myRouter2
	@touch buildstatus


sdnswitch: mySDNswitch2/Dockerfile
	$(BUILD_CMD) --file $< --tag my-sdnswitch2-mt mySDNswitch2
	@touch buildstatus

iot-device_n6:IoT_device_number6/Dockerfile
	$(BUILD_CMD) --file $< --tag iot-device-n6-mt IoT_device_number6
	@touch buildstatus

iot_device_n7:IoT_device_number7/Dockerfile
	$(BUILD_CMD) --file $< --tag iot-device-n7-mt IoT_device_number7
	@touch buildstatus

iot_device_n8:IoT_device_number8/Dockerfile
	$(BUILD_CMD)  --file $< --tag iot-device-n8-mt IoT_device_number8
	@touch buildstatus

my_mqtt_broker:MQTT_broker/Dockerfile
	$(BUILD_CMD)  --file $< --tag mqtt-broker2-mt MQTT_broker
	@touch buildstatus


clean:
	rm -rf my-router2-mt my-sdnswitch2-mt iot-device-n6-mt iot-device-n7-mt iot-device-n8-my buildstatus

