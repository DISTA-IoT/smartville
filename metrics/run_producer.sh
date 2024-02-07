image_name="producer"
options="--network jofoxy_kafka-net -d -it"  #sostituire il nome della propria rete

# Avvia il comando 200 volte
for i in {1..200}; do
    docker run $options $image_name
done