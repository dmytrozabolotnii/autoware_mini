#!/bin/bash

host=$1                    # Carla host
port=$2                    # Carla port

# wait Carla to start
sleep 15

# launch the script
$CARLA_ROOT/PythonAPI/examples/generate_traffic.py --host $host --port $port --async
