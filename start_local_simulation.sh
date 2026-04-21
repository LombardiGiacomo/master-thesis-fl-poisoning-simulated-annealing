#!/bin/bash

BASE_PORT=9094
NUM_NODES=5

########################################
# 1️) SuperLink
########################################
osascript <<EOF
tell application "Terminal"
  do script "source ~/venvs/flwr/bin/activate; cd ~/Desktop/fl_project; flower-superlink --insecure;"
end tell
EOF

sleep 2

########################################
# 2️) SuperNodes
########################################
for ((i=0; i<NUM_NODES; i++)); do
  PORT=$((BASE_PORT+i))

  osascript <<EOF
tell application "Terminal"
  do script "source ~/venvs/flwr/bin/activate; cd ~/Desktop/fl_project; flower-supernode --insecure --superlink 127.0.0.1:9092 --clientappio-api-address 127.0.0.1:$PORT --node-config \"partition-id=$i num-partitions=5\";"
end tell
EOF
done

sleep 3

########################################
# 3️) flwr run
########################################
osascript <<EOF
tell application "Terminal"
  do script "source ~/venvs/flwr/bin/activate; cd ~/Desktop/fl_project; flwr run . local-deployment --stream --run-config \"k-noise=0\";"
end tell
EOF
