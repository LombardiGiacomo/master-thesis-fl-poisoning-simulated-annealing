#!/bin/bash

cd ~/Desktop

for NODE in N05 N06 N07 N08 N09
do
    echo "Uploading to $NODE..."
    rsync -av --delete --exclude ".flwr" --exclude "Dataset" --exclude "DataPreprocessing.ipynb" --exclude "centralized_baseline.py" --exclude "final_model.pt" \
            --exclude "centralized_model.pt" --exclude "start_local_simulation.sh" --exclude "upload_on_nodes.sh" \
            fl_project/ ${NODE}:~/fl_project/
done