# Evolving Poisoning Attacks in Federated Learning: An Experimental Path To Attack Optimization via Simulated Annealing

This repository contains the code and experiments for the Master's Degree Thesis in Cybersecurity at the University of Pisa: **"Evolving Poisoning Attacks in Federated Learning: An Experimental Path To Attack Optimization via Simulated Annealing"**.

## 📖 Abstract
Federated Learning (FL) has emerged as a powerful paradigm for the distributed training of machine learning models, offering significant privacy advantages by avoiding the centralization of sensitive data. However, its decentralized architecture exposes it to critical security threats, particularly poisoning attacks conducted by malicious clients aimed at corrupting the global model.
The primary objective of this thesis is to undertake a structured experimental path that, starting from the analysis of basic FL system vulnerabilities, leads to the definition of an optimized attack strategy capable of inflicting significant damage to the model while simultaneously evading server-side defense mechanisms. The case study selected for the experiments is an air quality time-series forecasting system (PM 2.5), implemented using Long Short-Term Memory (LSTM) neural networks and the Flower federated framework. 
Initial analysis demonstrated that while basic poisoning attacks, such as injecting random noise into local models, have catastrophic effects on unprotected systems, they are entirely ineffective when the server implements similarity-based or anomaly-detection defenses (such as K-means clustering or centroid distance-based filters).
Consequently, to succeed in a protected environment, an attacker must calibrate a complex trade-off between attack effectiveness and stealthiness. The central contribution of this work lies in the application of the Simulated Annealing (SA) optimization algorithm for performing advanced model poisoning attacks. The algorithm is employed to explore the weight space and dynamically calculate the optimal perturbation that maximizes the malicious objective while constraining the update to remain below the detection threshold imposed by the defense.
Experimental results confirm the high threat level of this methodology: a strong adversary, employing SA for an untargeted attack, can systematically bypass server-side defense mechanisms, causing a severe degradation in the model’s general predictive performance. On the other hand, simulations conducted for targeted attack scenarios revealed that achieving similarly catastrophic results is a significantly more complex process, strictly limited by the statistical asymmetry of the data, thus preparing the ground for necessary future studies.

## 🛠️ Tools & Frameworks
* **Federated Learning Framework:** [Flower (flwr)](https://flower.ai/)
* **Deep Learning Library:** [PyTorch](https://pytorch.org/) (LSTM Neural Networks)

## 📊 Dataset & Task
The system performs one-hour-ahead time-series forecasting of **PM 2.5** air pollution. 
The experiments are based on the [Beijing Multi-Site Air Quality Dataset](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data), distributing data from 5 different weather stations to 5 simulated federated clients.

## ⚔️ Implemented Attacks and Defenses

### Attacks
The repository includes various adversarial strategies (executed by 1 attacker out of 5 clients):
* **Data Poisoning:** Label Flipping, Feature Bias Injection, Temporal Block Shuffling, Target Shuffling.
* **Basic Model Poisoning:** Gradient Ascent, Sign-Flipping, Naive/Systematic Random Noise Injection.
* **Advanced Evasion Attacks:** Adaptive Head-Only Noise Injection (with Brute-force $k_{max}$ Grid Search).
* **Optimized Attacks:** 
  * *Untargeted SA-based Attack:* Maximizes overall model degradation while evading defenses.
  * *Targeted SA-based Attack:* Maximizes error on specific sequences while preserving/minimize error on other sequences.

### Server-Side Defenses
* **Centroid Distance-based Defense:** Discards client updates that exceed a dynamic distance threshold from the round's centroid.
* **Clustering-based Defense (K-Means):** Partitions updates into two clusters and drops the minority cluster.

## 📂 Repository Structure
* `LSTMfederated/task.py`: Data loading, LSTM model definition, and train/test routines (including some malicious logic).
* `LSTMfederated/client_app.py`: Flower ClientApp logic, handling local training and adversarial manipulations (e.g., Noise Injection, Sign-Flipping).
* `LSTMfederated/server_app.py`: Flower ServerApp logic, implementing `FedAvg` and custom defense strategies (`DistanceBasedDefenseStrategy`, `DistanceBasedDefenseStrategyWithSA`, etc.).
* `pyproject.toml`: Configuration file defining federation parameters, number of rounds, learning rates, and malicious client assignments.
* `BaselineCode/`: Contains the clean base code without adversarial behaviors.
* `aotizhongxin/`, `changping/`, `dingling/`, `dongsi/`, `guanyuan/`: Contain the preprocessed datasets for the 5 monitoring stations.
* `SequencesBackdoorSA`: Contains the csv files with high-pollution and low-pollution sequences used to run the targeted SA-based attacks.
* `sa_selective_backdoor_results.txt`, `sa_selective_backdoor_with_defense_results.txt`: Contain the results, in terms of MSE, of the targeted attack execution with and without the server-side centroid distance-based defense.
* `setup_client_node.sh`, `setup_server_node.sh`, `upload_on_nodess.sh`: Scripts useful to run the experiment on the cluster.
* `start_local_simulation.sh`: Script useful to execute the experiment locally, running the 5 clients and the server on different terminals.

## 🚀 How to Run the Experiments
A comprehensive, step-by-step guide to setting up and executing the federated learning experiments is provided in the thesis document.
