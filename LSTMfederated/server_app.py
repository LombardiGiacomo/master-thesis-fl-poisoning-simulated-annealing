import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from LSTMfederated.task import Net, test

import numpy as np
import logging
from typing import Iterable
from flwr.common.logger import log
from flwr.common import Message
from sklearn.cluster import KMeans

# =========================================================================================================
#                               STRATEGIES IMPLEMENTING DEFENSE MECHANISMS                                        
# =========================================================================================================

class DistanceBasedDefenseStrategy(FedAvg):
    """ 
        Difesa basata sulla distanza tra l'update del client e un centroide calcolato come media tra gli update inviati dai client.
        Se la distanza è maggiore di una soglia, il client è considerato sospetto e il suo update non viene usato nell'aggregazione in quel round.
        Questa classe eredita da FedAvg: questa strategia utilizzerà FedAvg per l'aggregazione finale, ma filtra i modelli prima che vengano passati a FedAvg.
    """
    def __init__(self, *args, initial_global_sd=None, malicious_id=3, **kwargs):
        super().__init__(*args, **kwargs)
        self._global_sd = None  # Global model state dictionary (necessary to compute the updates)
        # The Flower server receives from the clients the new models (v_local),
        # but the anomaly detection is implemented considering the updates of each client (Delta_v = v_local - v_global)

        # Parameters to compute security metrics
        self.malicious_id = malicious_id
        self.metrics = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}     # Dictionary to accumulate counters over rounds

        if initial_global_sd is not None:
            self._global_sd = {k: v.detach().cpu().clone() for k, v in initial_global_sd.items()}

    def aggregate_train(self, server_round: int, replies: Iterable[Message]):
        replies_list = list(replies)

        if not replies_list:
            return super().aggregate_train(server_round, replies_list)  # Fallback on standard FedAvg

        if self._global_sd is None:
            log(logging.WARNING, f"[ROUND {server_round}] Global state missing: fallback on FedAvg without defense.")
            return super().aggregate_train(server_round, replies_list)  # Fallback on standard FedAvg

        log(logging.INFO, f"=== [ROUND {server_round}] DISTANCE-BASED DEFENSE STRATEGY ===")

        client_updates = []
        client_ids = []

        # 1. Compute the update of each client: Delta_i = v_local_i - v_global
        for i, msg in enumerate(replies_list):
            local_sd = msg.content["arrays"].to_torch_state_dict()
            delta_sd = {}

            for k in local_sd.keys():
                if torch.is_floating_point(local_sd[k]):
                    delta_sd[k] = local_sd[k].detach().cpu() - self._global_sd[k].detach().cpu()

            client_updates.append(delta_sd)
            pid = int(msg.content["metrics"]["partition-id"])
            client_ids.append(pid)

        # 2. Compute the cetroid of the updates
        centroid = {}
        keys = client_updates[0].keys()
        for k in keys:
            stacked = torch.stack([cu[k] for cu in client_updates])
            centroid[k] = torch.mean(stacked, dim=0)

        # 3. Compute the distance of each update from the centroid (in update space)
        distances = []
        for cu in client_updates:
            dist_sq = 0.0
            for k in keys:
                dist_sq += torch.sum((cu[k] - centroid[k]) ** 2).item()
            distances.append(float(np.sqrt(dist_sq)))

        # 4. Compute the threshold
        mean_dist = float(np.mean(distances))
        std_dist = float(np.std(distances))
        threshold = mean_dist + (1.0 * std_dist)

        # 5. Filter clients' updates: all updates whose distance from the centroid is higher than the threshold are not
        #    considereed in the aggregation process
        accepted_replies = []   # 
        for i, dist in enumerate(distances):
            client_id = client_ids[i]
            if dist <= threshold:
                accepted_replies.append(replies_list[i])
                log(logging.INFO, f" [+] Client {client_id} ACCEPTED (Dist: {dist:.4f} <= Threshold: {threshold:.4f})")

                if client_id == self.malicious_id:
                    self.metrics["FN"] += 1
                    log(logging.WARNING, f" [!] FALSE NEGATIVE: Attacker {client_id} ACCEPTED!")
                else:
                    self.metrics["TN"] += 1
                    log(logging.INFO, f" [+] TRUE NEGATIVE: Honest client {client_id} ACCEPTED")
            else:
                log(logging.WARNING, f" [!] Client {client_id} REJECTED (Dist: {dist:.4f} > Threshold: {threshold:.4f}) -> Suspected attacker!")
                if client_id == self.malicious_id:
                    self.metrics["TP"] += 1
                    log(logging.INFO, f" [+] TRUE POSITIVE: Attacker {client_id} REJECTED")
                else:
                    self.metrics["FP"] += 1
                    log(logging.WARNING, f" [!] FALSE POSITIVE: Honest client {client_id} REJECTED!")

        # Compute the security metrics updated at the current round
        TP, TN = self.metrics["TP"], self.metrics["TN"]
        FP, FN = self.metrics["FP"], self.metrics["FN"]
        
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0.0  # False Positive Rate
        FNR = FN / (FN + TP) if (FN + TP) > 0 else 0.0  # False Negative Rate

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f_measure = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        log(logging.INFO, f"--- SECURITY METRICS (Round {server_round}) ---")
        log(logging.INFO, f"Total values accumulated -> TP:{TP} | TN:{TN} | FP:{FP} | FN:{FN}")
        log(logging.INFO, f"Precision: {precision:.4f} ({precision:.2%})")
        log(logging.INFO, f"Recall:    {recall:.4f} ({recall:.2%})")
        log(logging.INFO, f"F-measure: {f_measure:.4f}")

        log(logging.INFO, "=============================================")

        # 6. Aggregate local models of accepted clients
        aggregated = super().aggregate_train(server_round, accepted_replies)

        # 7. Update _global_sd to be used in the next round for computing the model updates of each client
        agg_msg = aggregated[0] if isinstance(aggregated, tuple) else aggregated    # super().aggregate_train() could return a tuple
        try:
            if agg_msg is not None and hasattr(agg_msg, "content") and "arrays" in agg_msg.content:
                new_global = agg_msg.content["arrays"].to_torch_state_dict()
                self._global_sd = {k: v.detach().cpu().clone() for k, v in new_global.items()}
        except Exception as e:
            log(logging.WARNING, f"global_sd update not possible: {e}")

        return aggregated
    

class KMeansDefenseStrategy(FedAvg):
    """ Difesa basata su clustering con K-Means (k = 2). 
        Ad ogni round, il server crea due cluster utilizzando gli update inviati dai server: considera malevoli i client che finiscono nel clsuter 
        di minoranza e non li utilizza per l'aggregazione nel round.
    """
    def __init__(self, *args, initial_global_sd=None, malicious_id=3, **kwargs):
        super().__init__(*args, **kwargs)
        self._global_sd = None  # Global model state dictionary (necessary to compute the updates)
        # The Flower server receives from the clients the new models (v_local),
        # but the anomaly detection is implemented considering the updates of each client (Delta_v = v_local - v_global)

        # Parameters to compute security metrics
        self.malicious_id = malicious_id
        self.metrics = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}     # Dictionary to accumulate counters over rounds

        if initial_global_sd is not None:
            self._global_sd = {k: v.detach().cpu().clone() for k, v in initial_global_sd.items()}

    def aggregate_train(self, server_round: int, replies: Iterable[Message]):
        replies_list = list(replies)

        if len(replies_list) < 3:   # At least 3 clients are necessary to consider the majority cluster as benign
            log(logging.WARNING, f"[ROUND {server_round}] Too few clients: fallback on FedAvg without defense.")
            return super().aggregate_train(server_round, replies_list)  # Fallback on standard FedAvg

        if self._global_sd is None:
            log(logging.WARNING, f"[ROUND {server_round}] Global state missing: fallback on FedAvg without defense.")
            return super().aggregate_train(server_round, replies_list)  # Fallback on standard FedAvg

        log(logging.INFO, f"=== [ROUND {server_round}] CLUSTERING-BASED DEFENSE STRATEGY WITH K-MEANS ===")

        client_vectors = []
        client_ids = []
        client_state_dicts = []

        # 1. Compute the update of each client: Delta_i = v_local_i - v_global
        #    Updates are flatten to be passed to K-means
        for i, msg in enumerate(replies_list):
            state_dict = msg.content["arrays"].to_torch_state_dict()
            client_state_dicts.append(state_dict)

            flattened_update = np.concatenate([
                (state_dict[k].detach().cpu() - self._global_sd[k].detach().cpu()).numpy().ravel()
                for k in state_dict.keys()
                if torch.is_floating_point(state_dict[k])
            ]).astype(np.float64)

            client_vectors.append(flattened_update)

            pid = int(msg.content["metrics"]["partition-id"])
            client_ids.append(pid)

        X = np.array(client_vectors, dtype=np.float64)

        # 2. KMeans (K=2)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        # 3. Identify benign cluster as the one with the majority of clients
        counts = np.bincount(labels, minlength=2)
        benign_cluster_id = int(np.argmax(counts))

        # 4. Filter clients' updates: all updates not belonging to the benign cluster are considered malicious
        #    and eliminated from the aggregation process
        accepted_replies = []
        for i, label in enumerate(labels):
            client_id = client_ids[i]
            if int(label) == benign_cluster_id:
                accepted_replies.append(replies_list[i])
                log(logging.INFO, f" [+] Client {client_id} ACCEPTED (Cluster {label})")

                if client_id == self.malicious_id:
                    self.metrics["FN"] += 1
                    log(logging.WARNING, f" [!] FALSE NEGATIVE: Attacker {client_id} ACCEPTED!")
                else:
                    self.metrics["TN"] += 1
                    log(logging.INFO, f" [+] TRUE NEGATIVE: Honest client {client_id} ACCEPTED")
            else:
                log(logging.WARNING, f" [!] Client {client_id} REJECTED (Cluster {label}) -> Suspected attacker!")
                if client_id == self.malicious_id:
                    self.metrics["TP"] += 1
                    log(logging.INFO, f" [+] TRUE POSITIVE: Attacker {client_id} REJECTED")
                else:
                    self.metrics["FP"] += 1
                    log(logging.WARNING, f" [!] FALSE POSITIVE: Honest client {client_id} REJECTED!")

        # Compute the security metrics updated at the current round
        TP, TN = self.metrics["TP"], self.metrics["TN"]
        FP, FN = self.metrics["FP"], self.metrics["FN"]
        
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        FNR = FN / (FN + TP) if (FN + TP) > 0 else 0.0

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f_measure = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        log(logging.INFO, f"--- SECURITY METRICS (Round {server_round}) ---")
        log(logging.INFO, f"Total values accumulated -> TP:{TP} | TN:{TN} | FP:{FP} | FN:{FN}")
        log(logging.INFO, f"Precision: {precision:.4f} ({precision:.2%})")
        log(logging.INFO, f"Recall:    {recall:.4f} ({recall:.2%})")
        log(logging.INFO, f"F-measure: {f_measure:.4f}")

        log(logging.INFO, "=============================================")

        # 5. Aggregate local models of accepted clients
        aggregated = super().aggregate_train(server_round, accepted_replies)

        # 7. Update _global_sd to be used in the next round for computing the model updates of each client
        agg_msg = aggregated[0] if isinstance(aggregated, tuple) else aggregated
        try:
            if agg_msg is not None and hasattr(agg_msg, "content") and "arrays" in agg_msg.content:
                new_global = agg_msg.content["arrays"].to_torch_state_dict()
                self._global_sd = {k: v.detach().cpu().clone() for k, v in new_global.items()}
        except Exception as e:
            log(logging.WARNING, f"global_sd update not possible: {e}")

        return aggregated
    
# =========================================================================================================
#                               STRATEGIES SIMULATING KMAX COMPUTATION                                        
# =========================================================================================================

def flatten_update_head_only(sd_local, sd_global) -> np.ndarray:
    """Flatten degli update (Delta = w_local - w_global) dei pesi sul solo layer head (layer lineare finale).
    Fuori dalla head mette zeri, così la dimensione resta identica a flatten_update().
    """
    head_updates_flatten = []
    for layer_i in sd_local.keys():
        if torch.is_floating_point(sd_local[layer_i]):
            if "head" in layer_i:
                update_flatten = (sd_local[layer_i].detach().cpu() - sd_global[layer_i].detach().cpu()).numpy().ravel()
            else:
                update_flatten = np.zeros(sd_local[layer_i].numel(), dtype=np.float32)
            head_updates_flatten.append(update_flatten)
    return np.concatenate(head_updates_flatten).astype(np.float64)


def make_noise_flat_like_head_only(sd_local, seed: int) -> np.ndarray:
    """Rumore deterministico con stessa dimensione del flatten completo,ma non nullo solo nei parametri della head."""
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    parts = []
    for i in sd_local.keys():
        if torch.is_floating_point(sd_local[i]):
            if "head" in i:
                noise = torch.randn(sd_local[i].shape, generator=g, dtype=sd_local[i].dtype, device=sd_local[i].device)
                parts.append(noise.numpy().ravel())
            else:
                parts.append(np.zeros(sd_local[i].numel(), dtype=np.float32))
    return np.concatenate(parts).astype(np.float64)

def _make_seed(seed_base: int, partition_id: int) -> int:
    return int(seed_base) + int(partition_id)


def flatten_update(sd_local, sd_global) -> np.ndarray:
    """Flatten degli update di tutti i layer della rete """
    return np.concatenate(
        [
            (sd_local[k].detach().cpu() - sd_global[k].detach().cpu()).numpy().ravel()
            for k in sd_local.keys()
            if torch.is_floating_point(sd_local[k])
        ]
    ).astype(np.float64)


def make_noise_flat_like(sd_local, seed: int) -> np.ndarray:
    """Rumore deterministico (CPU) con shape dei tensori float nello stesso ordine."""
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))

    parts = []
    for k in sd_local.keys():
        if torch.is_floating_point(sd_local[k]):
            shape = tuple(sd_local[k].detach().cpu().shape)
            noise = torch.randn(shape, generator=g, device="cpu", dtype=torch.float32)
            parts.append(noise.numpy().ravel())
    return np.concatenate(parts).astype(np.float64)


def benign_cluster_by_majority(labels: np.ndarray) -> int:
    counts = np.bincount(labels, minlength=2)
    return int(np.argmax(counts))


def client_is_benign_after_refit(X: np.ndarray, idx: int, d: np.ndarray, k: float, random_state=42) -> bool:
    """ Rifitta KMeans su Xk (con la riga idx avvelenata) e verifica se idx cade nel cluster benigno.
        In pratica, risponde alla domanda: se l'attaccante si muove di k in questa direzione, la difesa lo scopre?
    """
    Xk = X.copy()   # Copia dei dati
    Xk[idx] = X[idx] + k * d    # Applica il rumore, moltiplicato per k, alla riga dell'attaccante

    # Lancio K-Means su Xk 
    km = KMeans(n_clusters=2, random_state=random_state, n_init=10)
    labels = km.fit_predict(Xk)

    benign_id = benign_cluster_by_majority(labels)  # benign_id contiene l'id del cluster con piu client
    return labels[idx] == benign_id     # True se l'attaccante appartiene al cluster maggioritario, altrimenti false 


def estimate_kmax_refit_kmeans(
    X: np.ndarray,
    idx: int,
    d: np.ndarray,
    k_hi_init: float = 1.0,
    k_hi_max: float = 256.0,
    grid_N: int = 80,
    refine_steps: int = 3,
    random_state: int = 42,
) -> float:
    """Stima k_max con ricerca 1D, rifittando KMeans ogni volta."""
    if not client_is_benign_after_refit(X, idx, d, 0.0, random_state):
        return 0.0

    # 1) Trova upper bound dove diventa non benigno: parte da k=1, raddoppia fino a quando attaccante è benigno, 
    #    e si ferma quando trova un k che non passa piu. 
    k_hi = float(k_hi_init)
    while k_hi <= k_hi_max and client_is_benign_after_refit(X, idx, d, k_hi, random_state):
        k_hi *= 2.0
    if k_hi > k_hi_max:
        return float("inf")
    # k_hi contiene l'ultimo valore di k buono

    # 2) GRIGLIA GROSSOLANA tra 0 e k_hi
    ks = np.linspace(0.0, k_hi, grid_N)
    flags = np.array(
        [client_is_benign_after_refit(X, idx, d, float(k), random_state) for k in ks],
        dtype=bool,
    )

    # Trova gli indici di TUTTI i k che sono passati (le "zone sicure") tra 0 e k_hi
    true_idxs = np.where(flags)[0]
    if len(true_idxs) == 0:
        return 0.0

    # Prendiamo last_true come l'ultimo k passato nell'intervallo 0-k_hi
    last_true = int(true_idxs[-1])
    if last_true == len(ks) - 1:
        return float(ks[-1])

    k_lo = float(ks[last_true])
    k_hi2 = float(ks[last_true + 1])

    # 3) raffinamenti locali
    for _ in range(refine_steps):
        ks2 = np.linspace(k_lo, k_hi2, grid_N)
        last_ok = k_lo
        for k in ks2:
            if client_is_benign_after_refit(X, idx, d, float(k), random_state):
                last_ok = float(k)
            else:
                k_lo, k_hi2 = last_ok, float(k)
                break

    return k_lo
# Limit nella funzione_kmax_refit_kemeans: si ferma al primo k dove l'attaccante non è più benigno.
# Cioè assume che client_is_benign_after_refit(k) sia tipo: True True True False False False
# Invece con KMeans quella funzione può essere tipo: True True False False True False, cioè non monotona.
# Questo concetto funziona bene con la Distance-based defense strategy ma non con KMeans.

def estimate_absolute_kmax_refit_kmeans(
    X: np.ndarray,
    idx: int,
    d: np.ndarray,
    k_limit: float = 30.0,    # Esplora SEMPRE fino a questo valore
    grid_N: int = 200,        
    refine_steps: int = 3,
    random_state: int = 42,
) -> float:
    """Stima il k_max ASSOLUTO con ricerca globale"""
    
    # Se con rumore 0.0 siamo già fuori, c'è un problema di base
    if not client_is_benign_after_refit(X, idx, d, 0.0, random_state):
        return 0.0

    # 1) GRIGLIA GLOBALE: Esploriamo tutto lo spazio da 0 a k_limit
    ks = np.linspace(0.0, k_limit, grid_N)
    
    # Interroghiamo l'oracolo per tutti i 200 punti della griglia
    flags = np.array(
        [client_is_benign_after_refit(X, idx, d, float(k), random_state) for k in ks],
        dtype=bool,
    )
    
    # Trova gli indici di TUTTI i k che sono passati (le "zone sicure")
    true_idxs = np.where(flags)[0]
    if len(true_idxs) == 0:
        return 0.0

    # 2) Prendiamo l'ultimo valore True (la zona sicura più lontana in assoluto)
    last_true_idx = true_idxs[-1]
    
    # Se l'ultimo valore True è proprio il limite massimo imposto
    if last_true_idx == len(ks) - 1:
        return float(ks[-1])

    # Sappiamo che il confine dell'ultima zona sicura è tra l'ultimo True e il primo False successivo
    k_lo = float(ks[last_true_idx])
    k_hi = float(ks[last_true_idx + 1])

    # 3) RAFFINAMENTO LOCALE: Facciamo lo zoom solo su quest'ultimo confine lontano
    for _ in range(refine_steps):
        ks2 = np.linspace(k_lo, k_hi, grid_N)
        last_ok = k_lo
        for k in ks2:
            if client_is_benign_after_refit(X, idx, d, float(k), random_state):
                last_ok = float(k)
            else:
                k_lo, k_hi = last_ok, float(k)
                break

    return k_lo

class KMeansDefenseStrategyWithKmaxComputation(FedAvg):
    """KMeans su update Δ; stampa k_max simulato per il client malevolo (target). Poi li salva su un file.
        Viene calcolato ad ogni round il valore di k_max che consente al client malevolo di restare dentro il cluster benigno.
    """
    def __init__(self, *args, initial_global_sd=None, kmax_target_id: int = 3, noise_seed_base: int = 1337, **kwargs):
        super().__init__(*args, **kwargs)
        self._global_sd = None
        self._kmax_target_id = int(kmax_target_id)
        self._noise_seed_base = int(noise_seed_base)
        self._kmax_log_file = "kmax_history3.csv"    # File in cui salviamo i valori di kmax ad ogni round

        if initial_global_sd is not None:
            self._global_sd = {k: v.detach().cpu().clone() for k, v in initial_global_sd.items()}

        # Crea o sovrascrive il file
        with open(self._kmax_log_file, "w") as f:
            f.write("round,kmax\n")

    def aggregate_train(self, server_round: int, replies: Iterable[Message]):
        replies_list = list(replies)

        if len(replies_list) < 3:
            return super().aggregate_train(server_round, replies_list)

        if self._global_sd is None:
            log(logging.WARNING, "Global state unavailable: fallback to FedAvg.")
            return super().aggregate_train(server_round, replies_list)

        log(logging.INFO, f"=== [ROUND {server_round}] KMEANS-BASED DEFENSE + KMAX COMPUTATION FOR CLIENT {self._kmax_target_id} AS ATTACKER ===")

        client_ids: list[int] = []
        client_state_dicts = []
        client_updates = []

        # 1) Costruisci X = update flatten
        for msg in replies_list:
            sd_local = msg.content["arrays"].to_torch_state_dict()
            pid = int(msg.content["metrics"]["partition-id"])

            client_ids.append(pid)
            client_state_dicts.append(sd_local)
            client_updates.append(flatten_update(sd_local, self._global_sd))

        X = np.array(client_updates, dtype=np.float64)

        # 2) SIMULA E INIETTA k_max per il client target
        target_id = self._kmax_target_id
        if target_id in client_ids:
            idx = client_ids.index(target_id)
            pid = target_id

            #delta_norm = float(np.linalg.norm(X[idx]) + 1e-12)
            #seed = _make_seed(self._noise_seed_base, pid)
#
            #noise_flat = make_noise_flat_like(client_state_dicts[idx], seed=seed)
            #noise_norm = float(np.linalg.norm(noise_flat) + 1e-12)
#
            #d = (delta_norm / noise_norm) * noise_flat

            seed = _make_seed(self._noise_seed_base, pid)

            delta_head = flatten_update_head_only(client_state_dicts[idx], self._global_sd)
            delta_norm = float(np.linalg.norm(delta_head) + 1e-12)

            noise_flat = make_noise_flat_like_head_only(client_state_dicts[idx], seed=seed)
            noise_norm = float(np.linalg.norm(noise_flat) + 1e-12)

            d = (delta_norm / noise_norm) * noise_flat

            # Trova il massimo k prima di essere rilevato
            #kmax = estimate_kmax_refit_kmeans(X, idx, d)
            kmax = estimate_absolute_kmax_refit_kmeans(X, idx, d)   # Versione con ricerca "globale"
            log(logging.INFO, f"[ROUND {server_round}] k_max simulation (client {pid}): {kmax:.6f}")

            # Salva nel CSV il valore di k_max nel round corrente
            with open(self._kmax_log_file, "a") as f:
                f.write(f"{server_round},{kmax}\n")

            # --- Iniezione dell'attacco ---
            if kmax > 0.0:
                # Applica un margine di sicurezza per essere certi di non essere scartati
                margin = 0.99 
                k_attack = kmax * margin
                
                # Calcola il nuovo update avvelenato appiattito
                poisoned_flat_update = X[idx] + (k_attack * d)
                X[idx] = poisoned_flat_update   # Aggiorniamo X per il KMeans del server

                # Dobbiamo ricostruire i tensori PyTorch per rimpiazzare il messaggio
                poisoned_sd = {}
                offset = 0
                original_sd = client_state_dicts[idx]
                
                for k in original_sd.keys():
                    if torch.is_floating_point(original_sd[k]):
                        shape = original_sd[k].shape
                        numel = original_sd[k].numel()
                        
                        # Estrae la porzione di update piatto
                        flat_tensor_update = poisoned_flat_update[offset : offset + numel]
                        offset += numel
                        
                        # Riformatta il tensore 1D nella forma originale
                        tensor_update = torch.from_numpy(flat_tensor_update).view(shape).float()
                        
                        # w_poisoned = w_global + poisoned_update
                        poisoned_sd[k] = self._global_sd[k] + tensor_update
                    else:
                        poisoned_sd[k] = original_sd[k]

                # Rimpiazza i pesi avvelenati direttamente nel contenuto del messaggio originale
                replies_list[idx].content["arrays"] = ArrayRecord(poisoned_sd)
                
                log(logging.INFO, f"[ROUND {server_round}] INJECTED attack with k={k_attack:.6f}")

        else:
            log(logging.INFO, f"[ROUND {server_round}] Target client {target_id} not present.")

        # 3) Difesa reale: KMeans su update reali (che ora include quello avvelenato)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        benign_cluster_id = benign_cluster_by_majority(labels)

        accepted_replies = []
        for i, label in enumerate(labels):
            pid = client_ids[i]
            if int(label) == benign_cluster_id:
                accepted_replies.append(replies_list[i])
                log(logging.INFO, f" [+] Client {pid} ACCEPTED (Cluster {label})")
            else:
                log(logging.WARNING, f" [!] Client {pid} REJECTED (Cluster {label}) -> Suspected attacker!")

        log(logging.INFO, "=============================================")

        aggregated = super().aggregate_train(server_round, accepted_replies)

       # 6) Aggiorna global_sd per calcolare in modo corretto gli update ai round successivi
        agg_msg = aggregated[0] if isinstance(aggregated, tuple) else aggregated

        try:
            if agg_msg is not None and hasattr(agg_msg, "content") and "arrays" in agg_msg.content:
                new_global = agg_msg.content["arrays"].to_torch_state_dict()
                self._global_sd = {k: v.detach().cpu().clone() for k, v in new_global.items()}
        except Exception as e:
            log(logging.WARNING, f"global_sd update not possible: {e}")

        return aggregated


def client_is_benign_after_refit_distance(X: np.ndarray, idx: int, d: np.ndarray, k: float, z: float = 1.0) -> bool:
    """Simula la DistanceBasedDefenseStrategy: verifica se il client malevolo rientra in mu + z*sigma."""
    Xk = X.copy()
    Xk[idx] = X[idx] + k * d

    # 1. Compute the new centroid 
    centroid = np.mean(Xk, axis=0)

    # 2. Compute the distances of all the clients from the centroid
    distances = np.linalg.norm(Xk - centroid, axis=1)

    # 3. Compute the new threshold
    mean_dist = float(np.mean(distances))
    std_dist = float(np.std(distances))
    threshold = mean_dist + (z * std_dist)

    # 4. Verify if the attacker is accepted from the defense
    return distances[idx] <= threshold

def estimate_kmax_refit_distance(
    X: np.ndarray,
    idx: int,
    d: np.ndarray,
    z: float = 1.0,             
    k_hi_init: float = 1.0,     # Initial point to search the upper-bound
    k_hi_max: float = 512.0,    # Upper limit
    grid_N: int = 100,          # Number of pieces of the coarse grid
    refine_steps: int = 3,      # Number of steps for the fine search
) -> float:
    """Stima k_max per la DistanceBasedDefenseStrategy con ricerca 1D iterativa."""
    if not client_is_benign_after_refit_distance(X, idx, d, 0.0, z):
        return 0.0

    # 1. Find the upper-bound (where the attacker reaches the threshold)
    k_hi = k_hi_init
    while k_hi <= k_hi_max and client_is_benign_after_refit_distance(X, idx, d, k_hi, z):
        k_hi *= 2.0
    if k_hi > k_hi_max:
        return k_hi_max

    # 2. Coarse grid
    ks = np.linspace(0.0, k_hi, grid_N)
    flags = np.array([client_is_benign_after_refit_distance(X, idx, d, float(k), z) for k in ks],dtype=bool)
    true_idxs = np.where(flags)[0]
    if len(true_idxs) == 0:
        return 0

    last_true = int(true_idxs[-1])
    if last_true == len(ks) - 1:
        return float(ks[-1])

    # Higher boundary
    k_lo_refined = float(ks[last_true])
    k_hi_refined = float(ks[last_true + 1])

    # 3. Local refinment to find the exact decimal value for kmax
    for _ in range(refine_steps):
        ks2 = np.linspace(k_lo_refined, k_hi_refined, grid_N)
        last_ok = k_lo_refined
        for k in ks2:
            if client_is_benign_after_refit_distance(X, idx, d, float(k), z):
                last_ok = float(k)
            else:
                k_lo_refined, k_hi_refined = last_ok, float(k)
                break

    return k_lo_refined

class DistanceBasedDefenseStrategyWithKmaxComputation(FedAvg):
    """DistanceBasedDefenseStrategy Statistica su update Delta; stima Kmax, li salva su file, e inietta l'attacco."""

    def __init__(self, *args, initial_global_sd=None, kmax_target_id: int = 3, noise_seed_base: int = 1337, **kwargs):
        super().__init__(*args, **kwargs)
        self._global_sd = None
        self._kmax_target_id = kmax_target_id
        self._noise_seed_base = noise_seed_base
        self._kmax_log_file = "kmax_log.csv"

        if initial_global_sd is not None:
            self._global_sd = {k: v.detach().cpu().clone() for k, v in initial_global_sd.items()}

        with open(self._kmax_log_file, "w") as f:
            f.write("round,kmax\n")

    def aggregate_train(self, server_round: int, replies: Iterable[Message]):
        replies_list = list(replies)

        if not replies_list:
            return super().aggregate_train(server_round, replies_list)  # Fallback on standard FedAvg

        if self._global_sd is None:
            log(logging.WARNING, f"[ROUND {server_round}] Global state missing: fallback on FedAvg without defense.")
            return super().aggregate_train(server_round, replies_list)  # Fallback on standard FedAvg

        log(logging.INFO, f"=== [ROUND {server_round}] DISTANCE-BASED DEFENSE + KMAX COMPUTATION FOR CLIENT {self._kmax_target_id} AS ATTACKER ===")

        # 1. Build X matrix of shape (Num clients, Dim), where each row is the flattened update of a client
        client_ids: list[int] = []
        client_state_dicts = []
        client_updates = []
        for msg in replies_list:
            local_sd = msg.content["arrays"].to_torch_state_dict()
            client_ids.append(int(msg.content["metrics"]["partition-id"]))
            client_state_dicts.append(local_sd)
            client_updates.append(flatten_update(local_sd, self._global_sd))
        X = np.array(client_updates, dtype=np.float64)  # Matrix with clients' flattened updates

        # 2. Determine and inject kmax only on head layer
        target_id = self._kmax_target_id
        if target_id in client_ids:
            idx = client_ids.index(target_id)

            #delta_norm = float(np.linalg.norm(X[idx]) + 1e-12)
            #seed = int(self._noise_seed_base) + int(pid)
            #noise_flat = make_noise_flat_like(client_state_dicts[idx], seed=seed)
            #noise_norm = float(np.linalg.norm(noise_flat) + 1e-12)
            #d = (delta_norm / noise_norm) * noise_flat

            seed = int(self._noise_seed_base) + int(target_id)
            delta_head = flatten_update_head_only(client_state_dicts[idx], self._global_sd)
            delta_norm = float(np.linalg.norm(delta_head) + 1e-12)
            noise_flat = make_noise_flat_like_head_only(client_state_dicts[idx], seed=seed)
            noise_norm = float(np.linalg.norm(noise_flat) + 1e-12)

            d = (delta_norm / noise_norm) * noise_flat

            # 2.1 Determine the kmax value through grid search
            kmax = estimate_kmax_refit_distance(X, idx, d, z=1.0)
            log(logging.INFO, f"[ROUND {server_round} (simulated attacker: client {target_id})] k_max = {kmax:.6f}")

            with open(self._kmax_log_file, "a") as f:
                f.write(f"{server_round},{kmax}\n")

            # 2.2 Injection of the identified k_max value into the attacker's model
            if kmax > 0.0:
                margin = 0.95               # Consider a small margin of error
                k_attack = kmax * margin    # Value to be injected
                
                # Inject the noise on the attacker's updates contained in the update matrix X
                # This is necessary to apply the real defense at the next step
                poisoned_flat_update = X[idx] + (k_attack * d)
                X[idx] = poisoned_flat_update 

                poisoned_sd = {}
                offset = 0
                original_sd = client_state_dicts[idx]
                
                # Inject the noise on the local model of the attacker
                # Necessary to see the final effect of the attack on the global model's performances since Flower works
                # on models and not on updates
                for k in original_sd.keys():
                    if torch.is_floating_point(original_sd[k]):
                        shape = original_sd[k].shape
                        numel = original_sd[k].numel()
                        flat_tensor_update = poisoned_flat_update[offset : offset + numel]
                        offset += numel
                        tensor_update = torch.from_numpy(flat_tensor_update).view(shape).float()
                        poisoned_sd[k] = self._global_sd[k] + tensor_update
                    else:
                        poisoned_sd[k] = original_sd[k]

                replies_list[idx].content["arrays"] = ArrayRecord(poisoned_sd)
                log(logging.INFO, f"[ROUND {server_round}] INJECTED ATTACK with k={k_attack:.6f}")
        else:
            log(logging.INFO, f"[ROUND {server_round}] Target client {target_id} not present.")

        # 3. Apply the real defense: compute centroid and distances on the updates matrix X that now contains the malicious update of the attacker
        centroid = np.mean(X, axis=0)
        distances = np.linalg.norm(X - centroid, axis=1)
        mean_dist = float(np.mean(distances))
        std_dist = float(np.std(distances))
        threshold = mean_dist + (1.0 * std_dist)

        accepted_replies = []
        for i, dist in enumerate(distances):
            pid = client_ids[i]
            if dist <= threshold:
                accepted_replies.append(replies_list[i])
                log(logging.INFO, f" [+] Client {pid} ACCEPTED (Dist: {dist:.4f} <= Threshold: {threshold:.4f})")
            else:
                log(logging.WARNING, f" [!] Client {pid} REJECTED (Dist: {dist:.4f} > Threshold: {threshold:.4f}) -> Suspected attacker!")

        log(logging.INFO, "=============================================")

        aggregated = super().aggregate_train(server_round, accepted_replies)

        # 4. Update _global_sd to compute updates at next round
        agg_msg = aggregated[0] if isinstance(aggregated, tuple) else aggregated
        try:
            if agg_msg is not None and hasattr(agg_msg, "content") and "arrays" in agg_msg.content:
                new_global = agg_msg.content["arrays"].to_torch_state_dict()
                self._global_sd = {k: v.detach().cpu().clone() for k, v in new_global.items()}
        except Exception as e:
            log(logging.WARNING, f"global_sd update not possible: {e}")

        return aggregated

# =========================================================================================================
#                   STRATEGY SIMULATING AN ATTACKER EXPLOITING SIMULATED ANNEALING
# =========================================================================================================
from copy import deepcopy
from torch.utils.data import DataLoader, ConcatDataset
from LSTMfederated.task import load_data
import pandas as pd

def build_centralized_validation_loader(batch_size: int = 64) -> DataLoader:
    """ Builds a validation loader combining tests sets. """
    datasets = []
    for pid in range(5):
        if pid != 1:
            _, testloader = load_data(partition_id=pid, batch_size=batch_size)
            datasets.append(testloader.dataset)
        #_, testloader = load_data(partition_id=pid, batch_size=batch_size)
        #datasets.append(testloader.dataset)
    merged = ConcatDataset(datasets)
    return DataLoader(merged, batch_size=batch_size, shuffle=False, drop_last=False)

def state_dict_to_update_flat(local_sd, global_sd) -> np.ndarray:
    """ Flattens an update. """
    parts = []
    for i, v_local in local_sd.items():
        if torch.is_floating_point(v_local):
            delta = (v_local.detach().cpu() - global_sd[i].detach().cpu()).numpy().ravel()
            parts.append(delta.astype(np.float64))
    return np.concatenate(parts)
    
def local_sd_from_flat_update(flat_update: np.ndarray, reference_local_sd, global_sd) -> dict:
    """ Known the global model, builds a local model from a model update (w_local = w_global + delta). """
    updated_local_model = {}
    offset = 0
    for k, v in reference_local_sd.items():
        if torch.is_floating_point(v):
            num_elems = v.numel()
            update_chunk = flat_update[offset: offset + num_elems]
            offset += num_elems
            delta_tensor = torch.from_numpy(update_chunk.reshape(v.shape)).to(dtype=v.dtype)
            updated_local_model[k] = global_sd[k].detach().cpu() + delta_tensor
        else:
            updated_local_model[k] = v.detach().cpu().clone()
    return updated_local_model

def weighted_average_state_dicts(state_dicts: list[dict], weights: list[float]) -> dict:
    """ Builds an aggregated model simulating FedAvg. """
    total_weight = float(sum(weights))     
    aggregated_model = {}
    for k in state_dicts[0].keys():
        if torch.is_floating_point(state_dicts[0][k]):
            aggregated_model[k] = torch.zeros_like(state_dicts[0][k].detach().cpu())
            for sd, w in zip(state_dicts, weights):
                aggregated_model[k] += sd[k].detach().cpu() * (w / total_weight)
        else:
            aggregated_model[k] = state_dicts[0][k].detach().cpu().clone()
    return aggregated_model

def evaluate_model_mse(state_dict: dict, valloader: DataLoader, device: torch.device) -> float:
    """ Compute the global model's MSE. """
    model = Net()
    model.load_state_dict(state_dict, strict=True)
    mse, _ = test(model, valloader, device)
    return float(mse)

def simulate_distance_defense_acceptance(
    client_updates_flat: np.ndarray,
    malicious_id: int,
) -> tuple[bool, np.ndarray, float, float, float]:
    """ Replicates the distance-based defense. """
    centroid = np.mean(client_updates_flat, axis=0)
    distances = np.linalg.norm(client_updates_flat - centroid, axis=1)

    mean_dist = float(np.mean(distances))
    std_dist = float(np.std(distances))
    threshold = mean_dist + (1.0 * std_dist)

    accepted_mask = [bool(d <= threshold) for d in distances]   # Contiene l'informazione se ogni client passa la difesa o no

    accepted = accepted_mask[malicious_id]  # True se l'attaccante è accettato dalla difesa

    return accepted, accepted_mask, distances, threshold, mean_dist, std_dist

def evaluate_target_sequence(state_dict: dict, x_seq, y_target, device="cpu"):
    """
    Valuta il modello su una singola sequenza target.

    Parametri:
    - state_dict: per modello PyTorch
    - x_seq: numpy array shape (24, 11)
    - y_target: float (valore desiderato PM2.5)
    - device: "cpu" o "cuda"

    Ritorna:
    - dict con prediction, mse, mae
    """

    model = Net()
    model.load_state_dict(state_dict, strict=True)

    model.eval()
    model.to(device)

    # --- controlli ---
    x_seq = np.asarray(x_seq, dtype=np.float32)
    if x_seq.shape != (24, x_seq.shape[1]):
        raise ValueError(f"x_seq must have shape (24, num_features), found {x_seq.shape}")

    # --- tensor ---
    xb = torch.from_numpy(x_seq).unsqueeze(0).to(device)  # (1, 24, 11)
    yb = torch.tensor([y_target], dtype=torch.float32).to(device)  # (1,)

    # --- forward ---
    pred = model(xb)  # (1,)

    # --- metriche ---
    mse = torch.mean((pred - yb) ** 2).item()
    #mae = torch.mean(torch.abs(pred - yb)).item()

    log(logging.INFO, f"prediction={float(pred.item())} target={float(y_target)} mse={float(mse):.6f} ")

    return float(mse), float(pred.item())

def load_sequences(csv_seq: str, csv_tgt: str) -> tuple[list[np.ndarray], list[float]]:
    df_seq = pd.read_csv(csv_seq)
    targets = pd.read_csv(csv_tgt)["target"].tolist()
    
    values = df_seq.values.astype(np.float32)
    n_seq = len(targets)  # ogni sequenza è esattamente 24 righe
    
    seqs = [values[i*24:(i+1)*24] for i in range(n_seq)]
    return seqs, targets


class DistanceBasedDefenseStrategyWithSA(FedAvg):
    def __init__(
        self,
        *args,
        initial_global_sd=None,
        malicious_id: int = 3,
        sa_T0: float = 1.0,
        sa_Tmin: float = 1e-3,
        sa_alpha: float = 0.95,
        sa_L: int = 30,
        sa_step_radius: float = 0.05,
        sa_reject_penalty: float = 1e6,
        sa_val_batch_size: int = 64,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._global_sd = None  # Current global model: used to comput updates from local models
        if initial_global_sd is not None:
            self._global_sd = {k: v.detach().cpu().clone() for k, v in initial_global_sd.items()}

        self.malicious_id = int(malicious_id)   # id del client malevolo
        self.metrics = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}

        self.sa_T0 = sa_T0                              # Initial temperature 
        self.sa_Tmin = sa_Tmin                          # Minimum temperature
        self.sa_alpha = sa_alpha                        # Cooling ratio
        self.sa_L = sa_L                                # Number of candidate solution per temperature stage
        self.sa_step_radius = sa_step_radius            # Step radius
        self.sa_reject_penalty = sa_reject_penalty      # Penalty to apply when candidate is rejected from defense

        self.device = torch.device("cpu")
        self.val_loader = build_centralized_validation_loader(batch_size=sa_val_batch_size)

    def _extract_round_data(self, replies_list: list[Message]):
        client_ids = []     # ID dei client
        local_sds = []      # Modelli locali dei client
        updates_flat = []   # Updates flatten
        num_examples = []   # Pesi per FedAvg, inviati dai client nelle metriche

        for msg in replies_list:
            local_sd = msg.content["arrays"].to_torch_state_dict()
            pid = int(msg.content["metrics"]["partition-id"])

            n_i = float(msg.content["metrics"].get("num-examples", 1.0))

            client_ids.append(pid)
            local_sds.append(local_sd)
            updates_flat.append(state_dict_to_update_flat(local_sd, self._global_sd))
            num_examples.append(n_i)

        return client_ids, local_sds, np.array(updates_flat), num_examples

    def _energy(
        self,
        candidate_delta: np.ndarray,    # Candidate solution
        base_legit_update: np.ndarray,  # Legitimate update of malicious client
        all_updates_flat: np.ndarray,   # All clients' flattened updates
        malicious_id: int,              # Index of malicious clientt
        local_sds: list[dict],          # All clients' local models
        num_examples: list[float],      # Training weights for simulated FedAvg
        base_mse: float,                # Baseine MSE      
    ) -> tuple[float, bool, float]:
        """
            Energy:
                - If candidate refused by the defense -> high penalty
                - If candidate accepted by the defense -> -rel_mse
        """
        # Build the malicious update and insert it into the dictionary Xcand of all client's update
        candidate_update = base_legit_update + candidate_delta
        Xcand = all_updates_flat.copy()
        Xcand[malicious_id] = candidate_update

        # Verify candidate acceptance by the defense: if rejected --> penalty
        accepted, accepted_mask, _, _, _, _ = simulate_distance_defense_acceptance(Xcand, malicious_id)
        if not accepted:
            log(logging.INFO, f"Candidate delta discarded from defense")
            return self.sa_reject_penalty, False, float("nan")

        # Reconstruct the poisoned local model of malicious client
        poisoned_local_sd = local_sd_from_flat_update(
            candidate_update,
            reference_local_sd=local_sds[malicious_id],
            global_sd=self._global_sd,
        )

        # Insert the poisoned local model into the dictionary local_sds_eval of all client's local models
        local_sds_eval = [deepcopy(sd) for sd in local_sds]
        local_sds_eval[malicious_id] = poisoned_local_sd

        # Simulate FedAvg on local models accepted by the defense
        accepted_sds = [sd for sd, ok in zip(local_sds_eval, accepted_mask) if ok]
        accepted_w = [w for w, ok in zip(num_examples, accepted_mask) if ok]
        if len(accepted_sds) == 0:
            return self.sa_reject_penalty, False, float("nan")
        agg_sd = weighted_average_state_dicts(accepted_sds, accepted_w)

        # Compute the MSE on the global model
        mse = evaluate_model_mse(agg_sd, self.val_loader, self.device)
        rel_mse = mse / (base_mse + 1e-12)
        
        # Compute the energy of the candidate solution
        energy = -rel_mse

        log(logging.INFO, f"rel_mse={rel_mse:.6f} mse={mse:.6f} energy={energy:.6f} ")

        return float(energy), True, float(mse)
    
#    def _energy_backdoor_attack(
#        self,
#        candidate_delta: np.ndarray,    # la soluzione candidata del SA
#        base_legit_update: np.ndarray,  # update legittimo del client malevolo
#        all_updates_flat: np.ndarray,   # tutti gli update del round
#        malicious_id: int,             # posizione del malevolo negli update
#        local_sds: list[dict],          # modelli locali
#        num_examples: list[float],      # pesi FedAvg
#        base_mse: float,                # MSE del round senza perturbazione
#        x_seq: np.ndarray,              # sequenza che vogliamo avvelenare
#        y_target: float                  # valore desiderato PM2.5 (25° timestep)
#    ) -> tuple[float, bool, float]:
#
#        # Costruisco l'update malevolo candidato e lo inserisco in Xcand al posto dell'update legittimo del client malevolo
#        candidate_update = base_legit_update + candidate_delta
#        Xcand = all_updates_flat.copy()
#        Xcand[malicious_id] = candidate_update
#
#        # Verifica se il candidato viene accettato o scartato dalla difesa: se viene scartato l'energia è enorme
#        #accepted, _, _, _, _, _ = simulate_distance_defense_acceptance(Xcand, malicious_id)
#        #if not accepted:
#        #    log(logging.INFO, f"Candidate delta discarded from defense")
#        #    return self.sa_reject_penalty, False, float("nan")
#
#        # Ricostruzione del modello locale (state_dict locali) con il candidato malevolo
#        poisoned_local_sd = local_sd_from_flat_update(
#            candidate_update,
#            reference_local_sd=local_sds[malicious_id],
#            global_sd=self._global_sd,
#        )
#
#        local_sds_eval = [deepcopy(sd) for sd in local_sds]
#        local_sds_eval[malicious_id] = poisoned_local_sd
#
#        # Ricostruiamo la difesa per capire quali client sarebbero accettati nel round simulato
#        accepted_mask = []
#        centroid = np.mean(Xcand, axis=0)
#        distances = np.linalg.norm(Xcand - centroid, axis=1)
#        threshold = float(np.mean(distances) + np.std(distances))
#        for d in distances:
#            accepted_mask.append(d <= threshold)
#
#        # Facciamo FedAvg simulato: calcoliamo il modello globale simulato e poi l'MSE
#        #accepted_sds = [sd for sd, ok in zip(local_sds_eval, accepted_mask) if ok]
#        #accepted_w = [w for w, ok in zip(num_examples, accepted_mask) if ok]
#        accepted_sds = local_sds_eval
#        accepted_w = num_examples
#
#        if len(accepted_sds) == 0:
#            return self.sa_reject_penalty, False, float("nan")
#
#        agg_sd = weighted_average_state_dicts(accepted_sds, accepted_w)
#        mse_on_original_dataset = evaluate_model_mse(agg_sd, self.val_loader, self.device)
#
#        delta_mse = mse_on_original_dataset - base_mse
#
#        mse_target, prediction = evaluate_target_sequence(agg_sd, x_seq, y_target, "cpu")
#
#        # Energia:
#        #   minimizzare l'errore su sequenza falsa
#        #   + delta_mse perchè vogliamo mantenere l'errore sul testset originale il più invariato possibile ()
#        #energy = mse_target + delta_mse
#        energy = mse_target
#        log(logging.INFO,
#            f"delta_mse={delta_mse:.6f} mse_target={mse_target} "
#            f"energy={energy:.6f} "
#        )
#
#        return float(energy), True, prediction

    def _energy_targeted_attack(
        self,
        candidate_delta: np.ndarray,
        base_legit_update: np.ndarray,
        all_updates_flat: np.ndarray,
        malicious_id: int,
        local_sds: list[dict],
        num_examples: list[float],
        base_mse_poison: float,
        base_mse_clean: float,
        x_poison: list[np.ndarray],     # Sequences to maximize error
        y_poison: list[float],          # Targets of sequences to poison
        x_clean: list[np.ndarray],      # Sequences to preserve or minimize error
        y_clean: list[float],           # Targets of sequences to maintain clean
        lambda_clean: float = 1.0,      
        lambda_poison: float = 1.0
    ) -> tuple[float, bool, float]:
        """
            Energy:
                - If candidate refused by the defense -> high penalty
                - If candidate accepted by the defense -> -lambda_poison * mse_poison + lambda_clean * mse_clean
        """

        # Costruisco l'update malevolo candidato e lo inserisco in Xcand al posto dell'update legittimo del client malevolo
        candidate_update = base_legit_update + candidate_delta
        Xcand = all_updates_flat.copy()
        Xcand[malicious_id] = candidate_update

        # Verifica se il candidato viene accettato o scartato dalla difesa: se viene scartato l'energia è enorme
        accepted, accepted_mask, _, _, _, _ = simulate_distance_defense_acceptance(Xcand, malicious_id)
        if not accepted:
            log(logging.INFO, f"Candidate delta discarded from defense")
            return self.sa_reject_penalty, False, float("nan"), float("nan"), float("nan")

        # Ricostruzione del modello locale (state_dict locali) del candidato malevolo a partire dall'update
        poisoned_local_sd = local_sd_from_flat_update(
            candidate_update,
            reference_local_sd=local_sds[malicious_id],
            global_sd=self._global_sd,
        )

        # Modelli dei client con quello avvelenato per il client malevolo 
        local_sds_eval = [deepcopy(sd) for sd in local_sds]
        local_sds_eval[malicious_id] = poisoned_local_sd

        # Ricostruiamo la difesa per capire quali client sarebbero accettati nel round simulato
        accepted_mask = []
        centroid = np.mean(Xcand, axis=0)
        distances = np.linalg.norm(Xcand - centroid, axis=1)
        threshold = float(np.mean(distances) + np.std(distances))
        for d in distances:
            accepted_mask.append(d <= threshold)

        # Facciamo FedAvg simulato: calcoliamo il modello globale simulato e poi l'MSE
        accepted_sds = [sd for sd, ok in zip(local_sds_eval, accepted_mask) if ok]
        accepted_w = [w for w, ok in zip(num_examples, accepted_mask) if ok]

        if len(accepted_sds) == 0:
            return self.sa_reject_penalty, False, float("nan"), float("nan"), float("nan")

        agg_sd = weighted_average_state_dicts(accepted_sds, accepted_w)

        #agg_sd = weighted_average_state_dicts(local_sds_eval, num_examples)
        
        # MSE sulle sequenze poison
        mse_poison_list = []
        for x_seq, y_t in zip(x_poison, y_poison):
            mse_i, _ = evaluate_target_sequence(agg_sd, x_seq, y_t, "cpu")
            mse_poison_list.append(mse_i)
        mse_poison = float(np.mean(mse_poison_list))

        # MSE sulle sequenze clean
        mse_clean_list = []
        for x_seq, y_t in zip(x_clean, y_clean):
            mse_i, _ = evaluate_target_sequence(agg_sd, x_seq, y_t, "cpu")
            mse_clean_list.append(mse_i)
        mse_clean = float(np.mean(mse_clean_list))

        # MSE globale 
        #mse = evaluate_model_mse(agg_sd, self.val_loader, self.device)
        mse = float("nan")

        rel_mse_poison = mse_poison / (base_mse_poison + 1e-12)
        rel_mse_clean = mse_clean / (base_mse_clean + 1e-12)
        # Energia:
        #   Massimizzare l'MSE sequenze poison: -lambda_poison * rel_mse_poison
        #   Minimizzare l'MSE sequenze clean: +lambda_clean * rel_mse_clean
        energy = -lambda_poison*rel_mse_poison + lambda_clean*rel_mse_clean
        log(logging.INFO,
            f"mse_poison={mse_poison:.6f} mse_clean={mse_clean:.6f} lambda_poison={lambda_poison} lambda_clean={lambda_clean} energy={energy:.6f} "
            f"total_mse={mse:.6f} "
        )

        return float(energy), True, mse_poison, mse_clean, mse

    def _run_simulated_annealing(
        self,
        server_round: int,
        base_legit_update: np.ndarray,
        all_updates_flat: np.ndarray,
        malicious_id: int,
        local_sds: list[dict],
        num_examples: list[float],
    ) -> np.ndarray:

        # Compute the base MSE of the round
        Xbase = all_updates_flat.copy()
        _, accepted_mask, _, _, _, _ = simulate_distance_defense_acceptance(Xbase, malicious_id)
        base_sds = [sd for sd, ok in zip(local_sds, accepted_mask) if ok]
        base_w = [w for w, ok in zip(num_examples, accepted_mask) if ok]
        base_agg_sd = weighted_average_state_dicts(base_sds, base_w)
        true_base_mse = evaluate_model_mse(base_agg_sd, self.val_loader, self.device)

        # Initial solution: delta = 0
        d = base_legit_update.shape[0]
        delta = np.zeros(d, dtype=np.float64)

        # Initialization
        curr_energy, _, curr_mse = self._energy(
            candidate_delta=delta,
            base_legit_update=base_legit_update,
            all_updates_flat=all_updates_flat,
            malicious_id=malicious_id,
            local_sds=local_sds,
            num_examples=num_examples,
            base_mse=true_base_mse,
        )
        best_delta = delta.copy()  
        best_energy = curr_energy
        best_mse = curr_mse

        # Creation of head_mask
        head_mask = np.zeros(d, dtype=np.float64)
        offset = 0
        reference_sd = local_sds[malicious_id]
        for k, v in reference_sd.items():
            if torch.is_floating_point(v):
                numel = v.numel()
                if "head" in k: 
                    head_mask[offset : offset + numel] = 1.0
                offset += numel

        T = self.sa_T0
        stage = 0
        patience = 20
        energy_tol = 1e-7
        no_improve_stages = 0
        prev_best_energy = best_energy
        freeze_stages = 0

        # Loop on temperature stages
        while T > self.sa_Tmin:
            accepted_moves = 0
            rho_T = self.sa_step_radius * (T / self.sa_T0)

            log(logging.INFO, f"[ROUND {server_round}] SA stage={stage} T={T:.6f}")

            for _ in range(self.sa_L):
                # Generate a random perturbation on all dimensions
                z = np.random.randn(d).astype(np.float64)

                # Apply the mask
                z = z * head_mask   # Resets all noise except that at the 'head'

                z_norm = np.linalg.norm(z) + 1e-12

                # Compute candidate solution
                cand_delta = delta + rho_T * (z / z_norm)

                # Compute the energy of the candidate solution
                cand_energy, cand_ok, cand_mse = self._energy(
                    candidate_delta=cand_delta,
                    base_legit_update=base_legit_update,
                    all_updates_flat=all_updates_flat,
                    malicious_id=malicious_id,
                    local_sds=local_sds,
                    num_examples=num_examples,
                    base_mse=true_base_mse,
                )

                if not cand_ok:
                    log(logging.INFO, "Solution not accepted")
                    continue

                dE = cand_energy - curr_energy

                if dE <= 0:
                    delta = cand_delta
                    curr_energy = cand_energy
                    curr_mse = cand_mse
                    accepted_moves += 1
                    log(logging.INFO, f"Solution accepted")
                else:
                    p = np.exp(-dE / max(T, 1e-12))
                    if np.random.rand() < p:
                        delta = cand_delta
                        curr_energy = cand_energy
                        curr_mse = cand_mse
                        accepted_moves += 1
                        log(logging.INFO, f"Solution accepted with P(dE)={p}")
                    else:
                        log(logging.INFO, f"Solution not accepted")

                if curr_energy < best_energy:
                    best_energy = curr_energy
                    best_delta = delta.copy()
                    best_mse = curr_mse

            log(logging.INFO,
                f"--> Tried {self.sa_L} candidate deltas: "
                f"accepted_moves={accepted_moves}/{self.sa_L} "
                f"best_energy={best_energy:.6f} best_mse={best_mse:.6f}, curr_energy={curr_energy:.6f}")

            # First early stopping criteria: Thermal Freeze
            acceptance_rate = accepted_moves/self.sa_L
            if acceptance_rate < 0.01:
                freeze_stages += 1
            else:
                freeze_stages = 0
            if freeze_stages >= 3:
                break

            # Second early stopping criteria: Patience Exhaustion
            improvement = prev_best_energy - best_energy
            if np.isnan(best_energy) and np.isnan(prev_best_energy):
                no_improve_stages += 1
            elif improvement <= energy_tol:
                no_improve_stages += 1
            else:
                no_improve_stages = 0
                prev_best_energy = best_energy
            if no_improve_stages >= patience:
                break
            
            # Cooling
            T *= self.sa_alpha

            stage += 1

        return best_delta

#    def _run_simulated_annealing_backdoor(
#        self,
#        server_round: int,
#        base_legit_update: np.ndarray,
#        all_updates_flat: np.ndarray,
#        malicious_id: int,
#        local_sds: list[dict],
#        num_examples: list[float],
#    ) -> np.ndarray:
#        """
#        Simulated Annealing vero e proprio: cerca la perturbazione migliore da aggiungere all'update del client malevolo nel round corrente.
#        Soluzione iniziale: delta = 0
#        Mossa: delta' = delta + rho_T * z / ||z||
#        """
#
#        # Calcolo l'MSE del round senza perturbazione
#        Xbase = all_updates_flat.copy()
#        Xbase[malicious_id] = base_legit_update
#        centroid = np.mean(Xbase, axis=0)
#        distances = np.linalg.norm(Xbase - centroid, axis=1)
#        threshold = float(np.mean(distances) + np.std(distances))
#        accepted_mask = [d <= threshold for d in distances]
#
#        base_sds = [sd for sd, ok in zip(local_sds, accepted_mask) if ok]
#        base_w = [w for w, ok in zip(num_examples, accepted_mask) if ok]
#        base_agg_sd = weighted_average_state_dicts(base_sds, base_w)
#        true_base_mse = evaluate_model_mse(base_agg_sd, self.val_loader, self.device)
#
#        df = pd.read_csv("sequence_to_poison.csv")
#        x_seq = df.values.astype(np.float32)
#        if x_seq.shape[0] != 24:
#            raise ValueError(f"CSV file must contain 24 rows, found {x_seq.shape[0]}")
#        y_target = 0
#
#        # Soluzione iniziale delta = 0, cioè si parte dall'update legittimo.
#        # Calcoliamo quindi l'energia iniziale, che sarà l'energia corrente, e il valore di mse iniziale, che sarà quello corrente.
#        # Ci manteniamo soluzione corrente (curr_delta) e migliore soluzione trovata finora (best_delta): questo perche in SA la soluzione corrente non è necessariamente
#        # la migliore soluzione visitata. Infatti, SA può accettare soluzioni peggiori
#        d = base_legit_update.shape[0]
#        delta = np.zeros(d, dtype=np.float64)
#
#        # --- 1. CREAZIONE DELLA MASCHERA 'HEAD' ---
#        head_mask = np.zeros(d, dtype=np.float64)
#        offset = 0
#        reference_sd = local_sds[malicious_id]
#        
#        for k, v in reference_sd.items():
#            if torch.is_floating_point(v):
#                numel = v.numel()
#                if "head" in k: 
#                    head_mask[offset : offset + numel] = 1.0
#                offset += numel
#        # --------------------------------------------------------
#
#        curr_energy, _, curr_prediction = self._energy_backdoor_attack(
#            candidate_delta=delta,
#            base_legit_update=base_legit_update,
#            all_updates_flat=all_updates_flat,
#            malicious_id=malicious_id,
#            local_sds=local_sds,
#            num_examples=num_examples,
#            base_mse=true_base_mse,
#            x_seq=x_seq,
#            y_target=y_target
#        )
#        best_delta = delta.copy()
#        best_energy = curr_energy
#        best_prediction = curr_prediction
#
#        # Ciclo sulle temperature
#        # Per ogni stadio della temperatura:
#        #   1) Generiamo una configurazione candidato ammissibile tramite una piccola perturbazione casuale della configurazione corrente.
#        #      Valutiamo la differenza di energia dE tra la soluzione candidato e la soluzione corrente.
#        #   2) Se dE <= 0 --> cand_energy <= curr_energy e accettiamo la soluzione candidato come soluzione corrente
#        #      Se dE > 0 --> cand_energy > curr_energy e accettiamo la soluzione candidato come soluzione corrente con probabilità P(dE)
#        #   3) Aggioriamo valore di T e se non abbiamo raggiunto l'equilibrio termico T_min torniamo allo step 1. Altrimenti, 
#        T = self.sa_T0
#        stage = 0
#        patience = 20
#        energy_tol = 1e-7
#        no_improve_stages = 0
#        prev_best_energy = best_energy
#        freeze_stages = 0
#
#        while T > self.sa_Tmin:
#            accepted_moves = 0
#            rho_T = self.sa_step_radius * (T / self.sa_T0)
#            #rho_T = max(rho_min, self.sa_step_radius * np.sqrt(T / self.sa_T0))
#
#            log(logging.INFO, f"[ROUND {server_round}] SA stage={stage} T={T:.6f}")
#
#            for _ in range(self.sa_L):
#                # Mossa candidata: genera una perturbazione casuale su tutte le dimensioni. Useremo la direzione di z sulla direzione unitaria
#                z = np.random.randn(d).astype(np.float64)
#                # L'ampiezza della perturbazione ce la da rho_T
#
#                # --- 2. APPLICAZIONE MASCHERA ---
#                # Azzera tutto il rumore tranne quello in corrispondenza della 'head'
#                z = z * head_mask
#                # ----------------------------------------------
#
#                z_norm = np.linalg.norm(z) + 1e-12
#
#                cand_delta = delta + rho_T * (z / z_norm)
#
#                # Energia e mse della soluzione candidata
#                cand_energy, cand_ok, cand_prediction = self._energy_backdoor_attack(
#                    candidate_delta=cand_delta,
#                    base_legit_update=base_legit_update,
#                    all_updates_flat=all_updates_flat,
#                    malicious_id=malicious_id,
#                    local_sds=local_sds,
#                    num_examples=num_examples,
#                    base_mse=true_base_mse,
#                    x_seq=x_seq,
#                    y_target=y_target
#                )
#
#                if not cand_ok:
#                    log(logging.INFO, "Solution not accepted")
#                    continue
#
#                # Valutazione e accettazione: 
#                #   - Se la candidata migliora l'energia --> la accettiamo sempre
#                #   - Se peggiora --> la accettiamo con probabilità exp(-dE/T)
#                dE = cand_energy - curr_energy
#                #print("dE:", cand_energy - curr_energy)
#                #print("||cand_delta||:", np.linalg.norm(cand_delta))
#                if dE <= 0:
#                    delta = cand_delta
#                    curr_energy = cand_energy
#                    curr_prediction = cand_prediction
#                    accepted_moves += 1
#                    log(logging.INFO, f"Solution accepted")
#                else:
#                    p = np.exp(-dE / max(T, 1e-12))
#                    if np.random.rand() < p:
#                        delta = cand_delta
#                        curr_energy = cand_energy
#                        curr_prediction = cand_prediction
#                        accepted_moves += 1
#                        log(logging.INFO, f"Solution accepted with P(dE)={p}")
#                    else:
#                        log(logging.INFO, f"Solution not accepted")
#
#                if curr_energy < best_energy:
#                    best_energy = curr_energy
#                    best_delta = delta.copy()
#                    best_prediction = curr_prediction
#
#            log(logging.INFO,
#                f"--> Tried {self.sa_L} candidate deltas: "
#                f"accepted_moves={accepted_moves}/{self.sa_L} "
#                f"best_energy={best_energy:.6f} best_prediction={best_prediction:.6f}")
#
#            # Stop anticipato se praticamente congelato
#            acceptance_rate = accepted_moves/self.sa_L
#            if acceptance_rate < 0.01:
#                freeze_stages += 1
#            else:
#                freeze_stages = 0
#            if freeze_stages >= 3:
#                break
#
#            # Stop anticipato se per patience stadi consecutivi best_energy non è migliorato
#            improvement = prev_best_energy - best_energy
#            if np.isnan(best_energy) and np.isnan(prev_best_energy):
#                no_improve_stages += 1
#            elif improvement <= energy_tol:
#                no_improve_stages += 1
#            else:
#                no_improve_stages = 0
#                prev_best_energy = best_energy
#            if no_improve_stages >= patience:
#                break
#            
#            # Raffreddamento: T_new = alpha*T
#            T *= self.sa_alpha
#            stage += 1
#
#        return best_delta

    def _run_simulated_annealing_targeted(
        self,
        server_round: int,
        base_legit_update: np.ndarray,
        all_updates_flat: np.ndarray,
        malicious_id: int,
        local_sds: list[dict],
        num_examples: list[float],
    ) -> np.ndarray:

        # Setup sequenze
        x_poison, y_poison = load_sequences("./SequencesBackdoorSA/high_pollution_sequences.csv", "./SequencesBackdoorSA/high_pollution_targets.csv")
        x_clean,  y_clean  = load_sequences("./SequencesBackdoorSA/low_pollution_sequences.csv",  "./SequencesBackdoorSA/low_pollution_targets.csv")
        #x_poison, y_poison = load_sequences("./SequencesBackdoorSA/low_pollution_sequences.csv",  "./SequencesBackdoorSA/low_pollution_targets.csv")
        #x_clean,  y_clean  = load_sequences("./SequencesBackdoorSA/high_pollution_sequences.csv", "./SequencesBackdoorSA/high_pollution_targets.csv")
        lambda_clean = 1
        lambda_poison = 1

        # Simulo la difesa per sapere chi viene accettato
        Xbase = all_updates_flat.copy()
        #Xbase[malicious_id] = base_legit_update
        #centroid = np.mean(Xbase, axis=0)
        #distances = np.linalg.norm(Xbase - centroid, axis=1)
        #threshold = float(np.mean(distances) + np.std(distances))
        #accepted_mask = [d <= threshold for d in distances]
        _, accepted_mask, _, _, _, _ = simulate_distance_defense_acceptance(Xbase, malicious_id)

        base_sds = [sd for sd, ok in zip(local_sds, accepted_mask) if ok]
        base_w = [w for w, ok in zip(num_examples, accepted_mask) if ok]
        base_agg_sd = weighted_average_state_dicts(base_sds, base_w)

        # Calcolo l'MSE del round sulle sequenze poison senza perturbazione
        mse_poison_list = []
        for x_seq, y_t in zip(x_poison, y_poison):
            mse_i, _ = evaluate_target_sequence(base_agg_sd, x_seq, y_t, "cpu")
            mse_poison_list.append(mse_i)
        base_mse_poison = np.mean(mse_poison_list)

        # Calcolo l'MSE del round sulle sequenze clean senza perturbazione
        mse_clean_list = []
        for x_seq, y_t in zip(x_clean, y_clean):
            mse_i, _ = evaluate_target_sequence(base_agg_sd, x_seq, y_t, "cpu")
            mse_clean_list.append(mse_i)
        base_mse_clean = np.mean(mse_clean_list)

        # Soluzione iniziale delta = 0, cioè si parte dall'update legittimo.
        # Calcoliamo quindi l'energia iniziale, che sarà l'energia corrente, e il valore di mse iniziale, che sarà quello corrente.
        # Ci manteniamo soluzione corrente (curr_delta) e migliore soluzione trovata finora (best_delta): questo perche in SA la soluzione corrente non è necessariamente
        # la migliore soluzione visitata. Infatti, SA può accettare soluzioni peggiori
        d = base_legit_update.shape[0]
        delta = np.zeros(d, dtype=np.float64)

        curr_energy, _, curr_mse_poison, curr_mse_clean, curr_mse = self._energy_targeted_attack(
            candidate_delta=delta,
            base_legit_update=base_legit_update,
            all_updates_flat=all_updates_flat,
            malicious_id=malicious_id,
            local_sds=local_sds,
            num_examples=num_examples,
            base_mse_poison=base_mse_poison,
            base_mse_clean=base_mse_clean,
            x_poison=x_poison,   # high pollution → massimizza MSE
            y_poison=y_poison,
            x_clean=x_clean,     # low pollution  → minimizza MSE
            y_clean=y_clean,
            lambda_clean=lambda_clean,
            lambda_poison=lambda_poison
        )
        best_delta = delta.copy()
        best_energy = curr_energy
        best_mse_poison = curr_mse_poison
        best_mse_clean = curr_mse_clean
        best_mse = curr_mse

        init_mse_poison = curr_mse_poison
        init_mse_clean = curr_mse_clean
        init_global_mse = curr_mse

        # --- 1. CREAZIONE DELLA MASCHERA 'HEAD' ---
        head_mask = np.zeros(d, dtype=np.float64)
        offset = 0
        reference_sd = local_sds[malicious_id]
        
        for k, v in reference_sd.items():
            if torch.is_floating_point(v):
                numel = v.numel()
                if "head" in k: 
                    head_mask[offset : offset + numel] = 1.0
                offset += numel
        # --------------------------------------------------------

        # Ciclo sulle temperature
        # Per ogni stadio della temperatura:
        #   1) Generiamo una configurazione candidato ammissibile tramite una piccola perturbazione casuale della configurazione corrente.
        #      Valutiamo la differenza di energia dE tra la soluzione candidato e la soluzione corrente.
        #   2) Se dE <= 0 --> cand_energy <= curr_energy e accettiamo la soluzione candidato come soluzione corrente
        #      Se dE > 0 --> cand_energy > curr_energy e accettiamo la soluzione candidato come soluzione corrente con probabilità P(dE)
        #   3) Aggioriamo valore di T e se non abbiamo raggiunto l'equilibrio termico T_min torniamo allo step 1. Altrimenti, 
        T = self.sa_T0
        stage = 0
        patience = 5
        energy_tol = 1e-7
        no_improve_stages = 0
        prev_best_energy = best_energy
        freeze_stages = 0

        while T > self.sa_Tmin:
            accepted_moves = 0
            rho_T = self.sa_step_radius * (T / self.sa_T0)

            log(logging.INFO, f"[ROUND {server_round}] SA stage={stage} T={T:.6f}")

            for _ in range(self.sa_L):
                # Mossa candidata: genera una perturbazione casuale su tutte le dimensioni. Useremo la direzione di z sulla direzione unitaria
                z = np.random.randn(d).astype(np.float64)
                # L'ampiezza della perturbazione ce la da rho_T

                # --- 2. APPLICAZIONE MASCHERA ---
                # Azzera tutto il rumore tranne quello in corrispondenza della 'head'
                z = z * head_mask
                # ----------------------------------------------

                z_norm = np.linalg.norm(z) + 1e-12

                cand_delta = delta + rho_T * (z / z_norm)

                # Energia e mse della soluzione candidata
                cand_energy, cand_ok, cand_mse_poison, cand_mse_clean, cand_mse = self._energy_targeted_attack(
                    candidate_delta=cand_delta,
                    base_legit_update=base_legit_update,
                    all_updates_flat=all_updates_flat,
                    malicious_id=malicious_id,
                    local_sds=local_sds,
                    num_examples=num_examples,
                    base_mse_poison=base_mse_poison,
                    base_mse_clean=base_mse_clean,
                    x_poison=x_poison,   # high pollution → massimizza MSE
                    y_poison=y_poison,
                    x_clean=x_clean,     # low pollution  → minimizza MSE
                    y_clean=y_clean,
                    lambda_clean=lambda_clean,
                    lambda_poison=lambda_poison
                )

                if not cand_ok:
                    log(logging.INFO, "Solution not accepted")
                    continue

                # Valutazione e accettazione: 
                #   - Se la candidata migliora l'energia --> la accettiamo sempre
                #   - Se peggiora --> la accettiamo con probabilità exp(-dE/T)
                dE = cand_energy - curr_energy
                #print("dE:", cand_energy - curr_energy)
                #print("||cand_delta||:", np.linalg.norm(cand_delta))
                if dE <= 0:
                    delta = cand_delta
                    curr_energy = cand_energy
                    curr_mse_poison = cand_mse_poison
                    curr_mse_clean = cand_mse_clean
                    curr_mse = cand_mse
                    accepted_moves += 1
                    log(logging.INFO, f"Solution accepted")
                else:
                    p = np.exp(-dE / max(T, 1e-12))
                    if np.random.rand() < p:
                        delta = cand_delta
                        curr_energy = cand_energy
                        curr_mse_poison = cand_mse_poison
                        curr_mse_clean = cand_mse_clean
                        curr_mse = cand_mse
                        accepted_moves += 1
                        log(logging.INFO, f"Solution accepted with P(dE)={p}")
                    else:
                        log(logging.INFO, f"Solution not accepted")

                if curr_energy < best_energy:
                    best_energy = curr_energy
                    best_delta = delta.copy()
                    best_mse_poison = curr_mse_poison
                    best_mse_clean = curr_mse_clean
                    best_mse = curr_mse

            log(logging.INFO,
                f"--> Tried {self.sa_L} candidate deltas: "
                f"accepted_moves={accepted_moves}/{self.sa_L} "
                f"best_energy={best_energy:.6f} "
                f"best_mse_poison={best_mse_poison:.6f} best_mse_clean={best_mse_clean:.6f} "
                f"best_global_mse={best_mse:.6f}"
            )

            # Stop anticipato se praticamente congelato
            acceptance_rate = accepted_moves/self.sa_L
            if acceptance_rate < 0.01:
                freeze_stages += 1
            else:
                freeze_stages = 0
            if freeze_stages >= 3:
                break

            # Stop anticipato se per patience stadi consecutivi best_energy non è migliorato
            improvement = prev_best_energy - best_energy
            if np.isnan(best_energy) and np.isnan(prev_best_energy):
                no_improve_stages += 1
            elif improvement <= energy_tol:
                no_improve_stages += 1
            else:
                no_improve_stages = 0
                prev_best_energy = best_energy
            if no_improve_stages >= patience:
                break
            
            # Raffreddamento: T_new = alpha*T
            T *= self.sa_alpha
            stage += 1

        if server_round == 1:
            with open("sa_selective_backdoor_with_defense_results.txt", "a") as f:
                f.write("\n=== SA Selective Backdoor Parameters ===\n")
                f.write(f"sa_T0={self.sa_T0}\n")
                f.write(f"sa_Tmin={self.sa_Tmin}\n")
                f.write(f"sa_alpha={self.sa_alpha}\n")
                f.write(f"sa_L={self.sa_L}\n")
                f.write(f"sa_step_radius={self.sa_step_radius}\n")
                f.write(f"lambda_clean={lambda_clean}\n")
                f.write(f"lambda_poison={lambda_poison}\n")
                f.write(f"patience={patience}\n")
                f.write("========================================\n")

        with open("sa_selective_backdoor_with_defense_results.txt", "a") as f:
            f.write(f"Round {server_round}: init_mse_poison={init_mse_poison:.6f}, init_mse_clean={init_mse_clean:.6f}, init_global_mse={init_global_mse:.6f} -> "
                    f"best_mse_poison={best_mse_poison:.6f}, best_mse_clean={best_mse_clean:.6f}, global_mse={best_mse:.6f}\n")

        return best_delta

    def aggregate_train(self, server_round: int, replies: Iterable[Message]):
        replies_list = list(replies)

        if not replies_list:
            return super().aggregate_train(server_round, replies_list)

        if self._global_sd is None:
            log(logging.WARNING, f"[ROUND {server_round}] Global state missing: fallback on FedAvg.")
            return super().aggregate_train(server_round, replies_list)

        log(logging.INFO, f"=== [ROUND {server_round}] DISTANCE-BASED DEFENSE + SA ATTACK ===")

        # Estrazione round data: informazioni utili per eseguire il SA
        client_ids, local_sds, client_updates_flat, num_examples = self._extract_round_data(replies_list)

        # --- SA attack sul client malevolo ---
        if self.malicious_id in client_ids:
            mal_idx = client_ids.index(self.malicious_id)
            base_legit_update = client_updates_flat[mal_idx].copy()

            best_delta = self._run_simulated_annealing_targeted(
                server_round=server_round,
                base_legit_update=base_legit_update,
                all_updates_flat=client_updates_flat,
                malicious_id=mal_idx,
                local_sds=local_sds,
                num_examples=num_examples,
            )

            # Sostituzione dell'update malevolo
            poisoned_update = base_legit_update + best_delta

            # Ricostruiamo il modello locale malevolo e lo reinseriamo in replies_list
            poisoned_local_sd = local_sd_from_flat_update(
                poisoned_update,
                reference_local_sd=local_sds[mal_idx],
                global_sd=self._global_sd,
            )
            replies_list[mal_idx].content["arrays"] = ArrayRecord(poisoned_local_sd)
            client_updates_flat[mal_idx] = poisoned_update

            log(logging.WARNING,
                f"[ROUND {server_round}] Injected SA attack on client {self.malicious_id} "
                f"||delta||={np.linalg.norm(best_delta):.6f}")

        # --- Difesa reale ---
        # Dopo che il SA ha finito, facciamo girare la difesa reale
        centroid = np.mean(client_updates_flat, axis=0)
        distances = np.linalg.norm(client_updates_flat - centroid, axis=1)

        mean_dist = float(np.mean(distances))
        std_dist = float(np.std(distances))
        threshold = mean_dist + (1.0 * std_dist)

        accepted_replies = []
        for i, dist in enumerate(distances):
            client_id = client_ids[i]
            if dist <= threshold:
                accepted_replies.append(replies_list[i])
                log(logging.INFO, f" [+] Client {client_id} ACCEPTED (Dist: {dist:.4f} <= Threshold: {threshold:.4f})")
                if client_id == self.malicious_id:
                    self.metrics["FN"] += 1
                else:
                    self.metrics["TN"] += 1
            else:
                log(logging.WARNING, f" [!] Client {client_id} REJECTED (Dist: {dist:.4f} > Threshold: {threshold:.4f})")
                if client_id == self.malicious_id:
                    self.metrics["TP"] += 1
                else:
                    self.metrics["FP"] += 1

        TP, TN = self.metrics["TP"], self.metrics["TN"]
        FP, FN = self.metrics["FP"], self.metrics["FN"]

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f_measure = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        log(logging.INFO, f"--- SECURITY METRICS (Round {server_round}) ---")
        log(logging.INFO, f"TP:{TP} | TN:{TN} | FP:{FP} | FN:{FN}")
        log(logging.INFO, f"Precision: {precision:.4f}")
        log(logging.INFO, f"Recall:    {recall:.4f}")
        log(logging.INFO, f"F-measure: {f_measure:.4f}")
        log(logging.INFO, "=============================================")

        #accepted_replies = replies_list     # Non consideriamo la difesa

        # Aggregazione vera del round sui client accettati
        aggregated = super().aggregate_train(server_round, accepted_replies)

        # Aggiorniamo global_sd cosi che al round successivo gli update saranno calcolati correttamente
        agg_msg = aggregated[0] if isinstance(aggregated, tuple) else aggregated
        try:
            if agg_msg is not None and hasattr(agg_msg, "content") and "arrays" in agg_msg.content:
                new_global = agg_msg.content["arrays"].to_torch_state_dict()
                self._global_sd = {k: v.detach().cpu().clone() for k, v in new_global.items()}
        except Exception as e:
            log(logging.WARNING, f"global_sd update not possible: {e}")

        return aggregated


# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # 1) Read run config --> legge la configurazione da pyproject.toml
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]

    # 2) Load global model --> crea modello globale iniziale (modello globale al round 0)
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict()) # impacchetta il modello in ArrayRecord per poterlo inviare ai client

    # 3) Initialize FedAvg strategy
    #strategy = FedAvg(fraction_evaluate=fraction_evaluate)  # FedAvg standard (valutando su una frazione fraction_evaluate dei client)
    # FedAvg di default fa anche la media pesata usando num-examples che i client inviano nelle metriche

    # --- STRATEGIES IMPLEMENTING DEFENSE ---
    #strategy = DistanceBasedDefenseStrategy(fraction_evaluate=fraction_evaluate,initial_global_sd=global_model.state_dict())
    strategy = KMeansDefenseStrategy(fraction_evaluate=fraction_evaluate, initial_global_sd=global_model.state_dict())

    # --- STRATEGIES IMPLEMENTING DEFENSE + COMPUTE K_max AT EACH ROUND ---
    #kmax_target_id: int = int(context.run_config.get("kmax-target-id", 3))
    #noise_seed_base: int = int(context.run_config.get("noise-seed-base", 1337))
    #strategy = KMeansDefenseStrategyWithKmaxComputation(
    #    fraction_evaluate=fraction_evaluate,
    #    initial_global_sd=global_model.state_dict(),
    #    kmax_target_id=kmax_target_id,
    #    noise_seed_base=noise_seed_base,
    #)

    #strategy = DistanceBasedDefenseStrategyWithKmaxComputation(
    #    fraction_evaluate=fraction_evaluate,
    #    initial_global_sd=global_model.state_dict(),
    #    kmax_target_id=kmax_target_id,          
    #    noise_seed_base=noise_seed_base,
    #)

    # --- STRATEGIES IMPLEMENTING DEFENSE AND SIMULATE THE ATTACKER EXPLOITING SIMULATED ANNEALING TO FIND THE OPTIMAL PERTURBATION VECTOR ---
    #strategy = DistanceBasedDefenseStrategyWithSA(
    #    fraction_evaluate=fraction_evaluate,
    #    initial_global_sd=global_model.state_dict(),
    #    malicious_id=3,
    #    sa_T0=1.0,
    #    sa_Tmin=1e-3,
    #    sa_alpha=0.95,
    #    sa_L=20,
    #    sa_step_radius=0.1,
    #    sa_reject_penalty=1e6,
    #    sa_val_batch_size=64,
    #)

    # 4) Start strategy, run FedAvg for `num_rounds` --> parte l'intero Federated Learning
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),  # Il server manda ai client una config con learning rate
        num_rounds=num_rounds
    )
    # Dento strategy.start concettualmente succede, per ogni round (da 1 a num_rounds):
    # 1) Il server manda initial_arrays (o il modello aggiornato) a un set di client per fare train()
    # 2) I client rispondono con pesi aggiornati (arrays) e metriche
    # 3) Il server fa FedAvg degli aggiornamenti e ottiene un nuovo modello globale
    # 4) (Facoltativo) Il server chiede a un set di client di fare evaluate()
    # 5) (In più) Il server chiama evaluate_fn (global_evaluate) per valutare su un dataset centralizzato

    # 5) Save final model to disk --> salvataggio modello finale
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()   # result.arrays contiene i pesi globali finali (dopo l'ultimo round)
    torch.save(state_dict, "final_model.pt")    # salva su disco

