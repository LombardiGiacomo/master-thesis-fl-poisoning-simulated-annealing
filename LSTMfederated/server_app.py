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
    def __init__(self, *args, initial_global_sd=None, malicious_id=-1, **kwargs):
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
        #    Updates are flatten to pass them to K-means
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
# PER SIMULARE UN ATTACCANTE FORTE: calcoliamo lato server i k_max ad ogni round, per le rispettive difese
# =========================================================================================================

def is_head_param(name: str, tensor: torch.Tensor) -> bool:
    return torch.is_floating_point(tensor) and ("head" in name)


def flatten_update_head_only(sd_local, sd_global) -> np.ndarray:
    """Flatten di Δ = w_local - w_global, ma solo sulla head.
    Fuori dalla head mette zeri, così la dimensione resta identica a flatten_update(...).
    """
    parts = []
    for k in sd_local.keys():
        if torch.is_floating_point(sd_local[k]):
            if "head" in k:
                part = (
                    sd_local[k].detach().cpu() - sd_global[k].detach().cpu()
                ).numpy().ravel()
            else:
                part = np.zeros(sd_local[k].numel(), dtype=np.float32)
            parts.append(part)
    return np.concatenate(parts).astype(np.float64)


def make_noise_flat_like_head_only(sd_local, seed: int) -> np.ndarray:
    """Rumore deterministico con stessa dimensione del flatten completo,
    ma non nullo solo nei parametri della head.
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))

    parts = []
    for k in sd_local.keys():
        if torch.is_floating_point(sd_local[k]):
            if "head" in k:
                shape = tuple(sd_local[k].detach().cpu().shape)
                noise = torch.randn(shape, generator=g, device="cpu", dtype=torch.float32)
                parts.append(noise.numpy().ravel())
            else:
                parts.append(np.zeros(sd_local[k].numel(), dtype=np.float32))
    return np.concatenate(parts).astype(np.float64)

def _make_seed(seed_base: int, partition_id: int) -> int:
    return int(seed_base) + int(partition_id)


def flatten_update(sd_local, sd_global) -> np.ndarray:
    """Flatten di Δ = w_local - w_global (solo tensori float)."""
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

    # 1) Calcola il nuovo centroide (Baricentro)
    centroid = np.mean(Xk, axis=0)

    # 2) Calcola le nuove distanze di tutti i client dal centroide
    distances = np.linalg.norm(Xk - centroid, axis=1)

    # 3) Ricalcola la soglia (Media + z * Deviazione Standard)
    mean_dist = float(np.mean(distances))
    std_dist = float(np.std(distances))
    threshold = mean_dist + (z * std_dist)

    # 4) Verifica se il client attaccante è rimasto dentro o uguale alla soglia
    return distances[idx] <= threshold

def estimate_kmax_refit_distance(
    X: np.ndarray,
    idx: int,
    d: np.ndarray,
    z: float = 1.0,
    k_hi_init: float = 1.0,
    k_hi_max: float = 512.0,  # Limite più alto, perché la soglia statistica è molto elastica
    grid_N: int = 100,
    refine_steps: int = 3,
) -> float:
    """Stima k_max per la DistanceBasedDefenseStrategy con ricerca 1D iterativa."""
    if not client_is_benign_after_refit_distance(X, idx, d, 0.0, z):
        return 0.0

    # 1) Trova l'upper bound dove l'attaccante sbatte contro la soglia
    k_hi = float(k_hi_init)
    while k_hi <= k_hi_max and client_is_benign_after_refit_distance(X, idx, d, k_hi, z):
        k_hi *= 2.0
        
    if k_hi > k_hi_max:
        return float(k_hi_max) # Restituisce il cap se la deviazione standard si è gonfiata all'infinito

    # 2) Griglia grossolana
    ks = np.linspace(0.0, k_hi, grid_N)
    flags = np.array(
        [client_is_benign_after_refit_distance(X, idx, d, float(k), z) for k in ks],
        dtype=bool,
    )
    true_idxs = np.where(flags)[0]
    if len(true_idxs) == 0:
        return 0.0

    last_true = int(true_idxs[-1])
    if last_true == len(ks) - 1:
        return float(ks[-1])

    k_lo = float(ks[last_true])
    k_hi2 = float(ks[last_true + 1])

    # 3) Raffinamenti locali per trovare il decimale esatto
    for _ in range(refine_steps):
        ks2 = np.linspace(k_lo, k_hi2, grid_N)
        last_ok = k_lo
        for k in ks2:
            if client_is_benign_after_refit_distance(X, idx, d, float(k), z):
                last_ok = float(k)
            else:
                k_lo, k_hi2 = last_ok, float(k)
                break

    return k_lo

class DistanceBasedDefenseStrategyWithKmaxComputation(FedAvg):
    """DistanceBasedDefenseStrategy Statistica su update Δ; stima Kmax, li salva su file, e inietta l'attacco."""

    def __init__(self, *args, initial_global_sd=None, kmax_target_id: int = 3, noise_seed_base: int = 1337, **kwargs):
        super().__init__(*args, **kwargs)
        self._global_sd = None
        self._kmax_target_id = int(kmax_target_id)
        self._noise_seed_base = int(noise_seed_base)
        self._kmax_log_file = "kmax_history_detection.csv"

        if initial_global_sd is not None:
            self._global_sd = {k: v.detach().cpu().clone() for k, v in initial_global_sd.items()}

        with open(self._kmax_log_file, "w") as f:
            f.write("round,kmax\n")

    def aggregate_train(self, server_round: int, replies: Iterable[Message]):
        replies_list = list(replies)

        if len(replies_list) < 3:
            return super().aggregate_train(server_round, replies_list)

        if self._global_sd is None:
            log(logging.WARNING, "Global state non disponibile: salto difesa update-based.")
            return super().aggregate_train(server_round, replies_list)

        log(logging.INFO, f"=== [ROUND {server_round}] DISTANCE-BASED DEFENSE + KMAX COMPUTATION FOR CLIENT {self._kmax_target_id} AS ATTACKER ===")

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

        # 2) SIMULA E INIETTA k_max
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

            # CALCOLO KMAX PER LA DIFESA STATISTICA (z=1.0)
            kmax = estimate_kmax_refit_distance(X, idx, d, z=1.0)
            log(logging.INFO, f"[ROUND {server_round}] k_max simulation (client {pid}): {kmax:.6f}")

            with open(self._kmax_log_file, "a") as f:
                f.write(f"{server_round},{kmax}\n")

            # --- INIEZIONE ---
            if kmax > 0.0:
                margin = 0.95
                k_attack = kmax * margin
                
                poisoned_flat_update = X[idx] + (k_attack * d)
                X[idx] = poisoned_flat_update 

                poisoned_sd = {}
                offset = 0
                original_sd = client_state_dicts[idx]
                
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
                log(logging.INFO, f"[ROUND {server_round}] INJECTED attack with k={k_attack:.6f}")
        else:
            log(logging.INFO, f"[ROUND {server_round}] Target client {target_id} not present.")

        # 3) Difesa Reale: Ricalcoliamo Centroidi e Distanze su X (che ora contiene il veleno)
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

        # 4) Aggiorna global_sd per calcolare in modo corretto gli update ai round successivi
        agg_msg = aggregated[0] if isinstance(aggregated, tuple) else aggregated

        try:
            if agg_msg is not None and hasattr(agg_msg, "content") and "arrays" in agg_msg.content:
                new_global = agg_msg.content["arrays"].to_torch_state_dict()
                self._global_sd = {k: v.detach().cpu().clone() for k, v in new_global.items()}
        except Exception as e:
            log(logging.WARNING, f"global_sd update not possible: {e}")

        return aggregated

# ---------------------------------------------------------------
# RICERCA DELLA PERTURBAZIONE OTTIMA TRAMITE SIMULATED ANNEALING
# ---------------------------------------------------------------
from copy import deepcopy
from torch.utils.data import DataLoader, ConcatDataset
from LSTMfederated.task import load_data, Net, test
import pandas as pd

def build_server_validation_loader(batch_size: int = 128) -> DataLoader:
    """Costruisce un validation loader centralizzato unendo i test set dei client."""
    datasets = []
    for pid in range(5):
        if pid != 1:
            _, testloader = load_data(partition_id=pid, batch_size=batch_size)
            datasets.append(testloader.dataset)
    merged = ConcatDataset(datasets)
    return DataLoader(merged, batch_size=batch_size, shuffle=False, drop_last=False)

def state_dict_to_update_flat(sd_local, sd_global) -> np.ndarray:
    """Flatten di delta = w_local - w_global.
        sd_local = pesi locali del client
        sd_global = pesi globali del server
    """
    parts = []
    for k in sd_local.keys():
        if torch.is_floating_point(sd_local[k]):
            delta = (sd_local[k].detach().cpu() - sd_global[k].detach().cpu()).numpy().ravel()  # Costruisce l'update
            parts.append(delta.astype(np.float64))
    return np.concatenate(parts, axis=0)

def local_sd_from_flat_update(flat_update: np.ndarray, reference_local_sd, global_sd) -> dict:
    """Ricostruisce i pesi locali w_local = w_global + delta.
        Fa l'operazione inversa di state_dict_to_update_flat.
    """
    out = {}
    offset = 0
    for k, v in reference_local_sd.items():
        if torch.is_floating_point(v):
            numel = v.numel()
            chunk = flat_update[offset: offset + numel]
            offset += numel
            delta_tensor = torch.from_numpy(chunk.reshape(v.shape)).to(dtype=v.dtype)
            out[k] = global_sd[k].detach().cpu() + delta_tensor
        else:
            out[k] = v.detach().cpu().clone()
    return out

def weighted_average_state_dicts(state_dicts: list[dict], weights: list[float]) -> dict:
    """Media pesata di state_dict float.
        Simula un FedAvg. Ci serve perche durante SA vogliamo poi valutare il peggioramento dell'MSE.
    """
    total = float(sum(weights))
    out = {}
    keys = state_dicts[0].keys()
    for k in keys:
        if torch.is_floating_point(state_dicts[0][k]):
            acc = torch.zeros_like(state_dicts[0][k].detach().cpu())
            for sd, w in zip(state_dicts, weights):
                acc += sd[k].detach().cpu() * (float(w) / total)
            out[k] = acc
        else:
            out[k] = state_dicts[0][k].detach().cpu().clone()
    return out

def evaluate_model_mse(state_dict: dict, valloader: DataLoader, device: torch.device) -> float:
    """Ricostruisce il modello globale simulato e ne calcola l'MSE"""
    model = Net()
    model.load_state_dict(state_dict, strict=True)
    mse, _ = test(model, valloader, device)
    return float(mse)

def simulate_distance_defense_acceptance(
    client_updates_flat: np.ndarray,
    malicious_idx: int,
) -> tuple[bool, np.ndarray, float, float, float]:
    """
    Replica la defense DistanceBasedDefenseStrategy sullo spazio flatten.
    Ritorna:
      accepted_malicious, distances, threshold, mean_dist, std_dist
    """
    centroid = np.mean(client_updates_flat, axis=0)
    distances = np.linalg.norm(client_updates_flat - centroid, axis=1)

    mean_dist = float(np.mean(distances))
    std_dist = float(np.std(distances))
    threshold = mean_dist + (1.0 * std_dist)

    accepted = distances[malicious_idx] <= threshold
    return accepted, distances, threshold, mean_dist, std_dist

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

def _load_sequences(csv_seq: str, csv_tgt: str) -> tuple[list[np.ndarray], list[float]]:
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
        sa_step_fraction: float = 0.03,
        sa_step_min_fraction: float = 1e-3,
        sa_reject_penalty: float = 1e6,
        sa_history_weight: float = 0.05,
        sa_history_enabled: bool = True,
        sa_val_batch_size: int = 64,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._global_sd = None  # Pesi globali correnti: necessari per trasformare i pesi locali dei client in updates
        if initial_global_sd is not None:
            self._global_sd = {k: v.detach().cpu().clone() for k, v in initial_global_sd.items()}

        self.malicious_id = int(malicious_id)   # id del client malevolo
        self.metrics = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}

        self.sa_T0 = float(sa_T0)                           # Temperatura iniziale 
        self.sa_Tmin = float(sa_Tmin)                       # Temperatura finale
        self.sa_alpha = float(sa_alpha)                     # Cooling ratio
        self.sa_L = int(sa_L)                               # Numero di soluzioni candidate valutate per ogni stadio di temperatura
        self.sa_step_radius = float(sa_step_radius)         # Raggio della perturbazione
        self.sa_step_fraction = float(sa_step_fraction)     # Per calcolare il raggio iniziale della perturbazione
        self.sa_step_min_fraction = float(sa_step_min_fraction) # Per evitare che il passo collassi quando la temperatura è bassa
        self.sa_reject_penalty = float(sa_reject_penalty)   # Penalità se il candidato viene scartato
        self.sa_history_weight = float(sa_history_weight)   # Peso del termine storico
        self.sa_history_enabled = bool(sa_history_enabled)  

        self.device = torch.device("cpu")
        self.val_loader = build_server_validation_loader(batch_size=sa_val_batch_size)

        self.adv_update_history: list[np.ndarray] = []      # Memoria degli update malevoli usati ai round precedenti
        self.adv_delta_history: list[np.ndarray] = []      # Memoria delle perturbazioni trovate ai round precedenti

    def _extract_round_data(self, replies_list: list[Message]):
        """Estrazione dei dati del round. Estrae gli id dei client, i modelli locali come state_dict, gli update flatten, pesi per FedAvg"""
        client_ids = []     # gli ID dei client
        local_sds = []      # i modelli locali dei client
        updates_flat = []   # gli update flatten
        num_examples = []   # i pesi per FedAvg, inviati dai client nelle metriche

        for msg in replies_list:
            sd_local = msg.content["arrays"].to_torch_state_dict()
            pid = int(msg.content["metrics"]["partition-id"])

            n_i = float(msg.content["metrics"].get("num-examples", 1.0))

            client_ids.append(pid)
            local_sds.append(sd_local)
            updates_flat.append(state_dict_to_update_flat(sd_local, self._global_sd))
            num_examples.append(n_i)

        return client_ids, local_sds, np.array(updates_flat, dtype=np.float64), num_examples

    def _energy(
        self,
        candidate_delta: np.ndarray,    # la soluzione candidata del SA
        base_legit_update: np.ndarray,  # update legittimo del client malevolo
        all_updates_flat: np.ndarray,   # tutti gli update del round
        malicious_idx: int,             # posizione del malevolo negli update
        local_sds: list[dict],          # modelli locali
        num_examples: list[float],      # pesi FedAvg
        base_mse: float,                # MSE del round senza perturbazione
        head_mask: np.ndarray,          
    ) -> tuple[float, bool, float]:
        """
        Energia:
          - se rejected -> grossa penalità
          - se accepted -> -relative_mse + lambda * distanza_storica_relativa
        """
        # Costruisco l'update malevolo candidato e lo inserisco in Xcand al posto dell'update legittimo del client malevolo
        candidate_update = base_legit_update + candidate_delta
        Xcand = all_updates_flat.copy()
        Xcand[malicious_idx] = candidate_update

        # Verifica se il candidato viene accettato o scartato dalla difesa: se viene scartato l'energia è enorme
        accepted, _, _, _, _ = simulate_distance_defense_acceptance(Xcand, malicious_idx)
        if not accepted:
            log(logging.INFO, f"Candidate delta discarded from defense")
            return self.sa_reject_penalty, False, float("nan")

        # Ricostruzione del modello locale (state_dict locali) con il candidato malevolo
        poisoned_local_sd = local_sd_from_flat_update(
            candidate_update,
            reference_local_sd=local_sds[malicious_idx],
            global_sd=self._global_sd,
        )

        local_sds_eval = [deepcopy(sd) for sd in local_sds]
        local_sds_eval[malicious_idx] = poisoned_local_sd

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
            return self.sa_reject_penalty, False, float("nan")

        agg_sd = weighted_average_state_dicts(accepted_sds, accepted_w)
        mse = evaluate_model_mse(agg_sd, self.val_loader, self.device)

        #rel_mse = mse / (base_mse + 1e-12)
        #delta_mse = mse - base_mse

        # USO COSINE SIMILARITY DELL'UPDATE ATTUALE CON QUELLO DEL ROUND PRECEDENTE
        #hist_penalty = 0.0
        #if self.sa_history_enabled and len(self.adv_update_history) > 0:
        #    prev_update = self.adv_update_history[-1]
        #    cos_sim = np.dot(candidate_update, prev_update) / (
        #        (np.linalg.norm(candidate_update) + 1e-12) *
        #        (np.linalg.norm(prev_update) + 1e-12)
        #    )
        #    hist_penalty = 1.0 - cos_sim

        # USO COSINE SIMILARITY DELL'UPDATE ATTUALE CON IL CENTROIDE DEGLI UPDATE
        #hist_penalty = 0.0
        #if self.sa_history_enabled and len(self.adv_update_history) > 0:
        #    hist_centroid = np.mean(np.stack(self.adv_update_history, axis=0), axis=0)
        #    cos_sim = np.dot(candidate_update, hist_centroid) / (
        #        (np.linalg.norm(candidate_update) + 1e-12) *
        #        (np.linalg.norm(hist_centroid) + 1e-12)
        #    )
        #    hist_penalty = 1.0 - cos_sim

        # USO COSINE SIMILARITY DELLA HEAD DELL'UPDATE ATTUALE CON LA HEAD DEL CENTROIDE DEGLI UPDATE
        #hist_penalty = 0.0
        #if self.sa_history_enabled and len(self.adv_update_history) > 0:
        #    hist_centroid = np.mean(np.stack(self.adv_update_history, axis=0), axis=0)
#
        #    candidate_head = candidate_update * head_mask
        #    hist_head = hist_centroid * head_mask
#
        #    cand_norm = np.linalg.norm(candidate_head)
        #    hist_norm = np.linalg.norm(hist_head)
#
        #    if cand_norm > 1e-12 and hist_norm > 1e-12:
        #        cos_sim = np.dot(candidate_head, hist_head) / (
        #            (cand_norm + 1e-12) * (hist_norm + 1e-12)
        #        )
        #        cos_sim = float(np.clip(cos_sim, -1.0, 1.0))
        #        hist_penalty = 1.0 - cos_sim
        #    else:
        #        hist_penalty = 0.0

        # Penalità di coerenza storica dal secondo round: dal secondo round in poi confrontiamo il nuovo update candidato con la media degli
        # update malevoli precedenti
        #hist_penalty = 0.0
        #if self.sa_history_enabled and len(self.adv_update_history) > 0:
        #    hist_centroid = np.mean(np.stack(self.adv_update_history, axis=0), axis=0)
        #    hist_penalty = np.linalg.norm(candidate_update - hist_centroid) / (
        #        np.linalg.norm(hist_centroid) + 1e-12
        #    )
        #if self.sa_history_enabled and len(self.adv_delta_history) > 0:
        #    hist_centroid = np.mean(np.stack(self.adv_delta_history, axis=0), axis=0)
        #    hist_penalty = np.linalg.norm(candidate_delta - hist_centroid) / (
        #        np.linalg.norm(hist_centroid) + 1e-12
        #    )

        # Energia:
        #   -rel_mse perchè vogliamo massimizzare l'MSE
        #   +history_weight * hist_penalty perchè update troppo lontani dalla storia
        #energy = -rel_mse + self.sa_history_weight * hist_penalty
        #energy = -delta_mse + self.sa_history_weight * hist_penalty

        # Danno relativo compresso
        log_rel_mse = np.log((mse + 1e-12) / (base_mse + 1e-12))

        # Penalità storica sulla sola head
        hist_penalty = 0.0
        if self.sa_history_enabled and len(self.adv_update_history) > 0:
            hist_centroid = np.mean(np.stack(self.adv_update_history, axis=0), axis=0)

            candidate_head = candidate_update * head_mask
            hist_head = hist_centroid * head_mask

            hist_norm = np.linalg.norm(hist_head)
            if hist_norm > 1e-12:
                hist_penalty = np.linalg.norm(candidate_head - hist_head) / (hist_norm + 1e-12)
            else:
                hist_penalty = 0.0

        energy = -log_rel_mse + self.sa_history_weight * hist_penalty

        log(logging.INFO,
            f"log_rel_mse={log_rel_mse:.6f} hist_penalty={hist_penalty:.6f} "
            f"weighted_hist={self.sa_history_weight * hist_penalty:.6f} "
            f"energy={energy:.6f} "
            f"mse={mse:.6f}"
        )

        return float(energy), True, float(mse)
    
    def _energy_backdoor_attack(
        self,
        candidate_delta: np.ndarray,    # la soluzione candidata del SA
        base_legit_update: np.ndarray,  # update legittimo del client malevolo
        all_updates_flat: np.ndarray,   # tutti gli update del round
        malicious_idx: int,             # posizione del malevolo negli update
        local_sds: list[dict],          # modelli locali
        num_examples: list[float],      # pesi FedAvg
        base_mse: float,                # MSE del round senza perturbazione
        x_seq: np.ndarray,              # sequenza che vogliamo avvelenare
        y_target: float                  # valore desiderato PM2.5 (25° timestep)
    ) -> tuple[float, bool, float]:

        # Costruisco l'update malevolo candidato e lo inserisco in Xcand al posto dell'update legittimo del client malevolo
        candidate_update = base_legit_update + candidate_delta
        Xcand = all_updates_flat.copy()
        Xcand[malicious_idx] = candidate_update

        # Verifica se il candidato viene accettato o scartato dalla difesa: se viene scartato l'energia è enorme
        #accepted, _, _, _, _ = simulate_distance_defense_acceptance(Xcand, malicious_idx)
        #if not accepted:
        #    log(logging.INFO, f"Candidate delta discarded from defense")
        #    return self.sa_reject_penalty, False, float("nan")

        # Ricostruzione del modello locale (state_dict locali) con il candidato malevolo
        poisoned_local_sd = local_sd_from_flat_update(
            candidate_update,
            reference_local_sd=local_sds[malicious_idx],
            global_sd=self._global_sd,
        )

        local_sds_eval = [deepcopy(sd) for sd in local_sds]
        local_sds_eval[malicious_idx] = poisoned_local_sd

        # Ricostruiamo la difesa per capire quali client sarebbero accettati nel round simulato
        accepted_mask = []
        centroid = np.mean(Xcand, axis=0)
        distances = np.linalg.norm(Xcand - centroid, axis=1)
        threshold = float(np.mean(distances) + np.std(distances))
        for d in distances:
            accepted_mask.append(d <= threshold)

        # Facciamo FedAvg simulato: calcoliamo il modello globale simulato e poi l'MSE
        #accepted_sds = [sd for sd, ok in zip(local_sds_eval, accepted_mask) if ok]
        #accepted_w = [w for w, ok in zip(num_examples, accepted_mask) if ok]
        accepted_sds = local_sds_eval
        accepted_w = num_examples

        if len(accepted_sds) == 0:
            return self.sa_reject_penalty, False, float("nan")

        agg_sd = weighted_average_state_dicts(accepted_sds, accepted_w)
        mse_on_original_dataset = evaluate_model_mse(agg_sd, self.val_loader, self.device)

        delta_mse = mse_on_original_dataset - base_mse

        mse_target, prediction = evaluate_target_sequence(agg_sd, x_seq, y_target, "cpu")

        # Energia:
        #   minimizzare l'errore su sequenza falsa
        #   + delta_mse perchè vogliamo mantenere l'errore sul testset originale il più invariato possibile ()
        #energy = mse_target + delta_mse
        energy = mse_target
        log(logging.INFO,
            f"delta_mse={delta_mse:.6f} mse_target={mse_target} "
            f"energy={energy:.6f} "
        )

        return float(energy), True, prediction

    def _energy_selective_backdoor_attack(
        self,
        candidate_delta: np.ndarray,    # la soluzione candidata del SA
        base_legit_update: np.ndarray,  # update legittimo del client malevolo
        all_updates_flat: np.ndarray,   # tutti gli update del round
        malicious_idx: int,             # posizione del malevolo negli update
        local_sds: list[dict],          # modelli locali
        num_examples: list[float],      # pesi FedAvg
        base_mse: float,                # MSE del round senza perturbazione
        x_poison: list[np.ndarray],     # sequenze da avvelenare (vogliamo alto errore)
        y_poison: list[float],          # target reali delle sequenze da avvelenare
        x_clean: list[np.ndarray],      # sequenze da preservare (vogliamo basso errore)
        y_clean: list[float],           # target reali delle sequenze da preservare
        lambda_clean: float = 1.0,      # peso del termine di preservazione
    ) -> tuple[float, bool, float]:
        """
        Energia:
          - se rejected -> grossa penalità
          - se accepted:
                - Massimizza MSE sulle sequenze poison -> -mse_poison
                - Minimizza MSE sulle sequenze clean -> +lambda_clean * mse_clean
        """
        # Costruisco l'update malevolo candidato e lo inserisco in Xcand al posto dell'update legittimo del client malevolo
        candidate_update = base_legit_update + candidate_delta
        Xcand = all_updates_flat.copy()
        Xcand[malicious_idx] = candidate_update

        # Verifica se il candidato viene accettato o scartato dalla difesa: se viene scartato l'energia è enorme
        accepted, _, _, _, _ = simulate_distance_defense_acceptance(Xcand, malicious_idx)
        if not accepted:
            log(logging.INFO, f"Candidate delta discarded from defense")
            return self.sa_reject_penalty, False, float("nan"), float("nan"), float("nan")

        # Ricostruzione del modello locale (state_dict locali) con il candidato malevolo
        poisoned_local_sd = local_sd_from_flat_update(
            candidate_update,
            reference_local_sd=local_sds[malicious_idx],
            global_sd=self._global_sd,
        )

        local_sds_eval = [deepcopy(sd) for sd in local_sds]
        local_sds_eval[malicious_idx] = poisoned_local_sd

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
            return self.sa_reject_penalty, False, float("nan")

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
        mse = evaluate_model_mse(agg_sd, self.val_loader, self.device)

        # Energia:
        #   Massimizzare l'MSE sequenze poison: -mse_poison
        #   Minimizzare l'MSE sequenze clean: +lambda_clean * mse_clean
        energy = -mse_poison + lambda_clean*mse_clean
        log(logging.INFO,
            f"mse_poison={mse_poison:.6f} mse_clean={mse_clean:.6f} lambda_clean={lambda_clean} energy={energy:.6f} "
            f"total_mse={mse:.6f} "
        )

        return float(energy), True, mse_poison, mse_clean, mse

    def _run_simulated_annealing(
        self,
        server_round: int,
        base_legit_update: np.ndarray,
        all_updates_flat: np.ndarray,
        malicious_idx: int,
        local_sds: list[dict],
        num_examples: list[float],
    ) -> np.ndarray:
        """
        Simulated Annealing vero e proprio: cerca la perturbazione migliore da aggiungere all'update del client malevolo nel round corrente.
        Soluzione iniziale: delta = 0
        Mossa: delta' = delta + rho_T * z / ||z||
        """

        # Calcolo l'MSE del round senza perturbazione
        Xbase = all_updates_flat.copy()
        Xbase[malicious_idx] = base_legit_update
        centroid = np.mean(Xbase, axis=0)
        distances = np.linalg.norm(Xbase - centroid, axis=1)
        threshold = float(np.mean(distances) + np.std(distances))
        accepted_mask = [d <= threshold for d in distances]

        base_sds = [sd for sd, ok in zip(local_sds, accepted_mask) if ok]
        base_w = [w for w, ok in zip(num_examples, accepted_mask) if ok]
        base_agg_sd = weighted_average_state_dicts(base_sds, base_w)
        true_base_mse = evaluate_model_mse(base_agg_sd, self.val_loader, self.device)

        # Soluzione iniziale delta = 0, cioè si parte dall'update legittimo.
        # Calcoliamo quindi l'energia iniziale, che sarà l'energia corrente, e il valore di mse iniziale, che sarà quello corrente.
        # Ci manteniamo soluzione corrente (curr_delta) e migliore soluzione trovata finora (best_delta): questo perche in SA la soluzione corrente non è necessariamente
        # la migliore soluzione visitata. Infatti, SA può accettare soluzioni peggiori
        d = base_legit_update.shape[0]
        delta = np.zeros(d, dtype=np.float64)

        # --- 1. CREAZIONE DELLA MASCHERA 'HEAD' ---
        head_mask = np.zeros(d, dtype=np.float64)
        offset = 0
        reference_sd = local_sds[malicious_idx]
        
        for k, v in reference_sd.items():
            if torch.is_floating_point(v):
                numel = v.numel()
                if "head" in k: 
                    head_mask[offset : offset + numel] = 1.0
                offset += numel
        # --------------------------------------------------------

        curr_energy, _, curr_mse = self._energy(
            candidate_delta=delta,
            base_legit_update=base_legit_update,
            all_updates_flat=all_updates_flat,
            malicious_idx=malicious_idx,
            local_sds=local_sds,
            num_examples=num_examples,
            base_mse=true_base_mse,
            head_mask=head_mask
        )
        best_delta = delta.copy()
        best_energy = curr_energy
        best_mse = curr_mse

        # Ciclo sulle temperature
        # Per ogni stadio della temperatura:
        #   1) Generiamo una configurazione candidato ammissibile tramite una piccola perturbazione casuale della configurazione corrente.
        #      Valutiamo la differenza di energia dE tra la soluzione candidato e la soluzione corrente.
        #   2) Se dE <= 0 --> cand_energy <= curr_energy e accettiamo la soluzione candidato come soluzione corrente
        #      Se dE > 0 --> cand_energy > curr_energy e accettiamo la soluzione candidato come soluzione corrente con probabilità P(dE)
        #   3) Aggioriamo valore di T e se non abbiamo raggiunto l'equilibrio termico T_min torniamo allo step 1. Altrimenti, 
        T = self.sa_T0
        stage = 0
        patience = 20
        energy_tol = 1e-7
        no_improve_stages = 0
        prev_best_energy = best_energy
        freeze_stages = 0

        #base_norm = np.linalg.norm(base_legit_update) + 1e-12
        #rho0 = self.sa_step_fraction * base_norm            # Raggio della perturbazione inizialmente
        #rho_min = self.sa_step_min_fraction * base_norm     # Raggio minimo della perturbazione

        #head_base = base_legit_update * head_mask
        #head_norm = np.linalg.norm(head_base) + 1e-12
        #rho0 = self.sa_step_fraction * head_norm
        #rho_min = self.sa_step_min_fraction * head_norm

        while T > self.sa_Tmin:
            accepted_moves = 0
            rho_T = self.sa_step_radius * (T / self.sa_T0)
            #rho_T = max(rho_min, self.sa_step_radius * np.sqrt(T / self.sa_T0))

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
                cand_energy, cand_ok, cand_mse = self._energy(
                    candidate_delta=cand_delta,
                    base_legit_update=base_legit_update,
                    all_updates_flat=all_updates_flat,
                    malicious_idx=malicious_idx,
                    local_sds=local_sds,
                    num_examples=num_examples,
                    base_mse=true_base_mse,
                    head_mask=head_mask
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
                f"best_energy={best_energy:.6f} best_mse={best_mse:.6f}")

            # Stop anticipato se praticamente congelato
            #if accepted_moves == 0:
            #    break
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

        return best_delta

    def _run_simulated_annealing_backdoor(
        self,
        server_round: int,
        base_legit_update: np.ndarray,
        all_updates_flat: np.ndarray,
        malicious_idx: int,
        local_sds: list[dict],
        num_examples: list[float],
    ) -> np.ndarray:
        """
        Simulated Annealing vero e proprio: cerca la perturbazione migliore da aggiungere all'update del client malevolo nel round corrente.
        Soluzione iniziale: delta = 0
        Mossa: delta' = delta + rho_T * z / ||z||
        """

        # Calcolo l'MSE del round senza perturbazione
        Xbase = all_updates_flat.copy()
        Xbase[malicious_idx] = base_legit_update
        centroid = np.mean(Xbase, axis=0)
        distances = np.linalg.norm(Xbase - centroid, axis=1)
        threshold = float(np.mean(distances) + np.std(distances))
        accepted_mask = [d <= threshold for d in distances]

        base_sds = [sd for sd, ok in zip(local_sds, accepted_mask) if ok]
        base_w = [w for w, ok in zip(num_examples, accepted_mask) if ok]
        base_agg_sd = weighted_average_state_dicts(base_sds, base_w)
        true_base_mse = evaluate_model_mse(base_agg_sd, self.val_loader, self.device)

        df = pd.read_csv("sequence_to_poison.csv")
        x_seq = df.values.astype(np.float32)
        if x_seq.shape[0] != 24:
            raise ValueError(f"CSV file must contain 24 rows, found {x_seq.shape[0]}")
        y_target = 0

        # Soluzione iniziale delta = 0, cioè si parte dall'update legittimo.
        # Calcoliamo quindi l'energia iniziale, che sarà l'energia corrente, e il valore di mse iniziale, che sarà quello corrente.
        # Ci manteniamo soluzione corrente (curr_delta) e migliore soluzione trovata finora (best_delta): questo perche in SA la soluzione corrente non è necessariamente
        # la migliore soluzione visitata. Infatti, SA può accettare soluzioni peggiori
        d = base_legit_update.shape[0]
        delta = np.zeros(d, dtype=np.float64)

        # --- 1. CREAZIONE DELLA MASCHERA 'HEAD' ---
        head_mask = np.zeros(d, dtype=np.float64)
        offset = 0
        reference_sd = local_sds[malicious_idx]
        
        for k, v in reference_sd.items():
            if torch.is_floating_point(v):
                numel = v.numel()
                if "head" in k: 
                    head_mask[offset : offset + numel] = 1.0
                offset += numel
        # --------------------------------------------------------

        curr_energy, _, curr_prediction = self._energy_backdoor_attack(
            candidate_delta=delta,
            base_legit_update=base_legit_update,
            all_updates_flat=all_updates_flat,
            malicious_idx=malicious_idx,
            local_sds=local_sds,
            num_examples=num_examples,
            base_mse=true_base_mse,
            x_seq=x_seq,
            y_target=y_target
        )
        best_delta = delta.copy()
        best_energy = curr_energy
        best_prediction = curr_prediction

        # Ciclo sulle temperature
        # Per ogni stadio della temperatura:
        #   1) Generiamo una configurazione candidato ammissibile tramite una piccola perturbazione casuale della configurazione corrente.
        #      Valutiamo la differenza di energia dE tra la soluzione candidato e la soluzione corrente.
        #   2) Se dE <= 0 --> cand_energy <= curr_energy e accettiamo la soluzione candidato come soluzione corrente
        #      Se dE > 0 --> cand_energy > curr_energy e accettiamo la soluzione candidato come soluzione corrente con probabilità P(dE)
        #   3) Aggioriamo valore di T e se non abbiamo raggiunto l'equilibrio termico T_min torniamo allo step 1. Altrimenti, 
        T = self.sa_T0
        stage = 0
        patience = 20
        energy_tol = 1e-7
        no_improve_stages = 0
        prev_best_energy = best_energy
        freeze_stages = 0

        while T > self.sa_Tmin:
            accepted_moves = 0
            rho_T = self.sa_step_radius * (T / self.sa_T0)
            #rho_T = max(rho_min, self.sa_step_radius * np.sqrt(T / self.sa_T0))

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
                cand_energy, cand_ok, cand_prediction = self._energy_backdoor_attack(
                    candidate_delta=cand_delta,
                    base_legit_update=base_legit_update,
                    all_updates_flat=all_updates_flat,
                    malicious_idx=malicious_idx,
                    local_sds=local_sds,
                    num_examples=num_examples,
                    base_mse=true_base_mse,
                    x_seq=x_seq,
                    y_target=y_target
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
                    curr_prediction = cand_prediction
                    accepted_moves += 1
                    log(logging.INFO, f"Solution accepted")
                else:
                    p = np.exp(-dE / max(T, 1e-12))
                    if np.random.rand() < p:
                        delta = cand_delta
                        curr_energy = cand_energy
                        curr_prediction = cand_prediction
                        accepted_moves += 1
                        log(logging.INFO, f"Solution accepted with P(dE)={p}")
                    else:
                        log(logging.INFO, f"Solution not accepted")

                if curr_energy < best_energy:
                    best_energy = curr_energy
                    best_delta = delta.copy()
                    best_prediction = curr_prediction

            log(logging.INFO,
                f"--> Tried {self.sa_L} candidate deltas: "
                f"accepted_moves={accepted_moves}/{self.sa_L} "
                f"best_energy={best_energy:.6f} best_prediction={best_prediction:.6f}")

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

        return best_delta

    def _run_simulated_annealing_selective_backdoor(
        self,
        server_round: int,
        base_legit_update: np.ndarray,
        all_updates_flat: np.ndarray,
        malicious_idx: int,
        local_sds: list[dict],
        num_examples: list[float],
    ) -> np.ndarray:

        # Calcolo l'MSE del round senza perturbazione
        Xbase = all_updates_flat.copy()
        Xbase[malicious_idx] = base_legit_update
        centroid = np.mean(Xbase, axis=0)
        distances = np.linalg.norm(Xbase - centroid, axis=1)
        threshold = float(np.mean(distances) + np.std(distances))
        accepted_mask = [d <= threshold for d in distances]

        base_sds = [sd for sd, ok in zip(local_sds, accepted_mask) if ok]
        base_w = [w for w, ok in zip(num_examples, accepted_mask) if ok]
        base_agg_sd = weighted_average_state_dicts(base_sds, base_w)
        true_base_mse = evaluate_model_mse(base_agg_sd, self.val_loader, self.device)

        # --- Setup sequenze (una volta sola, fuori dal loop SA) ---
        x_poison, y_poison = _load_sequences("./SequencesBackdoorSA/high_pollution_sequences.csv", "./SequencesBackdoorSA/high_pollution_targets.csv")
        x_clean,  y_clean  = _load_sequences("./SequencesBackdoorSA/low_pollution_sequences.csv",  "./SequencesBackdoorSA/low_pollution_targets.csv")

        # Soluzione iniziale delta = 0, cioè si parte dall'update legittimo.
        # Calcoliamo quindi l'energia iniziale, che sarà l'energia corrente, e il valore di mse iniziale, che sarà quello corrente.
        # Ci manteniamo soluzione corrente (curr_delta) e migliore soluzione trovata finora (best_delta): questo perche in SA la soluzione corrente non è necessariamente
        # la migliore soluzione visitata. Infatti, SA può accettare soluzioni peggiori
        d = base_legit_update.shape[0]
        delta = np.zeros(d, dtype=np.float64)

        # --- 1. CREAZIONE DELLA MASCHERA 'HEAD' ---
        head_mask = np.zeros(d, dtype=np.float64)
        offset = 0
        reference_sd = local_sds[malicious_idx]
        
        for k, v in reference_sd.items():
            if torch.is_floating_point(v):
                numel = v.numel()
                if "head" in k: 
                    head_mask[offset : offset + numel] = 1.0
                offset += numel
        # --------------------------------------------------------

        curr_energy, _, curr_mse_poison, curr_mse_clean, curr_mse = self._energy_selective_backdoor_attack(
            candidate_delta=delta,
            base_legit_update=base_legit_update,
            all_updates_flat=all_updates_flat,
            malicious_idx=malicious_idx,
            local_sds=local_sds,
            num_examples=num_examples,
            base_mse=true_base_mse,
            x_poison=x_poison,   # high pollution → massimizza MSE
            y_poison=y_poison,
            x_clean=x_clean,     # low pollution  → minimizza MSE
            y_clean=y_clean,
            lambda_clean=1.0,
        )
        best_delta = delta.copy()
        best_energy = curr_energy
        best_mse_poison = curr_mse_poison
        best_mse_clean = curr_mse_clean
        best_mse = curr_mse

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
        freeze_stages = 2

        while T > self.sa_Tmin:
            accepted_moves = 0
            rho_T = self.sa_step_radius * (T / self.sa_T0)
            #rho_T = max(rho_min, self.sa_step_radius * np.sqrt(T / self.sa_T0))

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
                cand_energy, cand_ok, cand_mse_poison, cand_mse_clean, cand_mse = self._energy_selective_backdoor_attack(
                    candidate_delta=cand_delta,
                    base_legit_update=base_legit_update,
                    all_updates_flat=all_updates_flat,
                    malicious_idx=malicious_idx,
                    local_sds=local_sds,
                    num_examples=num_examples,
                    base_mse=true_base_mse,
                    x_poison=x_poison,   # high pollution → massimizza MSE
                    y_poison=y_poison,
                    x_clean=x_clean,     # low pollution  → minimizza MSE
                    y_clean=y_clean,
                    lambda_clean=1.0,
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
                f.write(f"lambda_clean=1.0\n")
                f.write(f"patience={patience}\n")
                f.write("========================================\n")

        with open("sa_selective_backdoor_with_defense_results.txt", "a") as f:
            f.write(f"Round {server_round}: best_mse_poison={best_mse_poison:.6f}, best_mse_clean={best_mse_clean:.6f}, global_mse={best_mse:.6f}\n")

        return best_delta

    def aggregate_train(self, server_round: int, replies: Iterable[Message]):
        """Questa funzione è il punto in cui tutto viene integrato nella strategia Flower."""
        replies_list = list(replies)

        if not replies_list:
            return super().aggregate_train(server_round, replies_list)

        if self._global_sd is None:
            log(logging.WARNING, f"[ROUND {server_round}] Global state missing: fallback on FedAvg.")
            return super().aggregate_train(server_round, replies_list)

        log(logging.INFO, f"=== [ROUND {server_round}] DISTANCE-BASED DEFENSE + SELECTIVE SA ATTACK ===")

        # Estrazione round data: informazioni utili per eseguire il SA
        client_ids, local_sds, client_updates_flat, num_examples = self._extract_round_data(replies_list)

        # --- SA attack sul client malevolo, se presente ---
        if self.malicious_id in client_ids:
            mal_idx = client_ids.index(self.malicious_id)
            base_legit_update = client_updates_flat[mal_idx].copy()

            best_delta = self._run_simulated_annealing_selective_backdoor(
                server_round=server_round,
                base_legit_update=base_legit_update,
                all_updates_flat=client_updates_flat,
                malicious_idx=mal_idx,
                local_sds=local_sds,
                num_examples=num_examples,
            )

            # Sostituzione dell'update malevolo
            poisoned_update = base_legit_update + best_delta

            # Salviamo la storia degli update per i round successivi
            self.adv_update_history.append(poisoned_update.copy())

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
        # Dopo che il SA ha finito, facciamo girare la difesa reale del round
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

    # --- STRATEGIE CHE IMPLEMENTANO LA DIFESA E STIMANO IL K_max USABILE DALL'ATTACCANTE AD OGNI ROUND ---
    kmax_target_id: int = int(context.run_config.get("kmax-target-id", 3))
    noise_seed_base: int = int(context.run_config.get("noise-seed-base", 1337))
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

    # --- STRATEGIA CHE IMPLEMENTA LA DIFESA E SIMULA L'ATTACCANTE CHE USA SIMULATED ANNEALING PER TROVARE LA PERTURBAZIONE IDEALE ---
    #strategy = DistanceBasedDefenseStrategyWithSA(
    #    fraction_evaluate=fraction_evaluate,
    #    initial_global_sd=global_model.state_dict(),
    #    malicious_id=3,
    #    sa_T0=1.0,
    #    sa_Tmin=1e-3,
    #    sa_alpha=0.95,
    #    sa_L=30,
    #    sa_step_radius=0.1,
    #    sa_step_fraction=0.1,
    #    sa_step_min_fraction=0.001,
    #    sa_reject_penalty=1e6,
    #    sa_history_weight=0.05,
    #    sa_history_enabled=True,
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

