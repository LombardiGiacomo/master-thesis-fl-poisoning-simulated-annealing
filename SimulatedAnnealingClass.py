from copy import deepcopy
from torch.utils.data import DataLoader, ConcatDataset
from LSTMfederated.task import load_data, Net, test
import pandas as pd

def build_centralized_validation_loader(batch_size: int = 128) -> DataLoader:
    """Costruisce un validation loader centralizzato unendo i test set dei client."""
    datasets = []
    for pid in range(5):
        if pid != 3:
            _, testloader = load_data(partition_id=pid, batch_size=batch_size)
            datasets.append(testloader.dataset)
    merged = ConcatDataset(datasets)
    return DataLoader(merged, batch_size=batch_size, shuffle=False, drop_last=False)

def state_dict_to_update_flat(local_sd, global_sd) -> np.ndarray:
    """Flatten di delta = w_local - w_global.
        local_sd = pesi locali del client
        global_sd = pesi globali del server
    """
    parts = []
    #for k in local_sd.keys():
    #    if torch.is_floating_point(local_sd[k]):
    #        delta = (local_sd[k].detach().cpu() - global_sd[k].detach().cpu()).numpy().ravel()  # Costruisce l'update
    #        parts.append(delta.astype(np.float64))
    #return np.concatenate(parts)
    for i, v_local in local_sd.items():
        if torch.is_floating_point(v_local):
            delta = (v_local.detach().cpu() - global_sd[i].detatch().cpu()).numpy().ravel()
            parts.append(delta.astype(np.float64))
    return np.concatenate(parts)
    
def local_sd_from_flat_update(flat_update: np.ndarray, reference_local_sd, global_sd) -> dict:
    """Costruisce un dizionario di pesi locali a partire da un update vettoriale (w_local = w_global + delta) """
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
    """ 
        Costruisce e restituisce un modello aggregato simulando FedAvg.
        Ci serve perche durante SA vogliamo poi valutare il peggioramento dell'MSE.
    """
    total_weight = float(sum(weights))     # Denominatore totale della formula FedAvg (N)
    aggregated_model = {}
    #for k in state_dicts[0].keys():
    #    if torch.is_floating_point(state_dicts[0][k]):
    #        acc = torch.zeros_like(state_dicts[0][k].detach().cpu())    # Accumulatore inizializzato a zero
    #        for sd, w in zip(state_dicts, weights):
    #            acc += sd[k].detach().cpu() * (float(w) / total)
    #        aggregated_model[k] = acc
    #    else:
    #        aggregated_model[k] = state_dicts[0][k].detach().cpu().clone()
    for k in state_dicts[0].keys():
        if torch.is_floating_point(state_dicts[0][k]):
            aggregated_model[k] = torch.zeros_like(state_dicts[0][k].detach().cpu())    # Accumulatore inizializzato a zero
            for sd, w in zip(state_dicts, weights):
                aggregated_model[k] += sd[k].detach().cpu() * (w / total_weight)
        else:
            aggregated_model[k] = state_dicts[0][k].detach().cpu().clone()
    return aggregated_model

def evaluate_model_mse(state_dict: dict, valloader: DataLoader, device: torch.device) -> float:
    """Ricostruisce il modello globale simulato e ne calcola l'MSE."""
    model = Net()
    model.load_state_dict(state_dict, strict=True)
    mse, _ = test(model, valloader, device)
    return float(mse)

def simulate_distance_defense_acceptance(
    client_updates_flat: np.ndarray,
    malicious_id: int,
) -> tuple[bool, np.ndarray, float, float, float]:
    """Replica la difesa Distance-based.
       Restituisce True se l'attaccante è accettato dalla difesa; False altrimenti.
    """
    centroid = np.mean(client_updates_flat, axis=0)
    distances = np.linalg.norm(client_updates_flat - centroid, axis=1)

    mean_dist = float(np.mean(distances))
    std_dist = float(np.std(distances))
    threshold = mean_dist + (1.0 * std_dist)

    accepted = distances[malicious_id] <= threshold
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
        self.val_loader = build_centralized_validation_loader(batch_size=sa_val_batch_size)

        self.adv_update_history: list[np.ndarray] = []      # Memoria degli update malevoli usati ai round precedenti
        self.adv_delta_history: list[np.ndarray] = []       # Memoria delle perturbazioni trovate ai round precedenti

    def _extract_round_data(self, replies_list: list[Message]):
        """Estrazione dei dati del round. 
            Estrae gli id dei client, i modelli locali come state_dict, gli update flatten, pesi per FedAvg simulato.
        """
        client_ids = []     # gli ID dei client
        local_sds = []      # i modelli locali dei client
        updates_flat = []   # gli update flatten
        num_examples = []   # i pesi per FedAvg, inviati dai client nelle metriche

        for msg in replies_list:
            local_sd = msg.content["arrays"].to_torch_state_dict()
            pid = int(msg.content["metrics"]["partition-id"])

            n_i = float(msg.content["metrics"].get("num-examples", 1.0))

            client_ids.append(pid)
            local_sds.append(local_sd)
            updates_flat.append(state_dict_to_update_flat(local_sd, self._global_sd))
            num_examples.append(n_i)

        return client_ids, local_sds, np.array(updates_flat, dtype=np.float64), num_examples

    def _energy(
        self,
        candidate_delta: np.ndarray,    # la soluzione candidata del SA
        base_legit_update: np.ndarray,  # update legittimo del client malevolo
        all_updates_flat: np.ndarray,   # tutti gli update del round
        malicious_id: int,             # posizione del malevolo negli update
        local_sds: list[dict],          # modelli locali
        num_examples: list[float],      # pesi FedAvg
        base_mse: float,                # MSE del round senza perturbazione
        head_mask: np.ndarray,          
    ) -> tuple[float, bool, float]:
        """
        Energia:
          - se rifiutato dalla difesa -> grossa penalità
          - se accettato dalla difesa -> -relative_mse + lambda * distanza_storica_relativa
        """
        # Costruisco l'update malevolo candidato e lo inserisco in Xcand al posto dell'update legittimo del client malevolo
        candidate_update = base_legit_update + candidate_delta
        Xcand = all_updates_flat.copy()
        Xcand[malicious_id] = candidate_update

        # Verifica se il candidato viene accettato o scartato dalla difesa: se viene scartato l'energia è enorme
        accepted, _, _, _, _ = simulate_distance_defense_acceptance(Xcand, malicious_id)
        if not accepted:
            log(logging.INFO, f"Candidate delta discarded from defense")
            return self.sa_reject_penalty, False, float("nan")

        # Ricostruzione del modello locale (state_dict locali) con il candidato malevolo
        poisoned_local_sd = local_sd_from_flat_update(
            candidate_update,
            reference_local_sd=local_sds[malicious_id],
            global_sd=self._global_sd,
        )

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
            return self.sa_reject_penalty, False, float("nan")

        agg_sd = weighted_average_state_dicts(accepted_sds, accepted_w)
        mse = evaluate_model_mse(agg_sd, self.val_loader, self.device)

        rel_mse = mse / (base_mse + 1e-12)
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
        #log_rel_mse = np.log((mse + 1e-12) / (base_mse + 1e-12))

        # Penalità storica sulla sola head
        #hist_penalty = 0.0
        #if self.sa_history_enabled and len(self.adv_update_history) > 0:
        #    hist_centroid = np.mean(np.stack(self.adv_update_history, axis=0), axis=0)
#
        #    candidate_head = candidate_update * head_mask
        #    hist_head = hist_centroid * head_mask
#
        #    hist_norm = np.linalg.norm(hist_head)
        #    if hist_norm > 1e-12:
        #        hist_penalty = np.linalg.norm(candidate_head - hist_head) / (hist_norm + 1e-12)
        #    else:
        #        hist_penalty = 0.0

        #energy = -log_rel_mse + self.sa_history_weight * hist_penalty
        
        energy = -rel_mse

        #log(logging.INFO,
        #    f"log_rel_mse={log_rel_mse:.6f} hist_penalty={hist_penalty:.6f} "
        #    f"weighted_hist={self.sa_history_weight * hist_penalty:.6f} "
        #    f"energy={energy:.6f} "
        #    f"mse={mse:.6f}"
        #)

        log(logging.INFO, f"rel_mse={rel_mse:.6f} energy={energy:.6f} ")

        return float(energy), True, float(mse)
    
    def _energy_backdoor_attack(
        self,
        candidate_delta: np.ndarray,    # la soluzione candidata del SA
        base_legit_update: np.ndarray,  # update legittimo del client malevolo
        all_updates_flat: np.ndarray,   # tutti gli update del round
        malicious_id: int,             # posizione del malevolo negli update
        local_sds: list[dict],          # modelli locali
        num_examples: list[float],      # pesi FedAvg
        base_mse: float,                # MSE del round senza perturbazione
        x_seq: np.ndarray,              # sequenza che vogliamo avvelenare
        y_target: float                  # valore desiderato PM2.5 (25° timestep)
    ) -> tuple[float, bool, float]:

        # Costruisco l'update malevolo candidato e lo inserisco in Xcand al posto dell'update legittimo del client malevolo
        candidate_update = base_legit_update + candidate_delta
        Xcand = all_updates_flat.copy()
        Xcand[malicious_id] = candidate_update

        # Verifica se il candidato viene accettato o scartato dalla difesa: se viene scartato l'energia è enorme
        #accepted, _, _, _, _ = simulate_distance_defense_acceptance(Xcand, malicious_id)
        #if not accepted:
        #    log(logging.INFO, f"Candidate delta discarded from defense")
        #    return self.sa_reject_penalty, False, float("nan")

        # Ricostruzione del modello locale (state_dict locali) con il candidato malevolo
        poisoned_local_sd = local_sd_from_flat_update(
            candidate_update,
            reference_local_sd=local_sds[malicious_id],
            global_sd=self._global_sd,
        )

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
        malicious_id: int,             # posizione del malevolo negli update
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
        Xcand[malicious_id] = candidate_update

        # Verifica se il candidato viene accettato o scartato dalla difesa: se viene scartato l'energia è enorme
        accepted, _, _, _, _ = simulate_distance_defense_acceptance(Xcand, malicious_id)
        if not accepted:
            log(logging.INFO, f"Candidate delta discarded from defense")
            return self.sa_reject_penalty, False, float("nan"), float("nan"), float("nan")

        # Ricostruzione del modello locale (state_dict locali) con il candidato malevolo
        poisoned_local_sd = local_sd_from_flat_update(
            candidate_update,
            reference_local_sd=local_sds[malicious_id],
            global_sd=self._global_sd,
        )

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
        malicious_id: int,
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
        Xbase[malicious_id] = base_legit_update
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
        reference_sd = local_sds[malicious_id]
        
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
            malicious_id=malicious_id,
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
        freeze_stages = 2

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
                    malicious_id=malicious_id,
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
                f"best_energy={best_energy:.6f} best_mse={best_mse:.6f}, curr_energy={curr_energy:.6f}")

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
        malicious_id: int,
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
        Xbase[malicious_id] = base_legit_update
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
        reference_sd = local_sds[malicious_id]
        
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
            malicious_id=malicious_id,
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
                    malicious_id=malicious_id,
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
        malicious_id: int,
        local_sds: list[dict],
        num_examples: list[float],
    ) -> np.ndarray:

        # Calcolo l'MSE del round senza perturbazione
        Xbase = all_updates_flat.copy()
        Xbase[malicious_id] = base_legit_update
        centroid = np.mean(Xbase, axis=0)
        distances = np.linalg.norm(Xbase - centroid, axis=1)
        threshold = float(np.mean(distances) + np.std(distances))
        accepted_mask = [d <= threshold for d in distances]

        base_sds = [sd for sd, ok in zip(local_sds, accepted_mask) if ok]
        base_w = [w for w, ok in zip(num_examples, accepted_mask) if ok]
        base_agg_sd = weighted_average_state_dicts(base_sds, base_w)
        true_base_mse = evaluate_model_mse(base_agg_sd, self.val_loader, self.device)

        # --- Setup sequenze (una volta sola, fuori dal loop SA) ---
        x_poison, y_poison = load_sequences("./SequencesBackdoorSA/high_pollution_sequences.csv", "./SequencesBackdoorSA/high_pollution_targets.csv")
        x_clean,  y_clean  = load_sequences("./SequencesBackdoorSA/low_pollution_sequences.csv",  "./SequencesBackdoorSA/low_pollution_targets.csv")
        #x_poison, y_poison = load_sequences("./SequencesBackdoorSA/low_pollution_sequences.csv",  "./SequencesBackdoorSA/low_pollution_targets.csv")
        #x_clean,  y_clean  = load_sequences("./SequencesBackdoorSA/high_pollution_sequences.csv", "./SequencesBackdoorSA/high_pollution_targets.csv")
        lambda_clean = 0.5
        # Soluzione iniziale delta = 0, cioè si parte dall'update legittimo.
        # Calcoliamo quindi l'energia iniziale, che sarà l'energia corrente, e il valore di mse iniziale, che sarà quello corrente.
        # Ci manteniamo soluzione corrente (curr_delta) e migliore soluzione trovata finora (best_delta): questo perche in SA la soluzione corrente non è necessariamente
        # la migliore soluzione visitata. Infatti, SA può accettare soluzioni peggiori
        d = base_legit_update.shape[0]
        delta = np.zeros(d, dtype=np.float64)

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

        curr_energy, _, curr_mse_poison, curr_mse_clean, curr_mse = self._energy_selective_backdoor_attack(
            candidate_delta=delta,
            base_legit_update=base_legit_update,
            all_updates_flat=all_updates_flat,
            malicious_id=malicious_id,
            local_sds=local_sds,
            num_examples=num_examples,
            base_mse=true_base_mse,
            x_poison=x_poison,   # high pollution → massimizza MSE
            y_poison=y_poison,
            x_clean=x_clean,     # low pollution  → minimizza MSE
            y_clean=y_clean,
            lambda_clean=lambda_clean,
        )
        best_delta = delta.copy()
        best_energy = curr_energy
        best_mse_poison = curr_mse_poison
        best_mse_clean = curr_mse_clean
        best_mse = curr_mse

        init_mse_poison = curr_mse_poison
        init_mse_clean = curr_mse_clean
        init_global_mse = curr_mse

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
                    malicious_id=malicious_id,
                    local_sds=local_sds,
                    num_examples=num_examples,
                    base_mse=true_base_mse,
                    x_poison=x_poison,   # high pollution → massimizza MSE
                    y_poison=y_poison,
                    x_clean=x_clean,     # low pollution  → minimizza MSE
                    y_clean=y_clean,
                    lambda_clean=lambda_clean,
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
                f.write(f"patience={patience}\n")
                f.write("========================================\n")

        with open("sa_selective_backdoor_with_defense_results.txt", "a") as f:
            f.write(f"Round {server_round}: init_mse_poison={init_mse_poison:.6f}, init_mse_clean={init_mse_clean:.6f}, init_global_mse={init_global_mse:.6f} -> "
                    f"best_mse_poison={best_mse_poison:.6f}, best_mse_clean={best_mse_clean:.6f}, global_mse={best_mse:.6f}\n")

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

            best_delta = self._run_simulated_annealing(
                server_round=server_round,
                base_legit_update=base_legit_update,
                all_updates_flat=client_updates_flat,
                malicious_id=mal_idx,
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
