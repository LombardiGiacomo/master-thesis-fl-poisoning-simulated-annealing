import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from LSTMfederated.task import Net, load_data, load_data_malicious
from LSTMfederated.task import test as test_fn
from LSTMfederated.task import train as train_fn
from LSTMfederated.task import train_malicious as train_fn_malicious

from flwr.common import ConfigRecord

def _make_seed(seed_base: int, partition_id: int) -> int:
    # Seed deterministico, uguale per server e client, non dipende dal round
    return int(seed_base) + int(partition_id)

# Flower ClientApp
app = ClientApp()

@app.train()
def train(msg: Message, context: Context):  # Funzione chiamata dal server ad ogni round
    """Train the model on local data."""

    # Load the model and initialize it with the received weights
    # 1) Ricezione del modello globale  --> il modello globale diventa modello locale
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())  # Il server manda i pesi globali dentro msg.content["arrays"]
                                                                        # e li carichiamo dentro il modello locale model

    global_state = msg.content["arrays"].to_torch_state_dict()  # Utile per attacco sign-flip

    # 2) Device
    device = torch.device("cpu")
    model.to(device)

    # 3) Load the data --> Caricamento dei dati locali
    partition_id = context.node_config["partition-id"]

    raw_ids = context.run_config.get("malicious-ids", "")
    malicious_ids = set(int(x) for x in raw_ids.split(",") if x != "")
    is_malicious = partition_id in malicious_ids

    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]

    trainloader, _ = load_data(partition_id, batch_size)
    #trainloader, _ = load_data_malicious(partition_id, batch_size, is_malicious)

    # 4) Call the training function --> Training 
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],    # il learning rate viene preso dal messaggio perche è il server che decide la da config mandare ai client
        device
    )

    # ---------------------------------------------------------
    # MODEL POISONING #1: Model updates Sign-Flip Attack
    # ---------------------------------------------------------
    #if is_malicious:
    #    gamma = 4  # Possible values: 1, 1.5, 2, 4, 8
#
    #    with torch.no_grad():
    #        local_state = model.state_dict()
    #        poisoned = {}
    #        for k, v_local in local_state.items():  # v_local is the weight tensor of layer k after local training
    #            v_global = global_state[k]          # v_global is the weight tensor of layer k received from the server
#
    #            # Modify only float tensors (weights/bias); leave unaltered any buffer/int
    #            if torch.is_floating_point(v_local):
    #                # Normally: v_local = v_global + Delta, with Delta = v_local - v_global
    #                # Instead of adding the update to the global weights, the weights to be sent back to the server are obtained by subtracting the update to the global weights:
    #                # poisoned = v_global - Delta
    #                # Gamma allows to boost the attack
    #                poisoned[k] = v_global - gamma * (v_local - v_global)
    #            else:
    #                poisoned[k] = v_local
#
    #        model.load_state_dict(poisoned)     # Load the poisoned model replacing v_local
#
    #    print(f"[!!! ATTACK (Client {partition_id})!!!] SIGN FLIP on the update (gamma={gamma})")
    # -----------------------------------------------------------

    # ---------------------------------------------------------
    # MODEL POISONING #3.1: Naive Random Noise Attack: add random noise to the local model
    # ---------------------------------------------------------
    #if is_malicious:
    #    with torch.no_grad():
    #        local_state = model.state_dict()
    #        poisoned = {}
#
    #        for i, v_local in local_state.items():
    #            # Modify only float tensors (weights and bias)
    #            if torch.is_floating_point(v_local):
    #                noise = torch.randn_like(v_local)       # Generate gaussian noise with the same shape as v_local tensor
    #                poisoned[i] = v_local + noise           # Add noise to the local model parameters
    #            else:
    #                poisoned[i] = v_local
    #        model.load_state_dict(poisoned)             # Load the poisoned model replacing v_local
#
    #        print(f"[!!! ATTACK (Client {partition_id})!!!] Basic RANDOM NOISE ADDITION to local model")
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # MODEL POISONING #3.2: Systematic Noise Attack
    #  (Added noise scaling with parameter k and seed for random number generator
    #  The seed allows to implement a selective noise injection at each round)
    # ---------------------------------------------------------
    if is_malicious:
        k = context.run_config.get("k-noise", 0.0)  # Parameter that controls the strength of the attack

        # Use a seed so that at every round the generated noise is the same
        seed = int(context.run_config.get("noise-seed-base", 1337)) + int(partition_id)
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed))

        with torch.no_grad():
            local_state = model.state_dict()
            poisoned = {}

            for i, v_local in local_state.items():
                # Modify only float tensors (weights and bias)
                if torch.is_floating_point(v_local):
                    noise = torch.randn(v_local.shape, generator=g, dtype=v_local.dtype, device=v_local.device)     # Generate gaussian noise with the same shape as v_local tensor  
                    poisoned[i] = v_local + k * noise       # Add scaled noise to the local model parameters
                else:
                    poisoned[i] = v_local
            model.load_state_dict(poisoned)             # Load the poisoned model replacing v_local

            print(f"[!!! ATTACK (Client {partition_id})!!!] Scaled SYSTEMATIC NOISE ADDITION to local model")
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # MODEL POISONING #3.3: Adaptive Noise Addition Attack
    # ---------------------------------------------------------
    #if is_malicious:
    #    k = context.run_config.get("k-noise", 0.0)  # Parameter that controls the strength of the attack
#
    #    # Use a seed so that at every round the generated noise is the same
    #    seed = int(context.run_config.get("noise-seed-base", 1337)) + int(partition_id)
    #    g = torch.Generator(device="cpu")
    #    g.manual_seed(int(seed))
#
    #    if k > 0.0:
    #        with torch.no_grad():
    #            local_state = model.state_dict()
    #            poisoned = {}
    #    
    #            # 1. Compute the norm of legitimate update: dimension of the "real" update, if the client were honest
    #            delta_norm_sq = 0.0
    #            for i, v_local in local_state.items():
    #                v_global = global_state[i]
    #                if torch.is_floating_point(v_local):
    #                    delta = v_local - v_global
    #                    delta_norm_sq += torch.sum(delta**2)
    #            delta_norm = torch.sqrt(delta_norm_sq)  # Norm of legitimate update
#
    #            # 2. Generate noise for each layer and compute its norm
    #            noise_norm_sq = 0.0
    #            noise_dict = {}
    #            for i, v_local in local_state.items():
    #                if torch.is_floating_point(v_local):
    #                    noise = torch.randn(v_local.shape, generator=g, dtype=v_local.dtype, device=v_local.device) # Generate gaussian noise with the same shape as v_local tensor  
    #                    noise_dict[i] = noise
    #                    noise_norm_sq += torch.sum(noise * noise)
    #                else:
    #                    noise_dict[i] = None
    #            noise_norm = torch.sqrt(noise_norm_sq)
    #    
    #            # 3. Scale the noise to control the norm 
    #            scale = (k * delta_norm) / (noise_norm + 1e-12) # Scale is necessary so that ||scale*noise||=k*||delta||
    #            
    #            # 4. Add scaled noise to the local model
    #            for i, v_local in local_state.items():
    #                if torch.is_floating_point(v_local):
    #                    poisoned[i] = v_local + scale * noise_dict[i]
    #                else:
    #                    poisoned[i] = v_local
    #    
    #            model.load_state_dict(poisoned)
    #            print(f"[!!! ATTACK (Client {partition_id})!!!] ADAPTIVE NOISE ADDITION with normalized noise (magnitude k={k})")
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # MODEL POISONING #3.3.1: Adaptive Noise Attack (Head Only)
    # ---------------------------------------------------------
    #if is_malicious:
    #    k = context.run_config.get("k-noise", 0.0)  # Parameter that controls the strength of the attack
#
    #    seed = int(context.run_config.get("noise-seed-base", 1337)) + int(partition_id)
    #    g = torch.Generator(device="cpu")
    #    g.manual_seed(int(seed))
#
    #    if k > 0.0:
    #        with torch.no_grad():
    #            local_state = model.state_dict()
    #            poisoned = {}
#
    #            # 1. Compute the norm of legitimate head layer update
    #            delta_norm_sq = 0.0
    #            for i, v_local in local_state.items():
    #                v_global = global_state[i]
    #                if torch.is_floating_point(v_local) and "head" in i:
    #                    delta = v_local - v_global
    #                    delta_norm_sq += torch.sum(delta ** 2)
    #            delta_norm = torch.sqrt(delta_norm_sq)  # Norm of legitimate head layer update
#
    #            # 2. Generate noise for head layer and compute its norm
    #            noise_norm_sq = 0.0
    #            noise_dict = {}
    #            for i, v_local in local_state.items():
    #                if torch.is_floating_point(v_local) and "head" in i:
    #                    noise = torch.randn(v_local.shape, generator=g, dtype=v_local.dtype, device=v_local.device) # Generate gaussian noise with the same shape as v_local tensor  
    #                    noise_dict[i] = noise
    #                    noise_norm_sq += torch.sum(noise * noise)
    #                else:
    #                    noise_dict[i] = None
    #            noise_norm = torch.sqrt(noise_norm_sq)
#
    #            # 3. Scale the noise to control the norm 
    #            scale = (k * delta_norm) / (noise_norm + 1e-12)
#
    #            # 4. Add scaled noise to the head layer of local model
    #            for i, v_local in local_state.items():
    #                if torch.is_floating_point(v_local) and "head" in i:
    #                    poisoned[i] = v_local + scale * noise_dict[i]
    #                else:
    #                    poisoned[i] = v_local
#
    #            model.load_state_dict(poisoned)
    #            print(f"[!!! ATTACK (Client {partition_id})!!!] HEAD-ONLY ADAPTIVE NOISE ADDITION with normalized noise (magnitude={k})")
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # MODEL POISONING: Pure Systematic Additive Error
    # ---------------------------------------------------------
    #if is_malicious:
    #    # k è il valore dell'errore da aggiungere o togliere
    #    k = context.run_config.get("k-noise", 0.0) 
#
    #    # Seed FISSO: decide una volta per tutte a chi aggiungere e a chi togliere --> usato per formare la maschera di +1/-1
    #    seed_base = int(context.run_config.get("noise-seed-base", 1337))
    #    fixed_seed = int(seed_base) + int(partition_id) 
#
    #    g = torch.Generator(device="cpu")
    #    g.manual_seed(fixed_seed)
#
    #    with torch.no_grad():
    #        local_state = model.state_dict()
    #        poisoned_state = {}
#
    #        # Iteriamo su ogni singolo layer/parametro della rete
    #        for name, v_local in local_state.items():
    #            v_global = global_state[name]
    #            
    #            # Attacchiamo solo i pesi addestrabili (tensori float)
    #            if torch.is_floating_point(v_local):
    #                
    #                # --- FASE 1: L'update legittimo (il nostro vettore 'x') ---
    #                delta_true = v_local - v_global
#
    #                # --- FASE 2: Generazione della maschera sistematica di +1/-1 ---
    #                mask_cpu = torch.randint(0, 2, tuple(delta_true.shape), generator=g, dtype=torch.float32)
    #                mask = mask_cpu.to(v_local.device)
    #                mask = mask * 2.0 - 1.0  
    #                
    #                # --- FASE 3: Generazione della perturbazione fissa ---
    #                # L'errore è semplicemente il nostro k fisso applicato ai segni + e -
    #                perturbation = k * mask
    #                
    #                # --- FASE 4: Costruzione dell'Update Malevolo ---
    #                # delta_malicious = x + delta
    #                delta_malicious = delta_true + perturbation
    #                
    #                # Ricostruiamo i pesi finali da inviare al server
    #                poisoned_state[name] = v_global + delta_malicious
    #            else:
    #                # Se non è un peso float (es. num_batches_tracked), lo lasciamo intatto
    #                poisoned_state[name] = v_local
    #    
    #        # Sostituiamo i pesi onesti del modello con quelli avvelenati
    #        model.load_state_dict(poisoned_state)
    #        
    #        # Log per confermare l'esecuzione durante la simulazione
    #        print(f"[ATTACCO] Client {partition_id}: Attacco Sistematico Ortogonale iniettato (k={k})")
    # --------------------------------------------------------------------

    # ---------------------------------------------------------
    # MODEL POISONING: Pure Systematic Additive Error --> solo sulla head
    # ---------------------------------------------------------
    #if is_malicious:
    #    k = context.run_config.get("k-noise", 0.0)
#
    #    seed_base = int(context.run_config.get("noise-seed-base", 1337))
    #    fixed_seed = int(seed_base) + int(partition_id)
#
    #    g = torch.Generator(device="cpu")
    #    g.manual_seed(fixed_seed)
#
    #    with torch.no_grad():
    #        local_state = model.state_dict()
    #        poisoned_state = {}
#
    #        for name, v_local in local_state.items():
    #            v_global = global_state[name]
#
    #            # Attacchiamo SOLO i pesi float della head
    #            if torch.is_floating_point(v_local) and "head" in name:
    #                # Update legittimo
    #                delta_true = v_local - v_global
#
    #                # Maschera sistematica di +1/-1
    #                mask_cpu = torch.randint(
    #                    0,
    #                    2,
    #                    tuple(delta_true.shape),
    #                    generator=g,
    #                    dtype=torch.float32,
    #                )
    #                mask = mask_cpu.to(v_local.device).type_as(v_local)
    #                mask = mask * 2.0 - 1.0
#
    #                # Perturbazione fissa
    #                perturbation = k * mask
#
    #                # Update malevolo
    #                delta_malicious = delta_true + perturbation
#
    #                # Ricostruzione pesi finali
    #                poisoned_state[name] = v_global + delta_malicious
    #            else:
    #                poisoned_state[name] = v_local
#
    #        model.load_state_dict(poisoned_state)
    #        print(
    #            f"[ATTACCO] Client {partition_id}: "
    #            f"Head-only Systematic Additive Error iniettato (k={k})"
    #        )
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # MODEL POISONING: Pure Systematic Additive Error (aggiunto a metà componenti e tolto dall'altra metà)
    # con norma dell'update malevolo riscalata per essere uguale a quella dell'update onesto
    # ---------------------------------------------------------
    #if is_malicious:
    #    k_bias = float(context.run_config.get("k-noise", 0.0))
#
    #    if k_bias > 0.0:
    #        with torch.no_grad():
    #            local_state = model.state_dict()
    #            poisoned = {}
#
    #            # 1) calcola l'update benigno Delta = w_local - w_global
    #            delta_dict = {}
    #            delta_norm_sq = 0.0
    #            total_numel = 0
#
    #            for name, v_local in local_state.items():
    #                v_global = global_state[name]
    #                if torch.is_floating_point(v_local):
    #                    delta = v_local - v_global
    #                    delta_dict[name] = delta
    #                    delta_norm_sq += torch.sum(delta * delta)
    #                    total_numel += delta.numel()
    #                else:
    #                    delta_dict[name] = None
#
    #            delta_norm = torch.sqrt(delta_norm_sq) + 1e-12
#
    #            # 2) costruisci una direzione di bias fissa nel tempo: metà +1, metà -1
    #            #    la generiamo in modo deterministico usando il seed del client
    #            seed_base = int(context.run_config.get("noise-seed-base", 1337))
    #            seed = _make_seed(seed_base, partition_id)
#
    #            g = torch.Generator(device="cpu")
    #            g.manual_seed(int(seed))
#
    #            bias_dict = {}
    #            bias_norm_sq = 0.0
#
    #            for name, v_local in local_state.items():
    #                if torch.is_floating_point(v_local):
    #                    numel = v_local.numel()
#
    #                    # metà +1, metà -1, poi shuffle deterministico
    #                    flat = torch.ones(numel, dtype=torch.float32)
    #                    flat[numel // 2 :] = -1.0
    #                    perm = torch.randperm(numel, generator=g)
    #                    flat = flat[perm]
#
    #                    bias = flat.reshape(v_local.shape).to(v_local.device).type_as(v_local)
    #                    bias_dict[name] = bias
    #                    bias_norm_sq += torch.sum(bias * bias)
    #                else:
    #                    bias_dict[name] = None
#
    #            bias_norm = torch.sqrt(bias_norm_sq) + 1e-12
#
    #            # 3) aggiungi bias sistematico proporzionale a ||Delta||
    #            attacked_delta_dict = {}
    #            attacked_delta_norm_sq = 0.0
#
    #            for name, v_local in local_state.items():
    #                if torch.is_floating_point(v_local):
    #                    delta = delta_dict[name]
    #                    bias = bias_dict[name]
#
    #                    attacked_delta = delta + k_bias * delta_norm * (bias / bias_norm)
    #                    attacked_delta_dict[name] = attacked_delta
    #                    attacked_delta_norm_sq += torch.sum(attacked_delta * attacked_delta)
    #                else:
    #                    attacked_delta_dict[name] = None
#
    #            attacked_delta_norm = torch.sqrt(attacked_delta_norm_sq) + 1e-12
#
    #            # 4) rescaling: riporta la norma finale a ||Delta||
    #            rescale = delta_norm / attacked_delta_norm
#
    #            for name, v_local in local_state.items():
    #                v_global = global_state[name]
    #                if torch.is_floating_point(v_local):
    #                    poisoned[name] = v_global + rescale * attacked_delta_dict[name]
    #                else:
    #                    poisoned[name] = v_local
#
    #            model.load_state_dict(poisoned)
#
    #        print(
    #            f"[!!! ATTACK !!!] Client {partition_id}: systematic bias attack "
    #            f"with norm preservation (k_bias={k_bias})"
    #        )
    # =====================================================================

    # =====================================================================
    # MODEL POISONING: Attacco Sistematico LAYER-SPECIFIC (Cecchino)
    # Lasciamo intatto l'estrattore, distruggiamo solo il classificatore finale
    # =====================================================================
    #if is_malicious:
    #    # Possiamo alzare il k! Dato che colpiamo pochissimi pesi, la norma non esploderà.
    #    k_sys = context.run_config.get("k-noise", 0.0)  
#
    #    fixed_seed = 42 + int(partition_id)
    #    g = torch.Generator(device="cpu")
    #    g.manual_seed(fixed_seed)
#
    #   with torch.no_grad():
    #       local_state = model.state_dict()
    #       poisoned_state = {}
#
    #        somma_quadrati_benigna = 0.0
    #        somma_quadrati_maligna = 0.0
#
    #        for name, v_local in local_state.items():
    #            v_global = global_state[name]
    #            
    #            if torch.is_floating_point(v_local):
    #                delta_true = v_local - v_global
#
    #                # IL TRUCCO DEL CECCHINO: Colpiamo solo i layer finali!
    #                if "head" in name:
    #                    
    #                    mask_cpu = torch.randint(0, 2, tuple(delta_true.shape), generator=g, dtype=torch.float32)
    #                    mask = mask_cpu.to(v_local.device)
    #                    mask = mask * 2.0 - 1.0  
    #                    
    #                    # Iniezione massiccia ma concentrata
    #                    #errore_sistematico = k_sys * mask
    #                    errore_sistematico = torch.full_like(delta_true, k_sys)
    #                    delta_malicious = delta_true + errore_sistematico
#
    #                   # Accumulo i quadrati per il calcolo delle norme
    #                   somma_quadrati_benigna += torch.sum(delta_true ** 2).item()
    #                   somma_quadrati_maligna += torch.sum(delta_malicious ** 2).item()
    #                   
    #                   poisoned_state[name] = v_global + delta_malicious
    #                   # print(f"  -> Avvelenato layer critico: {name}") # Scommenta per debug
    #               else:
    #                   # I layer iniziali rimangono PURAMENTE ONESTI (Norma intatta)
    #                   poisoned_state[name] = v_local
    #           else:
    #               poisoned_state[name] = v_local
    #   
    #       # Calcolo delle norme degli update benigno e maligno
    #       norma_benigna = somma_quadrati_benigna ** 0.5
    #       norma_maligna = somma_quadrati_maligna ** 0.5
#
    #       model.load_state_dict(poisoned_state)
    #       print(f"[ATTACCO] Client {partition_id}: Avvelenato solo il layer finale con k={k_sys}")
    #       print(f"   -> Norma Update Onesto:   {norma_benigna}")
    #       print(f"   -> Norma Update Avvelenato: {norma_maligna}")
    # ------------------------------------------------------------------------------

    # 5) Construct and return reply Message --> Ritorno al server
    model_record = ArrayRecord(model.state_dict())  # Si trasformano i pesi PyTorch in un oggetto Flower
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),   # num-examples serve a FedAvg per pesare correttaemnte i client
        "partition-id": partition_id,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)   # Questo messaggio torna al server. Dentro ci sono pesi aggiornati, metriche, numero esempi


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    raw_ids = context.run_config.get("malicious-ids", "")
    malicious_ids = set(int(x) for x in raw_ids.split(",") if x != "")
    is_malicious = partition_id in malicious_ids
    batch_size = context.run_config["batch-size"]
    _, valloader = load_data(partition_id, batch_size)

    # Call the evaluation function
    eval_mse, eval_mae = test_fn(model, valloader, device)

    metrics = {
        "eval_mse": eval_mse,
        "eval_mae": eval_mae,
        "num-examples": len(valloader.dataset),
    }
    
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    print(f"[Client {partition_id}] eval_mse={eval_mse:.6f} eval_mae={eval_mae:.6f} n={len(valloader.dataset)}")
    return Message(content=content, reply_to=msg)
