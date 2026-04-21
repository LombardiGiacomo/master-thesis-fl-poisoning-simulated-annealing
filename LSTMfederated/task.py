import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Tuple

TARGET_COL = "PM2.5"
FEATURE_COLS = [
    "PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
    "TEMP", "PRES", "DEWP", "RAIN", "WSPM"
]

SEQ_LEN = 24    # Considero le ultime 24 ore
HORIZON = 1     # Predico la prossima ora

CLIENT_FILES = {
    0: ("aotizhongxin/aotizhongxin_train_scaled.csv",
        "aotizhongxin/aotizhongxin_test_scaled.csv"),
    1: ("changping/changping_train_scaled.csv",
        "changping/changping_test_scaled.csv"),
    2: ("dingling/dingling_train_scaled.csv",
        "dingling/dingling_test_scaled.csv"),
    3: ("dongsi/dongsi_train_scaled.csv",
        "dongsi/dongsi_test_scaled.csv"),
    4: ("guanyuan/guanyuan_train_scaled.csv",
        "guanyuan/guanyuan_test_scaled.csv")
}

# -----------------------------
# Dataset + utilities
# -----------------------------
def _make_windows(X: np.ndarray, y: np.ndarray, seq_len: int, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prende la tabella grande (T, F) e produce N finestre, ognuna con seq_len righe e F colonne
    Input:
        X: (T, F), y: (T,)
        X è la matrice delle features: matrice bidimensionale che rappresenta l'intera serie temporale storica.
            - T righe, dove ogni riga rappresenta un'ora di misurazione (es. 2013-03-01 00:00:00)
            - F colonne, dove ogni colonna è una variabile osservata. Noi abbiamo 11 colonne (FEATURE_COLS).
            - Dimensione della matrice: (T, F), dove F = len(FEATURE_COLS)
        y è il vettore target: array monodimensionale che contiene solo i valori che vogliamo predire. E' la colonna
        TARGET_COL. Dimensione di y: (T,)
    Output:
      Xs: (N, seq_len, F)
      ys: (N,)
    _make_windows trasforma la lunga sequenza temporale in un formato adatto all'addestramento: crea coppie di 
    (Sequenze Passate, Valore Futuro)
    """
    if len(X) != len(y):
        raise ValueError("X and y must have same temporal length.")

    # Assicuriamoci ci siano abbastanza dati futuri per l'etichetta target. 
    # Con horizon=1 ci fermiamo alla penultima riga per predire l'ultima.
    max_i = len(X) - horizon
    if max_i <= seq_len:
        raise ValueError(
            f"Few data to create windows: len={len(X)}, seq_len={seq_len}, horizon={horizon}."
        )

    Xs, ys = [], []
    for i in range(seq_len, max_i):     # Scorriamo lungo la serie temporale
        Xs.append(X[i - seq_len:i, :])  # Prendiamo una fetta di X che va da i - SEQ_LEN fino a i escluso. Cioè prende le 24 ore precedenti di tutte le 11 variabili
        ys.append(y[i + horizon - 1])   # Prende il valore di TARGET_COL nel futuro.
    # Date le ultime SEQ_LEN ore (da t-24 a t-1), indoviniamo la TARGET_COL all'ora t

    # Xs è un tensore 3D di forma (N, SEQ_LEN, F), con N=numero di finestre create, SEQ_LEN=lunghezza sequenza temporale,
    # F=numero di features (11).
    # ys è un vettore di forma (N,)

    return np.asarray(Xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


class SequenceDataset(Dataset):
    """
    E' un contenitore al cui interno ci sono Xw con Xw.shape=(N, 24, 11) e yw.shape=(N,).
    Prende gli array numpy creati con _make_windows e li trasforma in tensori torch pronti per i calcoli.
    Quando il DataLoader chiede un esempio (attraverso __getitem__), es. 7, lui risponde con Xw[7] e yw[7], 
    cioè una finestra di forma (24,11) e il suo target.
    """
    def __init__(self, Xw: np.ndarray, yw: np.ndarray):
        # I dati in ingresso sono array NumPy, mentre il modello LSTM richiede tensori PyTorch
        self.Xw = torch.from_numpy(Xw)  # (N, seq_len, F)
        self.yw = torch.from_numpy(yw)  # (N,)

    def __len__(self) -> int:
        return len(self.yw)

    def __getitem__(self, idx: int):
        return self.Xw[idx], self.yw[idx]


def load_data_malicious(partition_id: int, batch_size: int, is_malicious: bool):
    """
    Carica train/test per il client identificato da partition_id.
    """
    if partition_id not in CLIENT_FILES:
        raise ValueError(
            f"partition_id={partition_id} not present in CLIENT_FILES={list(CLIENT_FILES.keys())}. "
        )
    
    train_path, test_path = CLIENT_FILES[partition_id]

    df_tr = pd.read_csv(train_path)
    df_te = pd.read_csv(test_path)

    X_tr = df_tr[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_tr = df_tr[TARGET_COL].to_numpy(dtype=np.float32)

    # ---------------------------------------------------------
    # DATA POISONING #1: Label Flipping on normalized target
    # ---------------------------------------------------------
    #if is_malicious:
    #    
    #    # Since data are normalized [0,1] this operation inverts the semantic (High pollution <-> Low pollution)
    #    y_tr = 1.0 - y_tr
    #    
    #    print(f"[!!! ATTACK (Client {partition_id})!!!] LABEL FLIPPING executed on training data !")
    # The attack is implemented before _make_windows for performance reasons
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # DATA POISONING #3: Temporal block shuffle
    # ---------------------------------------------------------
    #if is_malicious:
    #    rng = np.random.default_rng(5000 + partition_id)
    #
    #    block_len = 6           # Block length 
    #    shuffle_frac = 0.7      # % of blocks participating to the shuffle
    #
    #    T = X_tr.shape[0]               # Total number of hours in the training set
    #    num_blocks = T // block_len     # Total number of blocks in the training set
    #    usable = num_blocks * block_len
    #
    #    X0 = X_tr[:usable].reshape(num_blocks, block_len, X_tr.shape[1])
    #    y0 = y_tr[:usable].reshape(num_blocks, block_len)
    #
    #    # Choose which blocks to shuffle
    #    mask_b = rng.random(num_blocks) < shuffle_frac
    #    idx_b = np.where(mask_b)[0]
    #
    #    # Swap the selected blocks
    #    perm = idx_b.copy()
    #    rng.shuffle(perm)
    #
    #    X0_sh = X0.copy()
    #    y0_sh = y0.copy()
#
    #    # The same swap is done for both features and target
    #    X0_sh[idx_b] = X0[perm]
    #    y0_sh[idx_b] = y0[perm]
    #
    #    X_tr[:usable] = X0_sh.reshape(usable, X_tr.shape[1])
    #    y_tr[:usable] = y0_sh.reshape(usable)
    #
    #    print(
    #        f"[!!! ATTACK (Client {partition_id})!!!] TEMPORAL BLOCK SHUFFLING executed on training data ! "
    #        f"\nBlock-shuffle on {len(idx_b)}/{num_blocks} blocks "
    #        f"\nAttack parameters: block_len={block_len}, shuffle_frac={shuffle_frac}"
    #    )
    # ---------------------------------------------------------

    X_te = df_te[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_te = df_te[TARGET_COL].to_numpy(dtype=np.float32)

    X_tr_w, y_tr_w = _make_windows(X_tr, y_tr, SEQ_LEN, HORIZON)
    X_te_w, y_te_w = _make_windows(X_te, y_te, SEQ_LEN, HORIZON)

    # ---------------------------------------------------------
    # DATA POISONING #2: Feature bias
    # ---------------------------------------------------------
    #if is_malicious:
    #    rng = np.random.default_rng(2000 + partition_id)
    #
    #    poison_rate = 0.7   # % of windows to poison
    #    k = 4               # last k timestep of the windows are poisoned 
    #
    #    bias = 0.15                     # For additive fixed bias
    #    bias_min, bias_max = 0.0, 0.3   # For additive random bias
    #
    #    # Features to poison
    #    js = [FEATURE_COLS.index("PM2.5"), FEATURE_COLS.index("PM10"), FEATURE_COLS.index("NO2")]
    #    
    #    n = X_tr_w.shape[0]                 # Number of windows
    #    mask = rng.random(n) < poison_rate  # Randomly selected windows to poison
    #    idx = np.where(mask)[0]             # Indexes of windows to poison
    #    noise = rng.uniform(bias_min, bias_max, size=(len(idx), k)).astype(np.float32)  # Noise matrix with shape: [len(idx), k]
    #                                                                                    # len(idx) = number of windows to poison
    #                                                                                    # k = number of last timestep to poison
    #
    #    # Apply bias to the selected windows (idx), on last k timestep, on features in js
    #    for j in js:
    #        #X_tr_w[mask, -k:, j] = np.clip(X_tr_w[mask, -k:, j] + bias, 0.0, 1.0)      # For additive fixed bias
    #        X_tr_w[idx, -k:, j] = np.clip(X_tr_w[idx, -k:, j] + noise, 0.0, 1.0)        # For additive random bias
    #
    #    print(
    #        f"[!!! ATTACK (Client {partition_id})!!!] FEATURE BIAS exeucted on training data ! "
    #        f"\nPoisoned feature(s): {[FEATURE_COLS[j] for j in js]} "
    #        f"\nAttack parameters: poison_rate={poison_rate}, k={k}, bias_min={bias_min}, bias_max={bias_max}"
    #    )
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # DATA POISONING #4: Target Shuffling between windows (24 length sequences)
    # ---------------------------------------------------------
    #if is_malicious:
    #    rng = np.random.default_rng(6000 + partition_id)
#
    #    poison_rate = 1.0               # % of windows to corrupt
    #    n = y_tr_w.shape[0]
    #    mask = rng.random(n) < poison_rate
    #    idx = np.where(mask)[0]         # Indexes of windows to corrupt
#
    #    y_perm = y_tr_w[idx].copy()    # Extract targets from selected windows
    #    rng.shuffle(y_perm)            # Selected targets are shuffled
    #    y_tr_w[idx] = y_perm           # Assign shuffled targets to windows
#
    #    print(
    #        f"[!!! ATTACK (Client {partition_id})!!!] TARGET SHUFFLING exeucted on training data ! "
    #        f"\nTargets shuffled between {len(idx)}/{n} windows"
    #        f"\nAttack parameter: poison_rate={poison_rate} "
    #    )
    # ---------------------------------------------------------


    train_ds = SequenceDataset(X_tr_w, y_tr_w)
    test_ds = SequenceDataset(X_te_w, y_te_w)

    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    testloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    # DataLoader mette insieme batch_size esempi (presi dal SequenceDataset), cioè prende 32 sequenze (24,11), (24,11), ...
    # e le impacchetta insieme --> Quindi costruisce i batch, ognuno con questa forma: (batch_size, SEQ_LEN, F).
    # Funzioni chiave del DataLoader:
    #   - Batching: se batch_size=32, il DataLoader preleva 32 finestre diverse dal dataset, ognuna è (SEQ_LEN, F). 
    #               Lui le impila per creare il tensore 3D di input per il modello (batch_size, SEQ_LEN, F).
    #   - Shuffling: Invece di dare alla rete i dati in ordine cronologico, le da esempi a caso. Questo impedisce alla
    #                rete di memorizzare l'ordine temporale dei giorni e la costringe a imparare la fisica del problema basandosi
    #                solo sulle 24 ore della finestra.
    #   - Gestione della memoria: carica i dati in modo efficiente durante il ciclo for xb, yb in trainloader:
    return trainloader, testloader


def load_data(partition_id: int, batch_size: int):
    """
    Carica train/test per il client identificato da partition_id.
    """
    if partition_id not in CLIENT_FILES:
        raise ValueError(
            f"partition_id={partition_id} not present in CLIENT_FILES={list(CLIENT_FILES.keys())}. "
        )
    
    train_path, test_path = CLIENT_FILES[partition_id]

    df_tr = pd.read_csv(train_path)
    df_te = pd.read_csv(test_path)

    X_tr = df_tr[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_tr = df_tr[TARGET_COL].to_numpy(dtype=np.float32)

    X_te = df_te[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_te = df_te[TARGET_COL].to_numpy(dtype=np.float32)

    X_tr_w, y_tr_w = _make_windows(X_tr, y_tr, SEQ_LEN, HORIZON)
    X_te_w, y_te_w = _make_windows(X_te, y_te, SEQ_LEN, HORIZON)

    train_ds = SequenceDataset(X_tr_w, y_tr_w)
    test_ds = SequenceDataset(X_te_w, y_te_w)

    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    testloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    # DataLoader mette insieme batch_size esempi (presi dal SequenceDataset), cioè prende 32 sequenze (24,11), (24,11), ...
    # e le impacchetta insieme --> Quindi costruisce i batch, ognuno con questa forma: (batch_size, SEQ_LEN, F).
    # Funzioni chiave del DataLoader:
    #   - Batching: se batch_size=32, il DataLoader preleva 32 finestre diverse dal dataset, ognuna è (SEQ_LEN, F). 
    #               Lui le impila per creare il tensore 3D di input per il modello (batch_size, SEQ_LEN, F).
    #   - Shuffling: Invece di dare alla rete i dati in ordine cronologico, le da esempi a caso. Questo impedisce alla
    #                rete di memorizzare l'ordine temporale dei giorni e la costringe a imparare la fisica del problema basandosi
    #                solo sulle 24 ore della finestra.
    #   - Gestione della memoria: carica i dati in modo efficiente durante il ciclo for xb, yb in trainloader:
    return trainloader, testloader

# -----------------------------
# Model
# -----------------------------
class Net(nn.Module):
    """Tiny LSTM regressor: predict PM2.5(t+HORIZON)."""

    def __init__(self, input_size: int = len(FEATURE_COLS), hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, F)
        out, _ = self.lstm(x)
        last = out[:, -1, :]           # (B, hidden)
        pred = self.head(last)         # (B, 1)
        return pred.squeeze(-1)        # (B,)
    

# -----------------------------
# Train / Test
# -----------------------------
def train(net: nn.Module, trainloader: DataLoader, epochs: int, lr: float, device: torch.device) -> float:
    net.to(device)
    net.train()

    criterion = nn.MSELoss()    # Loss function is MSE
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)   # Gradient Descent Algorithm: Adam

    total_loss = 0.0
    steps = 0

    for _ in range(epochs):
        for xb, yb in trainloader:
            xb = xb.to(device)
            yb = yb.to(device)          # Actual PM2.5 values

            optimizer.zero_grad()
            pred = net(xb)               # Predicted PM2.5 values

            #print("pred min/max:", pred.min().item(), pred.max().item())

            loss = criterion(pred, yb)  # Compute the MSE comparing predictions and actual values
            loss.backward()             # Compute the gradient for each weight

            optimizer.step()            # Updates the weight moving in the opposite direction of the gradient
                                        # This way the MSE is minimized 

            total_loss += float(loss.item())
            steps += 1

    return total_loss / max(steps, 1)

def train_malicious(net: nn.Module, trainloader: DataLoader, epochs: int, lr: float, device: torch.device, is_malicious: bool) -> float:
    net.to(device)
    net.train()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    total_loss = 0.0
    steps = 0

    for _ in range(epochs):
        for xb, yb in trainloader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = net(xb)

            loss = criterion(pred, yb)
            loss.backward()
            
            # ---------------------------------------------------------
            # MODEL POISONING #2: Gradient Ascent attack
            # ---------------------------------------------------------
            if is_malicious:
                for p in net.parameters():
                    if p.grad is not None:
                        p.grad.mul_(-1.0)   # Multiply gradient -1: invert the gradient direction
            # ---------------------------------------------------------

            optimizer.step()    # Now, by updating the weights moving in the opposite direction of the gradient, MSE is maximized

            total_loss += float(loss.item())
            steps += 1
    
    print(f"[!!! ATTACK !!!] GRADIENT ASCENT exeucted during training ! ")

    return total_loss / max(steps, 1)


@torch.no_grad()
def test(net: nn.Module, testloader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """
    Ritorna:
      mse, mae
    """
    net.to(device)
    net.eval()

    mse = 0.0
    mae = 0.0
    steps = 0

    for xb, yb in testloader:
        xb = xb.to(device)
        yb = yb.to(device)

        pred = net(xb)
        mse += torch.mean((pred - yb) ** 2).item()
        mae += torch.mean(torch.abs(pred - yb)).item()
        steps += 1

    return mse / max(steps, 1), mae / max(steps, 1)
