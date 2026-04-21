"""Contiene:
    - Modello
    - Caricamento dati
    - Training
    - Valutazione"""

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

SEQ_LEN = 24    # Consider the 24 last hours
HORIZON = 1     # Prediction one hour ahead

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
# Client with partition_id = 0 uses files of Aotizhongxin station
# Client with partition_id = 1 uses files of Changping station
# And so on


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

    max_i = len(X) - horizon
    if max_i <= seq_len:
        raise ValueError(
            f"Few data to create windows: len={len(X)}, seq_len={seq_len}, horizon={horizon}."
        )

    Xs, ys = [], []
    for i in range(seq_len, max_i):     # Scorriamo lungo la serie temporale
        Xs.append(X[i - seq_len:i, :])  # Prendiamo una fetta di X che va da i -SEQ_LEN fino a i escluso. Cioè prende le 24 ore precedenti di tutte le 11 variabili
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

            optimizer.step()

            total_loss += float(loss.item())
            steps += 1

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
