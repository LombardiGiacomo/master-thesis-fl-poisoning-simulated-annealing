import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from LSTMfederated.task import (
    Net,
    SequenceDataset,
    _make_windows,          
    FEATURE_COLS,
    TARGET_COL,
    SEQ_LEN,
    HORIZON,
)

TRAIN_Aotizhongxin = "aotizhongxin/aotizhongxin_train_scaled.csv"
TEST_Aotizhongxin = "aotizhongxin/aotizhongxin_test_scaled.csv"
TRAIN_Changping = "changping/changping_train_scaled.csv"
TEST_Changping = "changping/changping_test_scaled.csv"
TRAIN_Dingling = "dingling/dingling_train_scaled.csv"
TEST_Dingling = "dingling/dingling_test_scaled.csv"
TRAIN_Dongsi = "dongsi/dongsi_train_scaled.csv"
TEST_Dongsi = "dongsi/dongsi_test_scaled.csv"
TRAIN_Guanyuan = "guanyuan/guanyuan_train_scaled.csv"
TEST_Guanyuan = "guanyuan/guanyuan_test_scaled.csv"

def read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
    #df = df[FEATURE_COLS].copy()    # tengo solo colonne utili
    return df

def make_loader_from_multiple_dfs(dfs: list, batch_size: int, shuffle: bool) -> DataLoader:
    """
    Genera finestre separatamente per ogni dataframe e poi le unisce.
    """
    all_Xw = []
    all_yw = []

    for df in dfs:
        X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
        y = df[TARGET_COL].to_numpy(dtype=np.float32)
        
        # Creo le finestre SOLO per questo dataframe
        Xw, yw = _make_windows(X, y, SEQ_LEN, HORIZON)
        all_Xw.append(Xw)
        all_yw.append(yw)

    # Concateno i tensori risultanti (lungo l'asse degli esempi)
    Xw_final = np.concatenate(all_Xw, axis=0)
    yw_final = np.concatenate(all_yw, axis=0)

    ds = SequenceDataset(Xw_final, yw_final)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

def train_central(model, trainloader, epochs, lr, device):
    model.to(device)    # Sposta il modello sull'hardware specificato (nel nostro caso CPU) per eseguire i calcoli
    model.train()       # Imposta il modello in modalità addestramento: dice a PyTorch di attivare i meccanismi per il training (come il calcolo dei gradienti)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)     # Algoritmo di ottimizzazione: Adam -> adatta il lr per convergere piu velocamente
    criterion = torch.nn.MSELoss()  # Funzione di costo: siccome vogliamo predire un valore numerico continuo, usiamo il MSE.
                                    # Quindi il modello cercherà di minimizzare la differenza al quadrato tra le sue predizioni e il valore reale

    for _ in range(epochs):     # Il modello rivedrà il dataset epochs volte per raffinare l'apprendimento
        for xb, yb in trainloader:      
            # Questo ciclo estrae i dati dal trainloader. Invece di passare tutto il dataset, il loader fornisce piccoli batch di dati.
            # xb è il batch di input (tensore 3D: batch_size x seq_len x num_features)
            # yb è il batch dei target reali (vettore: batch_size di valori TARGET_COL reali) 
            xb, yb = xb.to(device), yb.to(device)   # I dati appena caricati vengono spostati sulla stessa periferica del modello
            optimizer.zero_grad()   # PyTorch per default accumula i gradienti. Azzerandoli ad ogni passo evitiamo che i calcoli del batch attuale si sommino a quelli del batch precedente
            pred = model(xb)    # Il modello riceve l'input xb attraverso i layer LSTM e Lineare, e produce la sua predizione (quanto pensa che varrà TARGET_COL)
            loss = criterion(pred, yb)  # Si confronta la predizione con il valore vero yb usando il MSE. loss dice quanto il modello ha sbagliato.
            loss.backward()     # PyTorch calcola in che direzione e di quanto ogni peso della rete deve essere spostato per ridurre l'errore
            optimizer.step()    # L'optimizer usa i gradienti calcolati per modificare effettivamente i pesi del modello

@torch.no_grad()
def eval_model(model, loader, device):
    model.to(device)
    model.eval()
    mse = 0.0
    mae = 0.0
    steps = 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        mse += torch.mean((pred - yb) ** 2).item()
        mae += torch.mean(torch.abs(pred - yb)).item()
        steps += 1

    return mse / max(steps, 1), mae / max(steps, 1)


def main():
    # iperparametri (uguali a quelli FL per confronto)
    batch_size = 64
    lr = 0.001
    epochs = 20     # epochs = local_epochs * num_rounds
                    # In FL faccio x round con y epoche locali ciascuno, quindi il modello vede i dati 
                    # globalmente x*y volte. Qui metto epochs=x*y per un confronto equo.
   
    device = torch.device("cpu")

    # --- Caricamento dati per train ---
    df_train_Aotizhongxin = read_csv(TRAIN_Aotizhongxin)
    df_train_Changping = read_csv(TRAIN_Changping)
    df_train_Dingling = read_csv(TRAIN_Dingling)
    df_train_Dongsi = read_csv(TRAIN_Dongsi)
    df_train_Guanyuan = read_csv(TRAIN_Guanyuan)
    
    #df_train = pd.concat([df_train_Aotizhongxin, df_train_Changping, df_train_Dingling, df_train_Dongsi, df_train_Guanyuan], axis=0, ignore_index=True)

    # Passo una lista di df
    trainloader = make_loader_from_multiple_dfs(
        [df_train_Aotizhongxin, df_train_Changping, df_train_Dingling, df_train_Dongsi, df_train_Guanyuan], 
        batch_size=batch_size, 
        shuffle=True
    )

    # --- Loader test ---
    df_test_Aotizhongxin = read_csv(TEST_Aotizhongxin)
    df_test_Changping = read_csv(TEST_Changping)
    df_test_Dingling = read_csv(TEST_Dingling)
    df_test_Dongsi = read_csv(TEST_Dongsi)
    df_test_Guanyuan = read_csv(TEST_Guanyuan)

    # Test globale (unione dei test set)
    testloader_global = make_loader_from_multiple_dfs(
        [df_test_Aotizhongxin, df_test_Changping, df_test_Dingling, df_test_Dongsi, df_test_Guanyuan],
        batch_size=batch_size,
        shuffle=False,
    )

    # Test singoli
    testloader_Aotizhongxin = make_loader_from_multiple_dfs([df_test_Aotizhongxin], batch_size, False)
    testloader_Changping = make_loader_from_multiple_dfs([df_test_Changping], batch_size, False)
    testloader_Dingling = make_loader_from_multiple_dfs([df_test_Dingling], batch_size, False)
    testloader_Dongsi = make_loader_from_multiple_dfs([df_test_Dongsi], batch_size, False)
    testloader_Guanyuan = make_loader_from_multiple_dfs([df_test_Guanyuan], batch_size, False)

    # --- Training ---
    model = Net()

    print("Training centralized baseline...")
    train_central(model, trainloader, epochs=epochs, lr=lr, device=device)

    # --- Evaluation ---
    mse_g, mae_g = eval_model(model, testloader_global, device)
    mse_Aotizhongxin, mae_Aotizhongxin = eval_model(model, testloader_Aotizhongxin, device)
    mse_Changping, mae_Changping = eval_model(model, testloader_Changping, device)
    mse_Dingling, mae_Dingling = eval_model(model, testloader_Dingling, device)
    mse_Dongsi, mae_Dongsi = eval_model(model, testloader_Dongsi, device)
    mse_Guanyuan, mae_Guanyuan = eval_model(model, testloader_Guanyuan, device)

    print("\nCentralized results:")
    print(f"  On aggregated test set -> MSE={mse_g:.6f}  MAE={mae_g:.6f}")
    print(f"  On Aotizhongxin test set -> MSE={mse_Aotizhongxin:.6f}  MAE={mae_Aotizhongxin:.6f}")
    print(f"  On Changping test set -> MSE={mse_Changping:.6f}  MAE={mae_Changping:.6f}")
    print(f"  On Dingling test set -> MSE={mse_Dingling:.6f}  MAE={mae_Dingling:.6f}")
    print(f"  On Dongsi test set -> MSE={mse_Dongsi:.6f}  MAE={mae_Dongsi:.6f}")
    print(f"  On Guanyuan test set -> MSE={mse_Guanyuan:.6f}  MAE={mae_Guanyuan:.6f}")

    torch.save(model.state_dict(), "centralized_model.pt")
    print("\nSaved: centralized_model.pt")

if __name__ == "__main__":
    main()