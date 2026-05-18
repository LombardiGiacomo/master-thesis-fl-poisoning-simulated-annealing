import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from LSTMfederated.task import Net

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
    strategy = FedAvg(fraction_evaluate=fraction_evaluate)  # FedAvg standard (valutando su una frazione fraction_evaluate dei client)
    # FedAvg di default fa anche la media pesata usando num-examples che i client inviano nelle metriche

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

