import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from LSTMfederated.task import Net, load_data
from LSTMfederated.task import test as test_fn
from LSTMfederated.task import train as train_fn

from flwr.common import ConfigRecord

app = ClientApp()   # Flower ClientApp

@app.train()
def train(msg: Message, context: Context):  # Funzione chiamata dal server ad ogni round
    """Train the model on local data."""

    # Load the model and initialize it with the received weights
    # 1) Ricezione del modello globale  --> il modello globale diventa modello locale
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())  # Il server manda i pesi globali dentro msg.content["arrays"]
                                                                        # e li carichiamo dentro il modello locale model

    # 2) Device (normale PyTorch)
    device = torch.device("cpu")
    model.to(device)

    # 3) Load the data --> Caricamento dei dati locali
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    trainloader, _ = load_data(partition_id, batch_size)

    # 4) Call the training function --> Training 
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],    # il learning rate viene preso dal messaggio perche è il server che decide che config mandare ai client
        device,
    )

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
