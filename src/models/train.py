import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import pandas as pd
from utils.metrics import print_metrics
import json

DATA = Path(__file__).resolve().parents[2] / "data" / "processed"
MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_DIR.mkdir(exist_ok=True)

class MLP(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

def load_split(name: str):
    return pd.read_parquet(DATA / f"{name}.parquet").to_numpy()

def run(epochs: int = 15, lr: float = 1e-3):
    X_train = torch.tensor(load_split("train_X"), dtype=torch.float32)
    y_train = torch.tensor(load_split("train_y"), dtype=torch.float32).view(-1, 1)
    model = MLP(X_train.shape[1])
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=256, shuffle=True)
    loss_fn, opt = nn.BCELoss(), optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for xb, yb in loader:
            opt.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); opt.step()
        if epoch % 3 == 0: print(f"epoch {epoch} loss {loss.item():.4f}")

    torch.save(model.state_dict(), MODEL_DIR / "mlp.pt")
    print("✓ saved model")

    # quick evaluation + threshold tuning
    X_test = torch.tensor(load_split("test_X"), dtype=torch.float32)
    y_test = torch.tensor(load_split("test_y"), dtype=torch.float32).view(-1, 1)
    scores = model(X_test).detach().numpy()
    threshold = print_metrics(scores, y_test.numpy())
    with open(MODEL_DIR / "threshold.txt", "w") as f:
        f.write(str(threshold))

    # save feature order for API consumption
    feature_path = DATA / "feature_order.json"
    if feature_path.exists():
        with open(feature_path) as f:
            feature_order = json.load(f)
        with open(MODEL_DIR / "feature_order.json", "w") as f:
            json.dump(feature_order, f)

if __name__ == "__main__":
    run()
