from fastapi import FastAPI
from pydantic import BaseModel
import torch, pandas as pd
from pathlib import Path
import json
from prometheus_fastapi_instrumentator import Instrumentator
from models.mlp import MLP          # reuse class definition
import shap

app = FastAPI()
Instrumentator().instrument(app).expose(app)

BASE = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE / "models" / "mlp.pt"
FEATURES_PATH = BASE / "models" / "feature_order.json"
THRESHOLD_PATH = BASE / "models" / "threshold.txt"

with open(FEATURES_PATH) as f:
    FEATURE_ORDER = json.load(f)

with open(THRESHOLD_PATH) as f:
    THRESHOLD = float(f.read().strip())

MODEL = MLP(len(FEATURE_ORDER))
MODEL.load_state_dict(torch.load(MODEL_PATH))
MODEL.eval()
EXPLAINER = shap.Explainer(MODEL, torch.zeros((1, len(FEATURE_ORDER))))

class Tx(BaseModel):
    # minimal subset of features used by the model
    step: int
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    isFlaggedFraud: int = 0
    type_CASH_OUT: int = 0
    type_DEBIT: int = 0
    type_PAYMENT: int = 0
    type_TRANSFER: int = 0

@app.post("/predict")
def predict(tx: Tx):
    df = pd.DataFrame([tx.dict()])[FEATURE_ORDER]
    with torch.no_grad():
        score = float(MODEL(torch.tensor(df.to_numpy(), dtype=torch.float32))[0])
    pred = score >= THRESHOLD
    return {"fraud_probability": score, "is_fraud": bool(pred)}


@app.post("/explain")
def explain(tx: Tx):
    df = pd.DataFrame([tx.dict()])[FEATURE_ORDER]
    sample = torch.tensor(df.to_numpy(), dtype=torch.float32)
    shap_values = EXPLAINER(sample).values[0].tolist()
    return {"shap_values": dict(zip(FEATURE_ORDER, shap_values))}
