from fastapi import FastAPI
from pydantic import BaseModel
import torch, pandas as pd
from pathlib import Path
from models.mlp import MLP          # reuse class definition

app = FastAPI()

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "mlp.pt"
FEATURE_ORDER = [...]               # list of column names, saved after ETL
MODEL = MLP(len(FEATURE_ORDER))
MODEL.load_state_dict(torch.load(MODEL_PATH))
MODEL.eval()

class Tx(BaseModel):
    # define one Pydantic field per feature
    step: int
    amount: float
    oldBalanceOrig: float
    newBalanceOrig: float
    # ... add others

@app.post("/predict")
def predict(tx: Tx):
    df = pd.DataFrame([tx.dict()])[FEATURE_ORDER]
    with torch.no_grad():
        score = float(MODEL(torch.tensor(df.to_numpy(), dtype=torch.float32))[0])
    return {"fraud_probability": score}
