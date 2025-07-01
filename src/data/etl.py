import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from pathlib import Path

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED = RAW.parent / "processed"
PROCESSED.mkdir(exist_ok=True)

def load_raw() -> pd.DataFrame:
    return pd.read_csv(RAW / "online_payments_fraud.csv")

def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = df.pop("isFraud")          # label column in chosen dataset
    df = pd.get_dummies(df, drop_first=True)
    return df, y

def oversample(X, y):
    sm = SMOTE(random_state=42)
    return sm.fit_resample(X, y)

def run():
    df = load_raw()
    X, y = preprocess(df)
    X_res, y_res = oversample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, stratify=y_res, random_state=42
    )
    for split, X_ in zip(("train", "test"), (X_train, X_test)):
        X_.to_parquet(PROCESSED / f"{split}_X.parquet")
    y_train.to_parquet(PROCESSED / "train_y.parquet")
    y_test.to_parquet(PROCESSED / "test_y.parquet")

if __name__ == "__main__":
    run()
