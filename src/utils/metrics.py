import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def best_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Return threshold in [0,1] that maximizes F1."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0, 1, 101):
        preds = scores >= t
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return float(best_t), float(best_f1)


def print_metrics(scores: np.ndarray, y_true: np.ndarray) -> float:
    """Print evaluation metrics and return optimal threshold."""
    threshold, best_f1 = best_threshold(y_true, scores)
    preds = scores >= threshold
    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds)
    rec = recall_score(y_true, preds)
    auc = roc_auc_score(y_true, scores)
    print(
        f"threshold={threshold:.2f} F1={best_f1:.3f} "
        f"Acc={acc:.3f} Prec={prec:.3f} Rec={rec:.3f} AUC={auc:.3f}"
    )
    return threshold
