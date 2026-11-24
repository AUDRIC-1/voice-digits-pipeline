"""Evaluation metrics from predicted probabilities (no TensorFlow needed)."""
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)

from src.config import LABELS


def from_proba(proba, y):
    """All the metrics used in the notebook and in the app."""
    y = np.asarray(y)
    pred = proba.argmax(axis=1)
    labels = list(range(len(LABELS)))
    out = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision_macro": float(precision_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y, np.clip(proba, 1e-7, 1), labels=labels)),
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
        "per_class": {},
    }
    if len(set(y.tolist())) == len(LABELS):
        out["roc_auc_ovr"] = float(roc_auc_score(y, proba, multi_class="ovr", labels=labels))
    report = classification_report(y, pred, labels=labels, target_names=LABELS,
                                    output_dict=True, zero_division=0)
    for name in LABELS:
        r = report[name]
        out["per_class"][name] = {"precision": r["precision"], "recall": r["recall"],
                                  "f1": r["f1-score"], "support": int(r["support"])}
    return out


def accuracy(proba, y):
    if len(y) == 0:
        return None
    return float((proba.argmax(axis=1) == np.asarray(y)).mean())
