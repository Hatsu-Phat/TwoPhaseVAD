from __future__ import annotations

import numpy as np


def safe_roc_auc(y_true, y_score):
    from sklearn.metrics import roc_auc_score
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return float('nan')
    return float(roc_auc_score(y_true, y_score))


def safe_average_precision(y_true, y_score):
    from sklearn.metrics import average_precision_score
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return float('nan')
    return float(average_precision_score(y_true, y_score))


def classification_report_dict(y_true, y_pred):
    from sklearn.metrics import accuracy_score, f1_score
    return {
        'acc': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
    }
