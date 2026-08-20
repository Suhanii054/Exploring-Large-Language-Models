"""From-scratch classification metrics: confusion matrix, per-class
precision/recall/F1, and macro-F1. No sklearn.metrics.
"""
import numpy as np


def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def precision_recall_f1_per_class(y_true, y_pred, n_classes):
    cm = confusion_matrix(y_true, y_pred, n_classes)
    tp = np.diag(cm).astype(np.float64)
    pred_pos = cm.sum(axis=0).astype(np.float64)   # column sums
    actual_pos = cm.sum(axis=1).astype(np.float64)  # row sums

    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.where(pred_pos > 0, tp / pred_pos, 0.0)
        recall = np.where(actual_pos > 0, tp / actual_pos, 0.0)
        f1 = np.where((precision + recall) > 0,
                       2 * precision * recall / (precision + recall + 1e-15),
                       0.0)
    return precision, recall, f1, cm


def macro_f1(y_true, y_pred, n_classes):
    _, _, f1, _ = precision_recall_f1_per_class(y_true, y_pred, n_classes)
    return f1.mean()


def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return (y_true == y_pred).mean()


def classification_report(y_true, y_pred, class_names):
    n_classes = len(class_names)
    precision, recall, f1, cm = precision_recall_f1_per_class(y_true, y_pred, n_classes)
    support = cm.sum(axis=1)

    lines = [f"{'label':<12}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}"]
    for i, name in enumerate(class_names):
        lines.append(f"{name:<12}{precision[i]:>10.4f}{recall[i]:>10.4f}{f1[i]:>10.4f}{support[i]:>10d}")
    lines.append("")
    lines.append(f"{'accuracy':<12}{'':>10}{'':>10}{accuracy(y_true, y_pred):>10.4f}{support.sum():>10d}")
    lines.append(f"{'macro avg':<12}{precision.mean():>10.4f}{recall.mean():>10.4f}{f1.mean():>10.4f}{support.sum():>10d}")
    return "\n".join(lines)
