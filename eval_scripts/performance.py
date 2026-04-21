"""Performance metrics used by training structures."""
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score as sklearn_f1
import numpy as np


def accuracy(true, pred):
    return accuracy_score(true.cpu().numpy(), pred.cpu().numpy())


def f1_score(true, pred, average="macro", zero_division=0):
    return sklearn_f1(true.cpu().numpy(), pred.cpu().numpy(), average=average, zero_division=zero_division)


def AUPRC(pts):
    """Area under precision-recall curve. pts is list of (score, label) tuples."""
    from sklearn.metrics import average_precision_score
    if len(pts) == 0:
        return 0.0
    scores = np.array([p[0] for p in pts])
    labels = np.array([p[1] for p in pts])
    return average_precision_score(labels, scores)


def eval_affect(true, pred, exclude_zero=True):
    """Accuracy for sentiment tasks with optional zero-label exclusion."""
    true = np.array(true)
    pred = np.array(pred)
    if exclude_zero:
        mask = true != 0
        true = true[mask]
        pred = pred[mask]
    return accuracy_score(true, pred)
