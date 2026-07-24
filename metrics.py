import numpy as np

def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> precision_at_k(r, 1)
    0.0
    >>> precision_at_k(r, 2)
    0.0
    >>> precision_at_k(r, 3)
    0.33333333333333331
    >>> precision_at_k(r, 4)
    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)


def average_precision(r):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    >>> delta_r = 1. / sum(r)
    >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
    0.7833333333333333
    >>> average_precision(r)
    0.78333333333333333
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Average precision
    """
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)


def mean_average_precision(rs):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]
    >>> mean_average_precision(rs)
    0.78333333333333333
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0]]
    >>> mean_average_precision(rs)
    0.39166666666666666
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])


def f1_score(y_true, y_pred, average="micro"):
    """Compute F1 score between two sequences of labels (e.g. for entity linking exact match).

    For single-label / exact-match evaluation, each pair (y_true[i], y_pred[i]) is either
    a match (TP) or not. With average='micro', F1 = (2 * TP) / (2 * TP + FP + FN) which
    for one prediction per sample reduces to accuracy: correct / total.

    Args:
        y_true: List of gold labels (e.g. gold entity strings).
        y_pred: List of predicted labels (same length as y_true).
        average: 'micro' (default) for global F1, or 'binary' for single-class.

    Returns:
        Float F1 score.
    """
    assert len(y_true) == len(y_pred)
    y_true = list(y_true)
    y_pred = list(y_pred)
    n = len(y_true)
    if n == 0:
        return 0.0
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    # For exact match: we have one prediction per sample, so
    # TP = tp, FP = n - tp, FN = n - tp  => P = tp/n, R = tp/n, F1 = tp/n
    if average == "micro" or average == "binary":
        return tp / n
    raise ValueError("average must be 'micro' or 'binary'")


def precision_recall_f1_for_sets(gold_set, pred_set):
    """Compute precision, recall, F1 for one sample where gold and pred are sets of labels.

    F1 formula: P = |pred ∩ gold| / |pred|, R = |pred ∩ gold| / |gold|,
    F1 = 2 * P * R / (P + R) when P+R > 0, else 0.

    Args:
        gold_set: Set (or list) of gold labels.
        pred_set: Set (or list) of predicted labels.

    Returns:
        (precision, recall, f1) floats.
    """
    gold_set = set(gold_set) if not isinstance(gold_set, set) else gold_set
    pred_set = set(pred_set) if not isinstance(pred_set, set) else pred_set
    intersection = len(gold_set & pred_set)
    if len(pred_set) == 0:
        precision = 0.0
    else:
        precision = intersection / len(pred_set)
    if len(gold_set) == 0:
        recall = 0.0
    else:
        recall = intersection / len(gold_set)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return (precision, recall, f1)


def macro_f1_score_sets(y_true_sets, y_pred_sets):
    """Macro F1: average of per-sample F1 when each sample has a set of labels (e.g. CTA).

    Args:
        y_true_sets: List of gold sets (or lists).
        y_pred_sets: List of predicted sets (or lists), same length as y_true_sets.

    Returns:
        Float: mean F1 over samples.
    """
    assert len(y_true_sets) == len(y_pred_sets)
    if not y_true_sets:
        return 0.0
    f1s = [precision_recall_f1_for_sets(g, p)[2] for g, p in zip(y_true_sets, y_pred_sets)]
    return np.mean(f1s)