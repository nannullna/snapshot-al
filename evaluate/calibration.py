from typing import Dict, Union
import numpy as np


def ece(y_hat: np.ndarray, y_true: np.ndarray, n_bins: int=10, return_dict: bool=False) -> Union[float, Dict]:
    """Calculates expected calibration error (ECE).
    
    Args:

        y_hat: np.ndarray -- softmax value (probabiltiy) of shape N-by-C (C: # of classes)

        y_true: np.ndarray -- target of shape N or N-by-C (one-hot vector)
    
    """
    if len(y_hat) != len(y_true):
        raise ValueError(f"Length of y_pred {len(y_pred)} and y_true {len(y_true)} do not match.")

    if y_true.ndim == 2 and y_true.shape(1) == 1:
        y_true.squeeze(1)
    elif y_true.ndim == 2 and y_true.shape(1) > 1:
        y_true = y_true.argmax(axis=1) # assumed to be an one-hot vector.

    bins = np.linspace(0.0, 1.0, n_bins+1)
    accs, confs, Bm = np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins)

    y_pred = y_hat.argmax(axis=1)
    all_conf = y_hat.max(axis=1)

    for i, (start, end) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (all_conf > start) & (all_conf <= end)
        Bm[i] = mask.sum()
        if Bm[i] == 0:
            # no example in the bin
            continue

        _pred = y_pred[mask]
        _conf = all_conf[mask]
        _true = y_true[mask]
        
        accs[i] = (_pred == _true).mean()
        confs[i] = _conf.mean()

    Bm = Bm / len(y_true)
    ece = (Bm * np.abs(accs - confs)).sum()

    if return_dict:
        return ece, {
            "n": Bm,
            "accs": accs,
            "confs": confs
        }
    else:
        return ece


def nll(y_hat: np.ndarray, y_true: np.ndarray) -> float:
    """Calculates negative log likelihood (nll)."""

    prob = np.take_along_axis(y_hat, np.expand_dims(y_true, axis=-1), axis=-1)
    nll = -np.log(prob, where=prob > 0.0).mean()
    
    return nll


def draw_reliability_plot(y_hat: np.ndarray, y_true: np.ndarray, n_bins: int=10, dpi: int=72, figsize=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, dpi=dpi, figsize=figsize)

    pass

