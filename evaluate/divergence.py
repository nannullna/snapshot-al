import numpy as np


def kl_divergence(p: np.ndarray, q: np.ndarray, log_p: bool=False):
    if log_p:
        scores = np.exp(p) * (p - q)
    else:
        scores = p * (np.log(p, where=p>0.0) - np.log(q, where=q>0.0))
    scores = scores.sum(axis=-1)

    return scores