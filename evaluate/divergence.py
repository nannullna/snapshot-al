import numpy as np


def kld(p: np.ndarray, q: np.ndarray, log_p: bool=False) -> np.ndarray:
    """Measures kl-divergence between p and q.
    p and q must be in the same shape of (N, C), 
    and it returns kl-divergences of shape (N,)"""
    if log_p:
        scores = np.exp(p) * (p - q)
    else:
        scores = p * (np.log(p, where=p>0.0) - np.log(q, where=q>0.0))
    scores = scores.sum(axis=-1)

    return scores


def symmetric_kld(p: np.ndarray, q: np.ndarray, log_p: bool=False) -> np.ndarray:
    """Measures symmetric kl-divergence between p and q."""
    return 0.5 * (kld(p, q, log_p) + kld(q, p, log_p))


def disagreement(p: np.ndarray, q: np.ndarray) -> float:
    pred_p = np.argmax(p, axis=-1)
    pred_q = np.argmax(q, axis=-1)
    return 1 - (pred_p == pred_q).float().mean()