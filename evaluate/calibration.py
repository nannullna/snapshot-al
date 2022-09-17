from typing import Dict, Union
import numpy as np
import matplotlib.pyplot as plt

def ece_loss(probs: np.ndarray, targets: np.ndarray, n_bins: int=10, return_dict=False):

    preds = np.argmax(probs, axis=-1)
    confs = np.take_along_axis(probs, np.expand_dims(preds, axis=-1), axis=-1).squeeze(1)

    bins = np.linspace(0.0, 1.0, n_bins+1)
    inds = np.digitize(confs, bins, right=True)

    ece = 0.0
    results = []

    for idx in range(1, n_bins+1):
        mask = (inds == idx)
        if np.sum(mask) == 0:
            weight = 0.0
            accuracy = 0.0
            confidence = 0.0
        
        else:
            weight = np.sum(mask) / len(mask)
            accuracy = np.sum(preds[mask] == targets[mask]) / np.sum(mask)
            confidence = np.mean(confs[mask])
        
        ece += weight * abs(accuracy - confidence)
        results.append({"bin_start": bins[idx-1], "bin_end": bins[idx], "weight": weight, "accuracy": accuracy, "confidence": confidence})

    if return_dict:
        return ece, results
    else:
        return ece


def nll(y_hat: np.ndarray, y_true: np.ndarray) -> float:
    """Calculates negative log likelihood (nll)."""

    prob = np.take_along_axis(y_hat, np.expand_dims(y_true, axis=-1), axis=-1)
    nll = -np.log(prob, where=prob > 0.0).mean()
    
    return nll


def draw_ece_plot(probs, targets, n_bins: int=10, figsize=(3, 3), dpi=150) -> plt.Figure:

    ece, results = ece_loss(probs, targets, n_bins, return_dict=True)

    x1 = [item['bin_start'] for item in results] #left edge
    x2 = [item['bin_end'] for item in results] #right edge
    y1 = [item['accuracy'] for item in results]
    y2 = [item['confidence'] for item in results]
    w = np.array(x2) - np.array(x1)

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor((1.0, 1.0, 1.0, 1.0))

    ax.plot([0.0, 1.0], [0.0, 1.0], ls='--', lw=2, c="#666666")
    ax.bar(x1, y2, width=w, align='edge', label="Expected", facecolor=(1, 0, 0.5, 0.2), edgecolor=(1, 0, 0.5, 1), lw=2, alpha=0.5)
    ax.bar(x1, y1, width=w, align='edge', label="Outputs", facecolor=(0, 0, 1, 1), edgecolor=(0, 0, 0, 1), lw=2, alpha=0.5)

    ax.text(0.9, 0.1, f"Error={ece*100:.1f}", fontsize=15, horizontalalignment='right', verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.7))

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")

    ax.legend()
    return fig


def draw_confidence_plot(probs, accuracy=None, n_bins:int = 20) -> plt.Figure:
    
    bins = np.linspace(0.0, 1.0, n_bins+1)
    avg_confidence = np.mean(probs)

    fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=150)
    fig.patch.set_facecolor((1.0, 1.0, 1.0, 1.0))

    ax.hist(probs, bins=bins, weights=np.ones(len(probs))/len(probs), facecolor=(0, 0, 1, 1), edgecolor=(0, 0, 0, 1), lw=2)

    ax.plot([avg_confidence, avg_confidence], [0.0, 1.0], ls='--', lw=2, c="#666666")
    ax.text(avg_confidence-0.05, 0.5, "Avg. Confidence", rotation='vertical', horizontalalignment='right', verticalalignment='bottom', fontsize=10, color=(1, 0, 0.5), bbox=dict(facecolor='white', alpha=0.7))

    if accuracy is not None:
        ax.plot([accuracy, accuracy], [0.0, 1.0], ls='--', lw=2, c="#666666")
        ax.text(accuracy-0.05, 0.5-0.1, "Accuracy", rotation='vertical', horizontalalignment='right', verticalalignment='top', fontsize=10, color=(0, 0, 1), bbox=dict(facecolor='white', alpha=0.7))

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("% of Samples")

    return fig
