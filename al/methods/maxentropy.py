from typing import Iterable, List, Any
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from .ensemble import EnsembleQuery
from ...pool import ActivePool
from ...query import QueryResult

class EnsembleMaxEntropy(EnsembleQuery):

    def __init__(self, model: nn.Module, pool: ActivePool, size: int = 1, device: torch.device = None, **kwargs):
        super().__init__(model, pool, size, device, **kwargs)
        self.descending = True

    @staticmethod
    def calc_entropy(all_probs: np.ndarray) -> np.ndarray:
        avg_probs = np.mean(all_probs, axis=1)
        entry = avg_probs * np.log(avg_probs, where=avg_probs>0)
        entropy = -np.sum(entry, axis=1)
        return entropy

    def _query_impl(self, size: int, dataloader: Iterable, **kwargs) -> QueryResult:
        all_logits = []
        self.model.eval()
        with torch.no_grad():
            for imgs, _ in dataloader:
                imgs = imgs.to(self.device)
                logits = self.classify(imgs) # [B, C]
                all_logits.extend(logits.unsqueeze(1)) # [B, 1, C]
        all_logits = torch.cat(all_logits, dim=0)
        all_probs = F.softmax(all_logits, dim=1).unsqueeze(1)
        return all_probs

    def postprocess(self, query_results: List[Any]) -> List:
        all_probs = torch.cat(query_results, dim=1).cpu().numpy() # [B, K, C]
        scores = self.calc_entropy(all_probs)
        return scores