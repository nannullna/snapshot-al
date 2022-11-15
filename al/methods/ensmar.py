from typing import Iterable, List, Any
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from .ensemble import EnsembleQuery
from ..pool import ActivePool
from ..query import QueryResult

class EnsembleMarginSampling(EnsembleQuery):

    def __init__(self, model: nn.Module, pool: ActivePool, size: int = 1, device: torch.device = None, **kwargs):
        super().__init__(model, pool, size, device, **kwargs)
        self.descending = True

    @staticmethod
    def calc_margin(all_probs: np.ndarray) -> np.ndarray:
        all_probs = np.asarray(all_probs)
        all_probs.sort(axis=-1)
        margin = all_probs[:, -1] - all_probs[:, -2]
        return margin

    def _query_impl(self, size: int, dataloader: Iterable, **kwargs) -> QueryResult:
        all_probs = []
        self.model.eval()
        with torch.no_grad():
            for imgs, _ in dataloader:
                imgs = imgs.to(self.device)
                logits = self.classify(imgs) # [B, C]
                preds = F.softmax(logits, dim=1) # [B, C]
                all_probs.append(preds.cpu())
        all_probs = torch.cat(all_probs, dim=0)
        return all_probs.unsqueeze(1)

    def postprocess(self, query_results: List[Any]) -> List:
        query_results = torch.cat(query_results, dim=1).cpu().numpy() # [B, K, C]
        all_probs = query_results.mean(axis=1) # [B, C]
        scores = self.calc_margin(all_probs)
        return scores