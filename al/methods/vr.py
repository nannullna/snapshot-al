from typing import Iterable, List, Any
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from .ensemble import EnsembleQuery
from ..pool import ActivePool
from ..query import QueryResult

class EnsembleVariationRatio(EnsembleQuery):

    def __init__(self, model: nn.Module, pool: ActivePool, size: int = 1, device: torch.device = None, **kwargs):
        super().__init__(model, pool, size, device, **kwargs)
        self.num_classes: int = None
        self.descending = True

    @staticmethod
    def calc_variation_ratio(all_preds: np.ndarray, num_classes: int) -> np.ndarray:
        B, K = all_preds.shape
        votes = np.zeros((B, num_classes), dtype=int)
        for cls in range(num_classes):
            votes[:, cls] = np.sum(all_preds == cls, axis=1)
        score = 1 - np.max(votes, axis=1) / K 
        # Variation ratio: 1 - f_m / K, 
        #   where f_m denotes that the number of predictions 
        #   falling into the modal class category, m, over K predictions.
        return score

    def _query_impl(self, size: int, dataloader: Iterable, **kwargs) -> QueryResult:
        all_preds = []
        self.model.eval()
        with torch.no_grad():
            for imgs, _ in dataloader:
                imgs = imgs.to(self.device)
                logits = self.classify(imgs) # [B, C]
                if self.num_classes is None:
                    self.num_classes = logits.size(1)
                preds = torch.argmax(logits, dim=1) # [B, 1]
                all_preds.extend(preds.cpu().tolist())
        return all_preds

    def postprocess(self, query_results: List[Any]) -> List:
        query_results = np.array(query_results).T
        scores = self.calc_variation_ratio(query_results, self.num_classes)
        return scores