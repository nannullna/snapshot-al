from typing import Optional, Iterable, List, Any
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm, trange

from .ensemble import EnsembleQuery
from ...pool import ActivePool
from ...query import QueryResult

class EnsembleGreedyVR(EnsembleQuery):

    def __init__(self, model: nn.Module, pool: ActivePool, size: int = 1, device: torch.device = None, **kwargs):
        super().__init__(model, pool, size, device, **kwargs)
        self.num_classes: int = None
        self.descending = True
        self.alpha = 0.02

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

    def query(self, checkpoints: List[str], size: int, dataloader: Optional[Iterable] = None, **kwargs) -> QueryResult:
        dataloader = dataloader or self.pool.get_unlabeled_dataloader()

        # temporarily save the model's state dict to restore it after querying
        temp_state_dict = deepcopy(self.model.state_dict())

        query_results = []
        for ckpt in tqdm(checkpoints, desc=self.__class__.__name__, unit="model"):
            self.model.load_state_dict(torch.load(ckpt)["state_dict"])
            query_results.append(self._query_impl(size, dataloader, **kwargs))
        all_probs = self.postprocess(query_results) # [N, K, C]
        all_preds = torch.argmax(all_probs, dim=2)  # [N, K]

        all_scores = self.calc_variation_ratio(all_preds.cpu().numpy(), self.num_classes) # [N,]

        # restore the model
        self.model.load_state_dict(temp_state_dict)

        # start querying here
        scores, indices = self.get_greedy_vr(all_probs.cpu().numpy(), all_scores, size)
        
        return QueryResult(scores=scores, indices=indices)

    
    def grid_kl_divergence(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """p: (N, C), q: (M, C) -> Returns: (N, M)"""
        p = p[:, None]
        q = q[None, :]
        scores = p * (np.log(p, where=p>0.0) - np.log(q, where=q>0.0))
        scores = scores.sum(axis=-1)

        return scores

    
    def get_greedy_vr(self, all_probs: np.ndarray, all_scores: np.ndarray, size: int):
        """all_probs of shape [N, K, C], all_scores of shape [N,]"""
        
        N, K, C = all_probs.shape
        _all_probs = all_probs.reshape(-1, C) # [N*K, C]
        query_scores, query_indices = [], []
        queried_klds = []

        for i in trange(size):

            if i == 0:
                _scores = all_scores
            
            else:
                queried_probs = all_probs[idx]      # [K, C]

                all_klds = self.grid_kl_divergence(queried_probs, _all_probs)   # [K, N * K]
                all_klds = all_klds.reshape(K, N, K)     # [K, N, K]
                avg_klds = all_klds.mean(axis=(0, 2))    # [N,]
                queried_klds.append(avg_klds)
                
                min_klds = np.vstack(queried_klds).min(axis=0)  # [N,]
                
                _scores = all_scores + self.alpha * min_klds
                _scores[query_indices] = -np.inf

            idx = np.argmax(_scores).item()
            query_scores.append(_scores[idx])
            query_indices.append(idx)

        return query_scores, query_indices
    

    def _query_impl(self, size: int, dataloader: Iterable, **kwargs) -> QueryResult:
        all_probs = []
        self.model.eval()
        with torch.no_grad():
            for imgs, _ in dataloader:
                imgs = imgs.to(self.device)
                logits = self.classify(imgs) # [B, C]
                if self.num_classes is None:
                    self.num_classes = logits.size(1)
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs)
        all_probs = torch.cat(all_probs, dim=0) # [N, C]
        return all_probs.unsqueeze(dim=1) # [N, 1, C]

    def postprocess(self, query_results: List[torch.Tensor]) -> torch.Tensor:
        query_results = torch.cat(query_results, dim=1) # [N, K, C]
        return query_results