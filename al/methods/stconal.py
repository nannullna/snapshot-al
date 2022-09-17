from typing import Optional, Iterable, List, Any
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from .ensemble import EnsembleQuery
from ..pool import ActivePool
from ..query import QueryResult

class EnsembleStConal(EnsembleQuery):

    def __init__(self, model: nn.Module, pool: ActivePool, size: int = 1, device: torch.device = None, **kwargs):
        super().__init__(model, pool, size, device, **kwargs)
        self.num_classes: int = None
        self.descending = True

    def kl_divergence(self, all_probs: np.ndarray, ens_probs: Optional[np.ndarray]=None) -> np.ndarray:
        """all_probs: (N, K, C), ens_probs: Optional, (N, C) -> (N, K)"""
        if ens_probs is None:
            ens_probs = all_probs.mean(axis=1)  # if using SWA, provide ens_probs calculated with swa's weights.
        ens_probs = ens_probs[:, None] # [N, 1, C]
        klds = ens_probs * (np.log(ens_probs, where=ens_probs>0.0) - np.log(all_probs, where=all_probs>0.0))
        return klds.sum(axis=-1)

    def query(self, checkpoints: List[str], size: int, dataloader: Optional[Iterable] = None, swa_checkpoint: Optional[str]=None, **kwargs) -> QueryResult:
        dataloader = dataloader or self.pool.get_unlabeled_dataloader()

        # temporarily save the model's state dict to restore it after querying
        temp_state_dict = deepcopy(self.model.state_dict())
        
        query_results = []
        for ckpt in tqdm(checkpoints, desc=self.__class__.__name__, unit="model"):
            state_dict = torch.load(ckpt, map_location='cpu')["state_dict"]
            self.model.load_state_dict(state_dict)
            query_results.append(self._query_impl(size, dataloader, **kwargs))
        
        if swa_checkpoint is not None:
            state_dict = torch.load(swa_checkpoint, map_location='cpu')['state_dict']
            state_dict = {k[7:]: v for k, v in state_dict.items() if k.count('module.') > 0}
            self.model.load_state_dict(state_dict)
            swa_probs = self._query_impl(size, dataloader, **kwargs)
        else:
            swa_probs = None
        
        scores = self.postprocess(query_results, swa_probs)

        # restore the model
        self.model.load_state_dict(temp_state_dict)
        return self.get_query_from_scores(scores, size=size, descending=self.descending, **kwargs)

    def _query_impl(self, size: int, dataloader: Iterable, **kwargs) -> torch.Tensor:
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
        return all_probs.unsqueeze(1) # [N, 1, C]

    def postprocess(self, query_results: List[Any], swa_probs: torch.Tensor=None) -> List:
        all_probs = torch.cat(query_results, dim=1) # [N, K, C]
        print(f"Shape of all_probs {all_probs.shape}, swa_probs {swa_probs.shape}")
        scores = self.kl_divergence(all_probs.cpu().numpy(), swa_probs.squeeze(1).cpu().numpy() if swa_probs is not None else None) # [N, K]
        scores = scores.mean(axis=1) # [N, ]
        return scores