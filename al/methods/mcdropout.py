from typing import List, Optional, Iterable

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from ..query import ActiveQuery, QueryResult
from ..pool import ActivePool

class MCDropoutQuery(ActiveQuery):
    """Base class of MC-Dropout-based Query Implementation."""

    def __init__(self, model: nn.Module, pool: ActivePool, size: int = 1, device: torch.device = None, num_samples: int=5, **kwargs):
        super().__init__(model, pool, size, device, **kwargs)
        self.K = num_samples
        self.num_classes = None


    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """Enables Dropout whenever it is called."""
        _was_training = self.model.training
        
        # BatchNorm Layers must not be affected by changing the training mode.
        for n, p in self.model.named_modules():
            if isinstance(p, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                p.train(True) # turn on dropout
        
        output = super().classify(x)
        
        # Turn it back to the original training mode.
        self.model.train(_was_training)

        return output

    def query(self, size: int, dataloader: Optional[Iterable] = None, **kwargs) -> QueryResult:

        if dataloader is None:
            dataloader = self.pool.get_unlabeled_dataloader(**kwargs)
        
        all_scores = []

        self.model.eval()
        with torch.no_grad():
            for X, _ in tqdm(dataloader, desc="MCDropout"):

                batch_outs = []
                for _ in range(self.K):
                    X = X.to(self.device)
                    # self.classify() will handle turning MC-Dropout on.
                    out = self.classify(X) # possibly, [B, C]
                    if self.num_classes is None:
                        self.num_classes = out.size(1)
                    batch_outs.append(out.unsqueeze(1)) # possibly, [B, 1, C]
                
                batch_outs = torch.cat(batch_outs, dim=1) # [B, K, C]
                scores = self._query_impl(batch_outs) # [B, ]
                all_scores.extend(scores) # [N, ]

        return self.get_query_from_scores(scores, size=size, descending=self.descending)
                

    def _query_impl(self, batch_outs: torch.Tensor) -> List[float]:
        """You need to implement an acquisition function here!!!"""
        raise NotImplementedError
