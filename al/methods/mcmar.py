from typing import List, Optional, Iterable

import torch
import torch.nn.functional as F

from ..pool import ActivePool
from .mcdropout import MCDropoutQuery


class MCMarginSampling(MCDropoutQuery):

    def __init__(self, model: torch.nn.Module, pool: ActivePool, size: int = 1, device: torch.device = None, **kwargs):
        super().__init__(model, pool, size, device, **kwargs)
        self.descending = False

    def _query_impl(self, batch_outs: torch.Tensor) -> List[float]:
        prob = F.softmax(batch_outs, dim=1) # [B, K, C]
        prob = prob.mean(dim=1).cpu().numpy() # [B, C]
        prob.sort(axis=-1)
        margin = prob[:, -1] - prob[:, -2]
        return margin.tolist()