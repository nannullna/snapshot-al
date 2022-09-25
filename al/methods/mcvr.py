from typing import List, Optional, Iterable

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..query import ActiveQuery, QueryResult, ActivePool
from .mcdropout import MCDropoutQuery
from .vr import EnsembleVariationRatio


class MCVariationRatio(MCDropoutQuery):

    def __init__(self, model: nn.Module, pool: ActivePool, size: int = 1, device: torch.device = None, num_samples: int = 5, **kwargs):
        super().__init__(model, pool, size, device, num_samples, **kwargs)
        self.descending = True

    def _query_impl(self, batch_outs: torch.Tensor) -> List[float]:
        all_preds = torch.argmax(batch_outs, dim=-1).cpu().numpy() # [B, K]
        score = EnsembleVariationRatio.calc_variation_ratio(all_preds, self.num_classes)
        return score.tolist()