from typing import List, Optional, Iterable

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..query import ActiveQuery, QueryResult
from .mcdropout import MCDropoutQuery
from .vr import EnsembleVariationRatio


class MCVariationRatio(MCDropoutQuery):

    def _query_impl(self, batch_outs: torch.Tensor) -> List[float]:
        all_preds = torch.argmax(batch_outs, dim=-1).cpu().numpy() # [B, K]
        score = EnsembleVariationRatio.calc_variation_ratio(all_preds)
        return score