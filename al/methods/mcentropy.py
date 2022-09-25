from typing import List, Optional, Iterable

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..query import ActiveQuery, QueryResult
from .mcdropout import MCDropoutQuery
from .mcbald import MCBald


class MCEntropy(MCDropoutQuery):

    def _query_impl(self, batch_outs: torch.Tensor) -> List[float]:
        log_prob = F.log_softmax(batch_outs, dim=-1)  # [B, K, C]
        score = MCBald.calc_entropy(log_prob)
        return score.cpu().tolist()