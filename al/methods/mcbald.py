from typing import List, Optional, Iterable

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..query import ActiveQuery, QueryResult
from .mcdropout import MCDropoutQuery


class MCBald(MCDropoutQuery):

    @staticmethod
    def calc_entropy(x: torch.Tensor, log_p: bool = True, keepdim: bool = False) -> torch.Tensor:
        # reduce dimension
        x = x.to(dtype=torch.double)
        if log_p:
            mean_log_prob = torch.logsumexp(x, dim=1) - math.log(x.size(1))
            entry = torch.exp(mean_log_prob) * mean_log_prob
        else:
            mean_prob = torch.mean(x, dim=1, dtype=torch.double)
            entry = mean_prob * torch.log(mean_prob)
            entry[mean_prob == 0.0] = 0.0
        entropy = -torch.sum(entry, dim=1, keepdim=keepdim)
        return entropy


    @staticmethod
    def calc_conditional_entropy(x: torch.Tensor, log_p: bool = True, keepdim: bool = False) -> torch.Tensor:
        x = x.to(dtype=torch.double)
        if log_p:
            entry = torch.exp(x) * x
        else:
            entry = x * torch.log(x)
            entry[x == 0.0] = 0.0
        entropy = -torch.sum(entry, dim=-1, dtype=torch.double)
        entropy = torch.mean(entropy, dim=1, keepdim=keepdim)
        return entropy


    def _query_impl(self, batch_outs: torch.Tensor) -> List[float]:
        log_prob = F.log_softmax(batch_outs, dim=-1)  # [B, K, C]
        score = self.calc_entropy(log_prob) - self.calc_conditional_entropy(log_prob)
        return score.cpu().numpy()