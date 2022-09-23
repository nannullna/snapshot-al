from typing import Iterable
import numpy as np
import torch
from tqdm.auto import tqdm

from ..query import ActiveQuery, QueryResult

class EntropySampling(ActiveQuery):

    descending = True
    # The higher the entropy, the more uncertain.

    @staticmethod
    def calc_entropy(x: torch.Tensor, log_p: bool=False, keepdim: bool=False) -> torch.Tensor:
        if log_p:
            entry = torch.exp(x) * x
        else:
            entry = x * torch.log(x)
            entry[x == 0.0] = 0.0
        entropy = -torch.sum(entry, dim=1, keepdim=keepdim)
        return entropy

    def _query_impl(self, size: int, dataloader: Iterable, **kwargs) -> QueryResult:
        all_scores = []

        self.model.eval()
        with torch.no_grad():
            for X, _ in tqdm(dataloader, desc="EntropySampling"):
                X = X.to(self.device)
                
                out = self.classify(X)
                log_prob = torch.log_softmax(out, dim=1)
                score = self.calc_entropy(log_prob, log_p=True)
                all_scores.extend(score.detach().cpu().tolist())
        
        # returns the query with the k highest entropies
        return self.get_query_from_scores(all_scores, size=size, descending=self.descending)