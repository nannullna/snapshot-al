from typing import Iterable
import numpy as np
import torch
from tqdm.auto import tqdm
from ..query import ActiveQuery, QueryResult

class LeastConfidentSampling(ActiveQuery):

    descending = False
    # The lower the predicted probability, the more uncertain.

    def _query_impl(self, size: int, dataloader: Iterable, **kwargs) -> QueryResult:
        dataloader = self.pool.get_unlabeled_dataloader(shuffle=False)
        all_scores = []

        device = self.device or torch.device("cuda")

        self.model.eval()
        with torch.no_grad():
            for X, _ in tqdm(dataloader, desc="LeastConfidentSampling"):
                X = X.to(device)
                
                out = self.classify(X)
                
                prob = torch.softmax(out, dim=1)
                score, _ = torch.max(prob, dim=1)
                all_scores.extend(score.detach().cpu().tolist())

        # returns the query with the k least confident probability
        return self.get_query_from_scores(all_scores, size=size, descending=self.descending)