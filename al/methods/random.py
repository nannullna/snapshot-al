from typing import Iterable
import numpy as np

from ..query import ActiveQuery, QueryResult

class RandomSampling(ActiveQuery):
    def _query_impl(self, size: int, dataloader: Iterable, **kwargs) -> QueryResult:
        return QueryResult(
            scores=[0.0] * size,
            indices=np.random.choice(np.arange(len(self.pool.unlabeled_data)), size, replace=False).tolist(),
        )