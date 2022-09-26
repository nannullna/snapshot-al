from abc import abstractmethod
from copy import deepcopy
from typing import Any, Optional, Iterable, List, Union
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from tqdm import tqdm

from ..query import ActiveQuery, QueryResult, ActivePool

class EnsembleQuery(ActiveQuery):
    """Base class of Ensemble-Based query strategies."""
    def __init__(self, model: nn.Module, pool: ActivePool, size: int = 1, device: torch.device = None, max_size: Optional[int] = None, **kwargs):
        super().__init__(model, pool, size, device, **kwargs)
        self.max_size = max_size

    def __call__(self, checkpoints: List[str], size: Optional[int] = None, dataloader: Optional[Iterable] = None, **kwargs) -> QueryResult:
        size = size or self.size

        remainings = len(self.pool.unlabeled_data)
        start = time.time()
        
        if remainings == 0 or size == 0:
            # nothing left to query
            result = QueryResult(scores=[], indices=[])
        elif remainings < size:
            # query all the remainings
            result = QueryResult(
                scores=[0.0] * remainings, 
                indices=list(range(remainings)),
            )
        else:
            # its behavior depends on actual implementation!
            result = self.query(checkpoints, size, dataloader, **kwargs)

        end = time.time()
        if isinstance(result, tuple):
            result[0].info.update({"method": self.__class__.__name__, "time": end-start})
        else:
            result.info.update({"method": self.__class__.__name__, "time": end-start})
        return result

    def query(self, checkpoints: List[str], size: int, dataloader: Optional[Iterable] = None, **kwargs) -> QueryResult:
        random_ids = None
        if dataloader is None:
            # An ensemble-based query takes long. Instead, we can utilize a random subset to reduce computation.
            if (self.max_size is not None) and (self.max_size > 0):
                random_ids = np.random.choice(np.arange(len(self.pool.get_unlabeled_ids())), size=self.max_size, replace=False).tolist()
                random_subset = Subset(self.pool.get_unlabeled_dataset(), indices=random_ids)
                dataloader = DataLoader(random_subset, batch_size=self.pool.batch_size, shuffle=False, **kwargs)
            else:
                dataloader = self.pool.get_unlabeled_dataloader()

        # temporarily save the model's state dict to restore it after querying
        temp_state_dict = deepcopy(self.model.state_dict())
        query_results = []
        for ckpt in tqdm(checkpoints, desc=self.__class__.__name__, unit="model"):
            self.model.load_state_dict(torch.load(ckpt)["state_dict"])
            query_results.append(self._query_impl(size, dataloader, **kwargs))
        scores = self.postprocess(query_results)

        # restore the model
        self.model.load_state_dict(temp_state_dict)
        
        temp_query_result = self.get_query_from_scores(scores, size=size, descending=self.descending, **kwargs)
        if (self.max_size is not None) and (self.max_size > 0):
            temp_ids = temp_query_result.indices
            true_ids = [random_ids[idx] for idx in temp_ids]
            return QueryResult(scores=temp_query_result.scores, indices=true_ids, info=temp_query_result.info)
        else:
            return temp_query_result

    @abstractmethod
    def _query_impl(self, size: int, dataloader: Iterable, **kwargs) -> Union[List, torch.Tensor, np.ndarray, Any]:
        """Return values for each checkpoint which will be used in the `postprocess()` method.
        
        The type of values can be a list, a tensor, a numpy array, or any."""
        pass

    @abstractmethod
    def postprocess(self, query_results: List[Union[List, torch.Tensor, np.ndarray, Any]]) -> List:
        """Postprocess a list of values for all checkpoints and return a list of scores.
        Refer to `query()` method to see how it works.
        
        Please be aware that the first index of `query_results` is the checkpoint's index, 
        and the second one is the (unlabeled) example's index.
        """
        pass