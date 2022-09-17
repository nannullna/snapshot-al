from typing import Any, Iterable, List, Dict, Optional, Union, Sequence
from abc import abstractmethod
import warnings
import time
import gc

import numpy as np
import torch
import torch.nn as nn

from .result import QueryResult

from .pool import ActivePool


class ActiveQuery:
    """An abstract class for deep active learning"""
    
    def __init__(self, model: nn.Module, pool: ActivePool, size: int=1, device: torch.device=None, **kwargs):
        """Initialize an active learning strategy. This class manages and tracks a labeled and unlabeled pool.
        If a model is not necessary for your strategy (e.g., RandomSampling), 
        then it is totally okay to provide `None` to the model argument.
        """
        if model is not None and not hasattr(model, "classify"):
            warnings.warn("The model provided must have a method `classify`. \
                Otherwise, it's assumed that the model's forward pass gives predictions.")
        self.model = model
        self.pool = pool
        self.size = size
        self.device = device
        self.descending = False
        
        self.num_workers = kwargs.get('num_workers', 1)
        self.pin_memory  = kwargs.get('pin_memory', False)

    def __repr__(self):
        return f"[{self.__class__.__name__}]"

    def __call__(self, size: Optional[int]=None, dataloader: Optional[Iterable]=None, **kwargs) -> QueryResult:
        """This wraps `query()` method but provides additional funtionalities.
        
        1. When called, it first checks whether the number of remaining unlabeled examples 
        is more than (or less than) the requested query size. 
        
            - If there is no remaining example in the unlabeled pool, 
            it just returns an empty list. 
            
            - Else if the number of remaining examples is less than the requested query size, 
            it does not run `query()` method and returns all indicies.
            
        2. It measures the time spent querying. 
        """
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
            result = self.query(size, dataloader, **kwargs)

        end = time.time()
        if isinstance(result, tuple):
            result[0].info.update({"method": self.__class__.__name__, "time": end-start})
        else:
            result.info.update({"method": self.__class__.__name__, "time": end-start})
        return result

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """Override this method to implement custom behaviors."""
        x = x.to(self.device)
        if hasattr(self.model, "classify"):
            return self.model.classify(x)
        else:
            return self.model(x)

    def query(self, size: int, dataloader: Optional[Iterable]=None, **kwargs) -> QueryResult:
        """This function wraps `_query_impl()` method and provides additional functionalities,
        including logging information regarding query method, time consumed, and etc.
        Therefore, you can safely assume that you are always provided size and dataloader 
        when implementing your `_query_impl()` method.

        You can override this method and implement custom behaviors.
        For example, BALD requires smaller batch size. So, you can customize batch size here."""
        dataloader = dataloader or self.pool.get_unlabeled_dataloader(num_workers=self.num_workers, pin_memory=self.pin_memory)
        return self._query_impl(size, dataloader, **kwargs)

    @abstractmethod
    def _query_impl(self, size: int, dataloader: Iterable, **kwargs) -> QueryResult:
        """Override this function to implement new acquisition functions."""
        pass
        
    def update_model(self, model: nn.Module):
        """Updates the reference to the model which will be used when querying."""
        del self.model
        gc.collect()
        self.model = model

    @staticmethod
    def get_query_from_scores(scores: Union[Sequence[float], Sequence[int]], size: Optional[int]=None, descending: bool = False, return_all_scores: bool = False, **kwargs):
        """Returns the queried indicies from the scores given. 
        
        Args: 
            scores: `Union[Sequence[float], Sequence[int]]` -- scores calculated from your `_query_impl`.

            size: `int` -- top-k size that will be queried. No default because it is a static method.

            descending: `bool` -- Query examples with k largest scores if True. (default: False)

            return_list: 'bool' -- The default behavior is to return Python's lists of scores and indices 
            of the queried examples. If return_list is False, this returns numpy arrays instead of Python's lists.

        Returns:
            QueryResult of indicies and scores. 
        """
        scores  = np.asarray(scores, dtype=np.float32)
        _indices = scores.argsort()
        if size is not None and isinstance(size, int):
            if descending:
                _indices = _indices[-size:]
            else:
                _indices = _indices[:size]
        elif descending:
            _indices = _indices[::-1]
        _scores = scores[_indices]
        
        if return_all_scores:
            return QueryResult(indices=_indices.tolist(), scores=_scores.tolist()), scores
        else:
            return QueryResult(indices=_indices.tolist(), scores=_scores.tolist())

