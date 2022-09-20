from typing import Any, List, Dict, Optional, Union, Sequence, Iterable, Callable
from dataclasses import dataclass, field
from abc import abstractmethod
from copy import deepcopy
import warnings

import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset

from .result import QueryResult

class ActivePool:
    """Active pool that manages both an labeled and an unlabeled pool.
    
    Example:
    ```python
    # Load the dataset and create a pool
    train_transform, test_transform = (...), (...)
    train_set, eval_set = (..., transform=train_transform), (..., transform=test_transform)
    pool = ActivePool(dataset=train_set, batch_size=256, eval_set=eval_set)

    # Initialize the model and choose query strategies
    model = (...)
    init_sampler = RandomSampling(model=model, pool=pool, size=size)
    sampler = EntropySampling(model=model, pool=pool, size=size)
    
    # Get random examples for initial training
    result = init_sampler()
    pool.update(result)

    # Iterate over episodes (or repeat until the budget is exhausted.)
    for episode in range(num_episodes):
        # Train the model
        model.train()
        for images, labels in pool.get_labeled_dataloader():
            (...)

        # Query examples
        result = sampler()
        pool.update(result)

        # Optionally, evaluate the model
        with torch.no_grad():
            for images, labels in pool.get_eval_dataloader():
                (...)
        
        # Optionally, re-initialize the model's weights.
        model.init_weights()
    ```
    """

    def __init__(
        self,
        train_set: Union[Dataset, Subset, Iterable],
        query_set: Optional[Union[Dataset, Subset, Iterable]] = None,
        eval_set: Optional[Union[Dataset, Subset, Iterable]] = None,
        test_set: Optional[Union[Dataset, Subset, Iterable]] = None,
        batch_size: int = 1,
        labeled_ids: Optional[Sequence[int]] = None,
        eval_ids: Optional[Sequence[int]] = None,
        leftover_ids: Optional[Sequence[int]] = None,
    ):
        """Active learning updates the labeled pool iteratively 
        by querying human annotators to label some unlabeled examples
        which are believed to be beneficial to the model's performance.
        
        Args:
            `train_set`: `Dataset or Iterable`. Dataset which will be used to train a model. It is expected to have random augmentations.

            `query_set`: `Dataset or Iterable`. Dataset which will be used to query labels. It may or may not have random augmentations. 
            It must have the same length and the same order as the train_set does. Do NOT re-shuffle it. Use `train_set` if not provided. 

            `eval_set`: `Dataset or Iterable`. Dataset which will be used to evaluate the performance of the trained model. It may or may not have random augmentations.
            It must have the same length and the same order as the train_set does. Do NOT re-shuffle it. Use `query_set` if not provided.

            `test_set`: `Dataset or Iterable`.

            `batch_size`: `int (default: 1)`. Default batch size for data loaders.

            `labeled_ids`: `Optional[Sequence[int]]`. Provide if pre-labeled examples for training exist. 
            Otherwise, the whole dataset provided is assumed to be not labeled at all at the beginning.

            `eval_ids`: `Optional[Sequence[int]]`. Provide if pre-labeled examples for evaluation exist. 
            Otherwise, the whole dataset provided is assumed to have no evaluation set at the beginning. 

            `leftover_ids`: `Optional[Sequence[int]]`. These indices will be excluded from all datasets.
        """
        self.train_set = train_set
        self.query_set = query_set or self.train_set
        self.eval_set  = eval_set  or self.query_set
        self.test_set  = test_set

        self.batch_size = batch_size

        labeled_ids   = list(labeled_ids) if labeled_ids else []
        eval_ids      = list(eval_ids) if eval_ids else []
        leftover_ids  = list(leftover_ids) if leftover_ids else []
        unlabeled_ids = self.reverse_ids(labeled_ids+eval_ids+leftover_ids)

        self.labeled_data   = Subset(self.train_set, labeled_ids)
        self.eval_data      = Subset(self.eval_set,  eval_ids)
        self.unlabeled_data = Subset(self.query_set, unlabeled_ids)
        self.leftover_ids   = leftover_ids
        self.active_data    = Subset(self.train_set, labeled_ids) 

    def __repr__(self):
        result = (
            f"[Active Pool] unlabeled set {len(self.unlabeled_data)} ({len(self.unlabeled_data)/len(self.train_set)*100:.1f}%) // "
            f"labeled set {len(self.labeled_data)} ({len(self.labeled_data)/len(self.train_set)*100:.1f}%) // "
            f"eval set {len(self.eval_data)} ({len(self.eval_data)/len(self.train_set)*100:.1f}%)."
        )
        return result

    def __len__(self):
        return len(self.train_set)

    def update(self, result: Union[QueryResult, Sequence[int]], original_ids: bool=False):
        """Adds the examples provided by `result` to the labeled pool.
        This method (along with other methods in this class) exists to avoid complicated index matching procedures.

        Must be called once per `QueryResult`.

        Args:
            `result`: `QueryResult` - a returned value of the call of `ActiveQuery`.

            `original_ids`: `bool`
        """
        indices = result.indices if isinstance(result, QueryResult) else result
        indices = indices if original_ids else self.convert_to_original_ids(indices)        
        
        labeled_ids = list(set(self.get_labeled_ids() + indices))
        if len(indices) + len(self.get_labeled_ids()) != len(labeled_ids):
            # sanity check!!!
            warnings.warn("Updated indices may contain duplicates from itself.")

        self.active_data.indices = indices
        self.labeled_data.indices   = labeled_ids
        self.unlabeled_data.indices = self.reverse_ids(self.get_labeled_ids() + self.get_eval_ids() + self.get_leftover_ids())

    def update_eval(self, result: Union[QueryResult, Sequence[int]], original_ids: bool=False):
        """Adds the examples provided by `result` to the eval set.
        This method (along with other methods in this class) exists to avoid complicated index converting procedures.

        Args:
            `result`: `QueryResult` - a returned value of the call of `ActiveQuery`.

            `original_ids`: `bool`
        """
        indices = result.indices if isinstance(result, QueryResult) else result
        indices = indices if original_ids else self.convert_to_original_ids(indices)        
        self.eval_data.indices += indices
        self.unlabeled_data.indices = self.reverse_ids(self.get_labeled_ids() + self.get_eval_ids() + self.get_leftover_ids())

    def reset(self):
        """Clears up the labeled pool and sets the whole dataset as the unlabeled pool."""
        self.unlabeled_data.indices = list(range(len(self.train_set)))
        self.labeled_data.indices   = []
        self.eval_data.indices      = []
        self.leftover_ids           = []


    def get_classes(self) -> List[Any]:
        if isinstance(self.train_set, Subset):
            # `self.train_set` can be a Subset instance.
            dataset_ptr = self.train_set.dataset
        else:
            dataset_ptr = self.train_set
        if hasattr(dataset_ptr, "classes"):
            return list(dataset_ptr.classes)
        else:
            return []

    def get_unlabeled_targets(self) -> List[int]:
        """Returns target values (y) of the unlabeled pool. This method exists for experimental/analytical purposes."""
        if isinstance(self.train_set, Subset):
            # `self.train_set` can be a Subset instance.
            dataset_ptr = self.train_set.dataset
        else:
            dataset_ptr = self.train_set

        if hasattr(dataset_ptr, "targets"):
            # torchvision.datasets (MNIST, CIFAR10, CIFAR100, ...) has `targets` attribute.
            _targets = getattr(dataset_ptr, "targets")
            return [_targets[idx] for idx in self.get_unlabeled_ids()]
        else:
            raise NotImplementedError("self.dataset must have an attribute `targets`.")

    def get_labeled_targets(self) -> List[int]:
        """Returns target values (y) of the labeled pool."""
        if isinstance(self.train_set, Subset):
            # `self.train_set` can be a Subset instance.
            dataset_ptr = self.train_set.dataset
        else:
            dataset_ptr = self.train_set

        if hasattr(dataset_ptr, "targets"):
            # torchvision.datasets (MNIST, CIFAR10, CIFAR100, ...) has `targets` attribute.
            _targets = getattr(dataset_ptr, "targets")
            return [_targets[idx] for idx in self.get_labeled_ids()]
        else:
            raise NotImplementedError("self.dataset must have an attribute `targets`.")

    def get_unlabeled_dataloader(self, transform:Optional[Callable]=None, batch_size: Optional[int] = None, shuffle: bool = False, **kwargs):
        batch_size = batch_size or self.batch_size
        if transform is not None:
            dataset_ptr = deepcopy(self.unlabeled_data)
            dataset_ptr.dataset.transform = transform
        else:
            dataset_ptr = self.unlabeled_data
        
        return DataLoader(dataset_ptr, batch_size=batch_size, shuffle=shuffle, **kwargs)

    def get_labeled_dataloader(self, transform:Optional[Callable]=None, batch_size: Optional[int] = None, shuffle: bool = True, **kwargs):
        batch_size = batch_size or self.batch_size
        if transform is not None:
            dataset_ptr = deepcopy(self.labeled_data)
            dataset_ptr.dataset.transform = transform
        else:
            dataset_ptr = self.labeled_data
        return DataLoader(dataset_ptr, batch_size=batch_size, shuffle=shuffle, **kwargs)

    def get_active_dataloader(self, transform:Optional[Callable]=None, batch_size: Optional[int] = None, shuffle: bool = True, **kwargs):
        batch_size = batch_size or self.batch_size
        if transform is not None:
            dataset_ptr = deepcopy(self.active_data)
            dataset_ptr.dataset.transform = transform
        else:
            dataset_ptr = self.active_data
        return DataLoader(dataset_ptr, batch_size=batch_size, shuffle=shuffle, **kwargs)

    def get_eval_dataloader(self, transform:Optional[Callable]=None, batch_size: Optional[int] = None, shuffle: bool = False, **kwargs):
        batch_size = batch_size or self.batch_size
        if transform is not None:
            dataset_ptr = deepcopy(self.eval_data)
            dataset_ptr.dataset.transform = transform
        else:
            dataset_ptr = self.eval_data
        return DataLoader(dataset_ptr, batch_size=batch_size, shuffle=shuffle, **kwargs)

    def get_test_dataloader(self, transform:Optional[Callable]=None, batch_size: Optional[int] = None, shuffle: bool = False, **kwargs):
        batch_size = batch_size or self.batch_size
        if transform is not None:
            dataset_ptr = deepcopy(self.test_set)
            dataset_ptr.transform = transform
        else:
            dataset_ptr = self.test_set
        return DataLoader(dataset_ptr, batch_size=batch_size, shuffle=shuffle, **kwargs)

    def reverse_ids(self, indices: Sequence[int], length: Optional[int]=None) -> List[int]:
        """Returns the reversed indices of a list of indices given. 
        This method is useful when only a labeled or an unlabeled pool's indices are given.

        Args:
            `indices`: `Sequence[int]` - A list of indices
            
            `length`: `Optional[int]` - The total length of the dataset.
        """
        length = length or len(self.train_set)
        return list(set(range(length)) - set(indices))
    
    def convert_to_original_ids(self, indices: Sequence[int]) -> List[int]:
        """Returns the original indices from the unlabeled pool for data visualization. Must be called before any modification to the indicies of the unlabeled pool is applied."""
        return [self.unlabeled_data.indices[idx] for idx in indices]

    def convert_to_original_ids_labeled(self, indices: Sequence[int]) -> List[int]:
        """Returns the original indices from the labeled pool for data visualization. Must be called before any modification to the indicies of the unlabeled pool is applied."""
        return [self.labeled_data.indices[idx] for idx in indices]

    def get_unlabeled_pool(self) -> Subset:
        return self.unlabeled_data

    def get_labeled_pool(self) -> Subset:
        return self.labeled_data

    def get_eval_pool(self) -> Subset:
        return self.eval_data

    def get_unlabeled_dataset(self) -> Subset:
        return self.unlabeled_data

    def get_labeled_dataset(self) -> Subset:
        return self.labeled_data

    def get_test_dataset(self):
        return self.test_set

    def get_eval_dataset(self) -> Subset:
        return self.eval_data

    def get_unlabeled_ids(self) -> List[int]:
        return self.unlabeled_data.indices

    def get_labeled_ids(self) -> List[int]:
        return self.labeled_data.indices

    def get_eval_ids(self) -> List[int]:
        return self.eval_data.indices

    def get_leftover_ids(self) -> List[int]:
        return self.leftover_ids