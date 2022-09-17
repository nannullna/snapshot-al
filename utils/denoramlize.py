from typing import Sequence, List
import torchvision.transforms as T

class Denormalize(object):
    """De-normalize a normalized image. Especially useful when visualizing images with random augmentations."""

    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        """
        Args:
            mean: original mean provided to torchvision.transforms.Normalize
            std: original standard deviation provided to torchvision.transforms.Normalize
        """
        self.mean: List[float] = list(mean)
        self.std:  List[float] = list(std)
        self.transform = T.Normalize(
            mean=self.inv_mean,
            std=self.inv_std
        )

    @property
    def inv_mean(self) -> List[float]:
        return [-m/s for m, s in zip(self.mean, self.std)]

    @property
    def inv_std(self) -> List[float]:
        return [1/s for s in self.std]

    def __call__(self, x):
        return self.transform(x)