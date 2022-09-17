from typing import Dict, Optional

class Tracker(object):
    """Useful utility for tracking a metric of a whole epoch."""

    _total_cnt: int = 0
    __slots__ = ['name', 'n', 'value']

    def __init__(self, name:Optional[str]=None) -> None:
        if name is not None:
            self.name = name
        else:
            self.name = f"Tracker_{Tracker._total_cnt}"
        Tracker._total_cnt += 1
        self.n = 0
        self.value = 0.0

    def __repr__(self) -> str:
        return f"[{self.name}] {self.value:.3f} ({self.n})"

    def __call__(self) -> float:
        return self.get()

    @property
    def __dict__(self) -> Dict[str, float]:
        return {self.name: self.value}

    def asdict(self) -> Dict[str, float]:
        return self.__dict__
    
    def get(self) -> float:
        return self.value

    def update(self, value: float, n: int=1):
        self.value = (self.value * self.n + value * n) / (self.n + n)
        self.n += n

    def reset(self):
        self.value = 0.0
        self.n = 0