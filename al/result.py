from typing import Any, List, Dict
from dataclasses import dataclass, field

@dataclass
class QueryResult:
    """A returned type of the class `ActiveQuery`"""
    scores: List[float]
    indices: List[int]
    labels: List[int] = field(init=False)
    info: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        return f"[Query Result] length {len(self.indices)}."

    def is_labeled(self):
        return bool(self.labels)

    def set_labels(self, labels):
        if len(self.scores) != len(labels):
            raise ValueError("The length of the query and that of the labels provided do not match.")
        self.labels = list(labels)