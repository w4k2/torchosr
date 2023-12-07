import numpy as np
from sklearn.datasets import make_classification
from torch.utils.data import Dataset
from typing import Any, Tuple

class Madelon_base(Dataset):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.data, self.targets = self._load_data()

    def _load_data(self):
        data, targets = make_classification(**self.kwargs)
        return data.astype(np.float32), targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        return self.data[index], self.targets[index]

    def __len__(self) -> int:
        return len(self.targets)
    
    def _n_classes(self) -> int:
        return len(np.unique(self.targets))
