import numpy as np
from sklearn.datasets import make_moons
from torch.utils.data import Dataset
from typing import Any, Tuple
from sklearn.utils import shuffle

def make_moonway(n_circles = 5, sigma = .1, n_samples_moons = 100, n_samples_per_circle = 100, mode = 'linear',
                 random_state=None, padding=2):
    np.random.seed(random_state)
    
    n_samples = n_samples_moons + n_circles * n_samples_per_circle
    noise = np.random.normal(0, sigma, size=(n_samples, 2))

    m_X, m_y = make_moons(n_samples=n_samples_moons)
    m_X -= [.5,0]

    if mode == 'linear':
        circle_radiuses = padding + np.arange(n_circles)
    elif mode == 'log':    
        circle_radiuses = np.logspace(padding, n_circles, n_circles)
    else:
        raise Exception("Only linear and log modes available")
    
    angles = np.random.uniform(0, np.pi*2, (n_samples_per_circle, n_circles))

    c_X = np.array([np.sin(angles) * circle_radiuses[None,:],
                    np.cos(angles) * circle_radiuses[None,:]])
    c_y = np.ones((n_samples_per_circle, n_circles)) * (np.arange(n_circles)+2)[None, :]

    c_X = c_X.reshape(2,-1).T
    c_y = c_y.reshape(-1)

    X = np.concatenate((c_X, m_X)) + noise
    y = np.concatenate((c_y, m_y))
    
    X, y = shuffle(X, y, random_state=random_state)
    
    return X, y

class Moonway_base(Dataset):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.data, self.targets = self._load_data()

    def _load_data(self):
        data, targets = make_moonway(**self.kwargs)
        return data.astype(np.float32), targets.astype(np.int64)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        return self.data[index], self.targets[index]

    def __len__(self) -> int:
        return len(self.targets)
    
    def _n_classes(self) -> int:
        return len(np.unique(self.targets))
