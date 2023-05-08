from os.path import join
from typing import Any, Tuple
import numpy as np
from ..utils.base import onehot_bg
from torchvision.datasets import VisionDataset


class OutlierDataset(VisionDataset):
    """
    OutlierDataset for combining instances from (known) training and (unknown) outlier data.
    
    :type root: string
    :param root: Data directory.
    
    :type dataset: VisionDataset
    :param dataset: Dataset of known-class testing examples.

    :type outliers: VisionDataset
    :param outliers: Dataset of unknown-class testing examples, which will be labeled as unknowns.
    
    :type shuffle: boolean
    :param shuffle: If True, the final data will be shuffled.
    
    :type random_state: int
    :param random_state: Random state (for shuffle).
    
    :type unknown_label: int
    :param unknown_label: Label with which the unknowns will be marked.

    :type onehot: boolean
    :param onehot: If True will perform one-hot encoding on labels.
    
    :type onehot_num_classes: int
    :param onehot_num_classes: Number of classes for one-hot encoding (in case outlier data is generated for testing). 
    """
    def __init__(
        self,
        root: str,
        dataset,
        outliers,
        shuffle: bool = True,
        random_state: int = None,
        unknown_label: int = None,
        onehot = False,
        onehot_num_classes = None
        ) -> None:
        super().__init__(join(root))
        
        self.dataset = dataset
        self.outliers = outliers
        self.shuffle = shuffle
        self.unknown_label = unknown_label
        self.onehot = onehot
        self.onehot_num_classes = onehot_num_classes
        
        if random_state is not None:
            np.random.seed(random_state)

        #Combine two datasets
        data = []
        targets = []
        for i in range(len(self.dataset)):
            d, t  = self.dataset.__getitem__(i)
            data.append(d)
            targets.append(t)
        for i in range(len(self.outliers)):
            d, t = self.outliers.__getitem__(i)
            data.append(d)
            targets.append(t)
            
        if self.shuffle:
            order = np.random.permutation(len(targets))
            data = np.array(data)[order]
            targets = np.array(targets)[order]
            
        self.data = data 
        self.targets = targets
    
    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        d, t = self.data[index], self.targets[index]
        if self.onehot:
            transform = onehot_bg(self.onehot_num_classes)
            t = transform(t)
        return d,t

    def _num_classes(self):
        return len(np.unique(self.targets))