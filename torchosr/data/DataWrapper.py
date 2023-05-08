from typing import Any, Tuple
import numpy as np
import torch
import copy
from ..utils.base import onehot_bg
from itertools import compress
from operator import itemgetter 

from torchvision.datasets import VisionDataset

class DataWrapper(VisionDataset):
    """
    DataWrapper for base datasets.
    
    :type root: string
    :param root: Data directory.
    
    :type base_dataset: VisionDataset
    :param base_dataset: base dataset implementing __getitem__ function.
    
    :type indexes: List
    :param indexes: Indexes of wrapped dataset objects that will be returned.
    
    :type get_classes: List
    :param get_classes: Considered class indexes from base dataset (known + unknown).
    
    :type known_classes: List
    :param known_classes: Considered known class indexes from base dataset.
    
    :type return_only_known: boolean
    :param return_only_known: If True will return only known instances (for training). If False will assign new class index (equal to the number of classes) to unknown samples, and return them with known data (for testing).

    :type onehot: boolean
    :param onehot: If True will perform one-hot encoding on labels.
    
    :type onehot_num_classes: int
    :param onehot_num_classes: Number of classes for one-hot encoding (in case outlier data is generated for testing). 
    """
    
    def __init__(
        self,
        root: str,
        base_dataset,
        indexes, # for k-fold
        get_classes, #known + unknown
        known_classes,
        return_only_known, # for training
        onehot = False,
        onehot_num_classes = None
    ) -> None:
        super().__init__(root)
        
        self.base_dataset = base_dataset
        self.indexes = indexes
        self.get_classes = get_classes
        self.known_classes = known_classes
        self.return_only_known = return_only_known
        self.onehot = onehot
        self.onehot_num_classes = onehot_num_classes
        
        self.new_index_mask = None
        
        #Get initial dataset
        self.data, self.targets = [], []
        for i in range(len(self.base_dataset)):
            d, t = self.base_dataset.__getitem__(i)
            if t not in self.get_classes:
                continue
            self.data.append(d)
            self.targets.append(t)
            
        #Select known        
        self.targets = torch.tensor(self.targets)
        self.original_targets = copy.deepcopy(self.targets)

        is_known = torch.any(torch.stack([torch.eq(self.targets, aelem).logical_or_(torch.eq(self.targets, aelem)) for aelem in self.known_classes], dim=0), dim = 0)

        # Reorder all the known classes
        for new_label, label in enumerate(np.sort(self.known_classes)):
            self.targets[self.targets==label] = new_label
        
        # Set all the unknown as the last class
        self.targets[~is_known] = len(self.known_classes)

        # Remove unknown if return_only_known
        if self.return_only_known:
            self.data = list(compress(self.data, is_known))
            self.targets = list(compress(self.targets, is_known))
            self.original_targets = list(compress(self.original_targets, is_known))

        #Get with specific indexes
        if self.indexes != 'all':
            self.data = itemgetter(*indexes)(self.data)
            self.targets = itemgetter(*indexes)(self.targets)
            self.original_targets = itemgetter(*indexes)(self.original_targets)
            
    def reindex(self, new_index_mask):
        self.new_index_mask = new_index_mask
            
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if self.new_index_mask is not None:
            index = self.new_index_mask[index]
            
        d, t = self.data[index], self.targets[index]
            
        if self.onehot:
            transform = onehot_bg(self.onehot_num_classes)
            t = transform(t)
        return d,t

    def __len__(self) -> int:
        if self.new_index_mask is not None:
            return len(self.new_index_mask)

        return len(self.targets)

    def _num_classes(self):
        return len(np.unique(self.targets))
    