import os.path
from typing import Any, Callable, Optional, Tuple
import numpy as np
from PIL import Image
import scipy.io as sio

from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity, download_url



class SVHN_base(VisionDataset):
    """
    `SVHN <http://ufldl.stanford.edu/housenumbers/>`_ Dataset.
        
    Base implementation from torch library. No split to train/test data.
    """

    split_list = {
        "train": [
            "http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
            "train_32x32.mat",
            "e26dedcc434d2e4c54c9b2d4a06d8373",
        ],
        "test": [
            "http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
            "test_32x32.mat",
            "eb5a983be6a315427106f1b164d9cef3",
        ],
        # "extra": [
        #     "http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
        #     "extra_32x32.mat",
        #     "a93ce644f1a588dc4d68dda5feec44a7",
        # ],
    }

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        # reading(loading) mat file as array
        loaded_mat = [sio.loadmat(os.path.join(self.root, self.split_list[split][1])) for split in self.split_list.keys()]
        
        self.data = []
        self.labels = []
        
        for lm in loaded_mat:
            data = lm["X"]
            # loading from the .mat file gives an np array of type np.uint8
            # converting to np.int64, so that we have a LongTensor after
            # the conversion from the numpy array
            # the squeeze is needed to obtain a 1D tensor
            labels = lm["y"].astype(np.int64).squeeze()

            # the svhn dataset assigns the class label "10" to the digit 0
            # this makes it inconsistent with several loss functions
            # which expect the class labels to be in the range [0, C-1]
            np.place(labels, labels == 10, 0)
            data = np.transpose(data, (3, 2, 0, 1))
            
            self.data.append(data)
            self.labels.append(labels)
        
        self.data = np.concatenate(self.data)
        self.labels = np.concatenate(self.labels)
        

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], int(self.labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)
    
    def _n_classes(self) -> int:
        return len(np.unique(self.labels))

    def _check_integrity(self) -> bool:
        root = self.root
        for split in self.split_list.keys():
            md5 = self.split_list[split][2]
            fpath = os.path.join(root, self.split_list[split][1])
            return check_integrity(fpath, md5)

    def download(self) -> None:
        for split in self.split_list.keys():
            md5 = self.split_list[split][2]
            download_url(self.split_list[split][0], self.root, self.split_list[split][1], md5)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)