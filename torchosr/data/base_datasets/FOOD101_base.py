import json
from pathlib import Path
from typing import Any, Tuple, Callable, Optional, Dict
import numpy as np
from PIL import Image
import os
import pandas as pd
from torchvision.datasets.utils import verify_str_arg, download_and_extract_archive
from torchvision.datasets import VisionDataset



class FOOD101_base(VisionDataset):
    """
    Base implementation from torch library. No split to train/test data.
    """

    _URL = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
    _MD5 = "85eeb15f3717b99a5da872d97d918f87"
    files = ["train.json", "test.json"]
    
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._base_folder = Path(self.root) / "food-101"
        self._meta_folder = self._base_folder / "meta"
        self._images_folder = self._base_folder / "images"

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")


        def merge_JsonValues():
            json_files = [file for file in os.listdir('../data/food-101/meta') if file.endswith('.json')]
            with open(self._meta_folder / json_files[0]) as f1:
                data1 = json.loads(f1.read())
            with open(self._meta_folder / json_files[1]) as f2:
                data2 = json.loads(f2.read())
            for key in data1:
                data1[key].extend(data2[key])
            return data1            

        
        self._labels = []
        self._image_files = []

        metadata = merge_JsonValues()

        self.classes = sorted(metadata.keys())
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

        for class_label, im_rel_paths in metadata.items():
            self._labels += [self.class_to_idx[class_label]] * len(im_rel_paths)
            self._image_files += [
                self._images_folder.joinpath(*f"{im_rel_path}.jpg".split("/")) for im_rel_path in im_rel_paths
            ]

    def __len__(self) -> int:
        return len(self._image_files)


    def __getitem__(self, idx) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        image = Image.fromarray(image_file.numpy(), mode="L")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label


    def _check_exists(self) -> bool:
        return all(folder.exists() and folder.is_dir() for folder in (self._meta_folder, self._images_folder))

    def download(self) -> None:
        if self._check_exists():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self._URL, download_root=self.root, md5=self._MD5)

    def _n_classes(self) -> int:
        return len(np.unique(self.classes))
    