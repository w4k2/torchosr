from os import listdir
from os.path import join
from typing import Any, Callable, List, Optional, Tuple
import glob 
import numpy as np

from PIL import Image

from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive


class Omniglot_base(VisionDataset):
    """
    `Omniglot <https://github.com/brendenlake/omniglot>`_ Dataset.
    
    Base implementation from torch library.
    """

    folder = "omniglot-py"
    download_url_prefix = "https://raw.githubusercontent.com/brendenlake/omniglot/master/python"
    zips_md5 = {
        "images_background": "68d2efa1b9178cc56df9314c21c6e718",
        "images_evaluation": "6b91aef0f799c5bb55b94e3f2daec811",
    }

    def __init__(
        self,
        root: str,
        background: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
    ) -> None:
        super().__init__(join(root, self.folder), transform=transform, target_transform=target_transform)
        self.background = background

        if download:
            self.download()
            
        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        self.data, self.targets = self._load_data()
        

    def _load_data(self):
        self.target_folder = join(self.root, self._get_target_folder())
        self._alphabets = listdir(self.target_folder)
        try:
            self._alphabets.remove('.DS_Store')
        except:
            pass

        self._characters: List[str] = sum(
            ([join(a, c) for c in listdir(join(self.target_folder, a))] for a in self._alphabets), []
        )

        self._character_images = [
            [(image, idx) for image in glob.glob(join(self.target_folder, character, '*.png'))]
            
            for idx, character in enumerate(self._characters)
        ]
        
        self._flat_character_images: List[Tuple[str, int]] = sum(self._character_images, [])
        
        data = [Image.open(i, mode="r").convert("L") for i, _ in self._flat_character_images]
        targets = [i for _,i in self._flat_character_images]
        
        return data, targets
    
    def __len__(self) -> int:
        return len(self._flat_character_images)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image, target = self.data[index], self.targets[index]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            target = self.target_transform(target)

        return image, target

    def _check_integrity(self) -> bool:
        zip_filename = self._get_target_folder()
        if not check_integrity(join(self.root, zip_filename + ".zip"), self.zips_md5[zip_filename]):
            return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        filename = self._get_target_folder()
        zip_filename = filename + ".zip"
        url = self.download_url_prefix + "/" + zip_filename
        download_and_extract_archive(url, self.root, filename=zip_filename, md5=self.zips_md5[filename])

    def _get_target_folder(self) -> str:
        return "images_background" if self.background else "images_evaluation"
    
    def _n_classes(self) -> int:
        return len(np.unique(self.targets))