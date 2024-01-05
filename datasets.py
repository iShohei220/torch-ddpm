import os
from typing import Any, Callable, Optional, Tuple

from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader


class CelebAHQ(VisionDataset):
    def __init__(
        self, 
        root: str, 
        transform: Optional[Callable] = None, 
    ) -> None:
        super().__init__(root, None, transform, None)
        root = os.path.expanduser(root)
        fnames = os.listdir(root)
        self.paths = [os.path.join(root, fname) for fname in fnames]

    def __getitem__(self, index: int) -> Tuple[Any, None]:
        path = self.paths[index]
        sample = default_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, None

    def __len__(self) -> int:
        return len(self.paths)
