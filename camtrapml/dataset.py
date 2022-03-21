"""
Dataset Handling
"""

from pathlib import Path
from os import walk

from camtrapml.image.utils import is_image


class ImageDataset:
    """
    A dataset is a collection of images.
    """
    name: str
    path: Path

    def __init__(self, name: str, path: Path):
        self.name = name
        self.path = Path(path).expanduser()
        self.image_set = None
        self.detection_set = None

        if not self.path.exists() or not self.path.is_dir():
            raise ValueError(f"{self.path} is not a directory")

    def enumerate_images(self):
        """
        Enumerates all images in the dataset.
        """
        for root, _, files in walk(self.path):
            for file in files:
                if is_image(Path(root) / file):
                    yield Path(root) / file

    @staticmethod
    def from_coco(source):
        """
        Load a Dataset Instance from COCO-CameraTrap
        """
        raise NotImplementedError()
