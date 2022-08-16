"""
Dataset Handling
"""

from typing import Union
from pathlib import Path
from os import walk
from PIL import Image
from camtrapml.image.utils import is_image


class ImageDataset:
    """
    A dataset is a collection of images.
    """

    name: str
    path: Path

    def __init__(self, path: Path, name: Union[None, str] = None):
        self.name = name
        self.path = Path(path).expanduser()
        self.image_set = None
        self.detection_set = None

        if not self.path.exists() or not self.path.is_dir():
            raise ValueError(f"{self.path} is not a directory")

    def enumerate_images(self, enhanced_validation: bool = False):
        """
        Enumerates all images in the dataset.
        """

        supported_extensions = {
            extention.lower() for extention, image_format in
            Image.registered_extensions().items()
            if image_format in Image.OPEN
        }

        for root, _, files in walk(self.path):
            for file in files:
                file_path = Path(root) / file

                if enhanced_validation:
                    if is_image(file_path):
                        yield file_path

                else:
                    if file_path.suffix.lower() in supported_extensions:
                        yield file_path

    @staticmethod
    def from_coco(source):
        """
        Load a Dataset Instance from COCO-CameraTrap
        """
        raise NotImplementedError()
