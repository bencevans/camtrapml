"""
General image processing functions.
"""

from typing import Tuple, Union
from PIL import Image
from pathlib import Path


def load_image(path: Union[Path, str]) -> Image.Image:
    """
    Loads an image from a path.
    """
    return Image.open(path)


def thumbnail(image: Image.Image, size: Tuple[int, int] = (400, 400)) -> Image.Image:
    """
    Generates a thumbnail of an image keeping aspect ratio. The size argument
    is the maximum size of the thumbnail.
    """
    image = image.copy()
    image.thumbnail(size, Image.ANTIALIAS)
    return image


def is_image(path: Union[Path, str]) -> bool:
    """
    Checks if a path is an image, or at least one that can be read with Pillow.
    """
    try:
        Image.open(path)
        return True
    except:
        return False
