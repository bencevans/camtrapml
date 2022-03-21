"""
General image processing functions.
"""

from typing import Tuple, Union
from pathlib import Path
from urllib.parse import urlparse
from io import BytesIO
from PIL import Image
from requests import get


def load_image(path: Union[Path, str]) -> Image.Image:
    """
    Loads an image from a path.
    """

    if isinstance(path, str) and urlparse(path).scheme in ["http", "https"]:
        return Image.open(BytesIO(get(path).content))

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
        load_image(path)
        return True
    except Exception:# pylint: disable=W0703
        return False
