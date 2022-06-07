"""
General image processing functions.
"""

from typing import Tuple, Union
from pathlib import Path
from urllib.parse import urlparse
from io import BytesIO
from PIL import Image
from requests import get

ImageSource = Union[Image.Image, Path, str]


def load_image(source: ImageSource, mode="RGB", timeout=10) -> Image.Image:
    """
    Loads an image from a path or url or returns a copy of a previously loaded image.
    """

    if isinstance(source, str) and urlparse(source).scheme in ["http", "https"]:
        image = Image.open(BytesIO(get(source, timeout=timeout).content))

    elif isinstance(source, Image.Image):
        image = source.copy()

    else:
        image = Image.open(source)

    if mode is not None:
        image = image.convert(mode)

    return image


def thumbnail(source: ImageSource, size: Tuple[int, int] = (400, 400)) -> Image.Image:
    """
    Generates a thumbnail of an image keeping aspect ratio. The size argument
    is the maximum size of the thumbnail.
    """
    image = load_image(source)
    image.thumbnail(size, Image.ANTIALIAS)
    return image


def is_image(path: ImageSource) -> bool:
    """
    Checks if a path is an image, or at least one that can be read with Pillow.
    """
    try:
        load_image(path)
        return True
    except Exception:  # pylint: disable=W0703
        return False
