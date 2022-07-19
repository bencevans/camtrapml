"""
Tools for extracting EXIF data from images.
"""

from pathlib import Path
from typing import List, Union
from exiftool import ExifTool
from tqdm.contrib.concurrent import process_map


def extract_exif(path: Union[Path, str]) -> dict:
    """
    Extracts EXIF data from an image.
    """
    with ExifTool() as exiftool:
        return exiftool.get_metadata(filename=str(path))


def extract_multiple_exif(paths: List[Union[Path, str]]) -> list:
    """
    Extracts EXIF data from multiple images.
    """
    with ExifTool() as exiftool:
        return exiftool.get_metadata_batch([str(path) for path in paths])


def extract_multiple_exif_fast(paths: List[Union[Path, str]], batch_size=100) -> list:
    """
    Extracts EXIF data from a list of images by spreading the load across
    multiple processes making use of multiple CPU cores.
    """
    if len(paths) <= batch_size:
        batch_size = len(paths)

    paths = [str(path) for path in paths]
    batched_paths = list(zip(*[iter(paths)] * batch_size))

    exif_data = process_map(
        extract_multiple_exif, batched_paths, desc="Extracting EXIF data", chunksize=3
    )

    # Unbatch
    return [item for sublist in exif_data for item in sublist]

def parse_iptc(keywords) -> dict:
    """
    Parses IPTC Tags
    """

    if isinstance(keywords, str):
        keywords = [keywords]

    def parse(keyword):
        split = keyword.split(':')
        if len(split) == 1:
            return split[0], True

        key = split[0].strip()
        val = split[1].strip()

        return key, val

    return [parse(keyword) for keyword in keywords]
