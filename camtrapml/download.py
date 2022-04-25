"""
Download / Cache Utilities
"""

from pathlib import Path
from hashlib import md5
from requests import get
from tqdm import tqdm

CACHE_HOME = Path.home() / ".camtrapml"


def download(url: str, path: Path, md5_hash: str) -> None:
    """
    Downloads a file from a URL to a path.
    """

    if path.exists() and (md5_hash == "" or md5_hash == hash_file(path)):
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    resp = get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))

    with path.open("wb") as file, tqdm(
        desc="Downloading Model",
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)

    if md5_hash != "":
        assert md5_hash == hash_file(path)


def hash_file(path: Path) -> str:
    """
    Hashes a file.
    """

    with open(path, "rb") as file_handle:
        return md5(file_handle.read()).hexdigest()
