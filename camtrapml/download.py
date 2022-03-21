from pathlib import Path

CACHE_HOME = Path.home() / ".camtrapml"


def download(url: str, path: Path, hash: str) -> None:
    """
    Downloads a file from a URL to a path.
    """
    from requests import get
    from tqdm import tqdm

    if path.exists() and (hash == "" or hash == hash(path)):
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
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

    if hash != "":
        assert hash == hash(path)


def hash(path: Path) -> str:
    """
    Hashes a file.
    """
    import hashlib

    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()
