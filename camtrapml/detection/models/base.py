from pathlib import Path
from typing import Union

from camtrapml.image.utils import ImageSource


class BaseDetector:
    def __init__(
        self,
        model_path: Union[Path, str, None] = None,
        model_url: str = "",
        model_hash: str = "",
        class_map: Union[dict, None] = None,
    ) -> None:
        self.model_path = model_path
        self.model_url = model_url
        self.model_hash = model_hash
        self.class_map = class_map

    def load_model(self,) -> None:
        raise NotImplementedError()

    def close(self) -> None:
        raise NotImplementedError()

    def detect(self, image_source: ImageSource, min_score: float = 0.1) -> list:
        raise NotImplementedError()

    def __enter__(self):
        raise NotImplementedError()

    def __exit__(self, exc_type, exc_val, exc_tb):
        raise NotImplementedError()
