import json
from pathlib import Path
from os import walk


class Dataset:
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
        for root, _, files in walk(self.path):
            for file in files:
                if file.lower().endswith(".jpg"):
                    yield Path(root) / file


class Detection:
    def __init__(
        self,
        image_path: Path,
        label: str,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
    ):
        self.image_path = image_path
        self.label = label
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    def __str__(self):
        return f"{self.image_path} {self.label} {self.xmin} {self.ymin} {self.xmax} {self.ymax}"


class DetectionSet:
    def __init__(self, detections):
        self.detections = detections

    def from_megadetector_file(self, json_path: Path):
        with json_path.open() as f:
            data = json.load(f)

        for image_data in data["images"]:
            image_path = Path(image_data["path"])
            for annotation in image_data["annotations"]:
                label = annotation["label"]
                xmin = annotation["xmin"]
                ymin = annotation["ymin"]
                xmax = annotation["xmax"]
                ymax = annotation["ymax"]
                detection = Detection(image_path, label, xmin, ymin, xmax, ymax)
                self.detections.append(detection)


