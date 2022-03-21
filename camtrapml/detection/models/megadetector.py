"""
MegaDetector, developed by Microsoft and collaborators aims to be a
generlisable detection model for detecting animals, humans and vehicles in
camera trap imagery.

More information can be found on Microsoft's
[CameraTrap GitHub Repository](https://github.com/microsoft/CameraTraps/blob/main/megadetector.md).

This module contains just enough code to run each of MegaDetector versions
where each version is provided with a consistent API.


    from camtrapml.detection.models.megadetector import MegaDetectorV4_1

    with MegaDetectorV4_1() as detector:
        # Run detection on a single image
        detections = detector.detect(image)

"""
from pathlib import Path
from typing import Tuple, Union
from json import load
from .tensorflow import TF1ODAPIFrozenModel
from ...download import CACHE_HOME

MODEL_FILES_ROUTE = "https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/"


class MegaDetectorV4_1(TF1ODAPIFrozenModel):  # pylint: disable=C0103
    """
    MegaDetector v4.1
    """

    class_map = {
        1: "animal",
        2: "human",
        3: "vehicle",
    }

    model_name = "megadetector"
    model_version = "v4.1.0"
    model_url = MODEL_FILES_ROUTE + "md_v4.1.0/md_v4.1.0.pb"
    model_hash = "7008707c66787a4d1c6544c27523f714"
    model_path = CACHE_HOME / "models" / "megadetector" / "v4.1.0" / "md_v4.1.0.pb"


class MegaDetectorV3(TF1ODAPIFrozenModel):
    """
    MegaDetector v3.0
    """

    class_map = {
        1: "animal",
        2: "human",
    }

    model_name = "megadetector"
    model_version = "v3.0.0"
    model_url = MODEL_FILES_ROUTE + "megadetector/megadetector_v3.pb"
    model_hash = "1a4ca31abf9dca286ff1641eedbccc3d"
    model_path = CACHE_HOME / "models" / "megadetector" / "v3.0.0" / "md_v3.0.0.pb"


class MegaDetectorV2(TF1ODAPIFrozenModel):
    """
    MegaDetector v2.0
    """

    class_map = {
        1: "animal",
    }

    model_name = "megadetector"
    model_version = "v2.0.0"
    model_url = MODEL_FILES_ROUTE + "/megadetector/megadetector_v2.pb"
    model_hash = "3a634a610d9a74f671cef08f75cfd661"
    model_path = CACHE_HOME / "models" / "megadetector" / "v2.0.0" / "md_v2.0.0.pb"


def read_megadetector_batch_file(
    path: Union[Path, str], image_dir
) -> Tuple[list, list]:
    """
    Reads a batch file from a MegaDetector model.

    Args:
        path: Path to the batch file.
        image_dir: Path to the directory containing the images.

    Returns:
        A tuple of (image_paths, image_detections)
    """

    image_paths = []
    image_detections = []

    with open(path, "r", encoding="UTF-8") as file_handle:
        images = load(file_handle)["images"]

    for image_data in images:
        image_path = image_dir / image_data["file"]
        image_paths.append(image_path)
        image_detections.append(image_data["detections"])

    return image_paths, image_detections
