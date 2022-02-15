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
from .tensorflow import TF1ODAPIFrozenModel


class MegaDetectorV4_1(TF1ODAPIFrozenModel):
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
    model_url = "https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/md_v4.1.0/md_v4.1.0.pb"
    model_hash = ""
    model_path = Path("~").expanduser() / "Downloads" / "md_v4.1.0.pb"


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
    model_url = "https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/megadetector_v3.pb"
    model_hash = ""
    model_path = Path("~").expanduser() / "Downloads" / "megadetector_v3.pb"


class MegaDetectorV2(TF1ODAPIFrozenModel):
    """
    MegaDetector v2.0
    """

    class_map = {
        1: "animal",
    }

    model_name = "megadetector"
    model_version = "v2.0.0"
    model_url = "https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/megadetector_v2.pb"
    model_hash = ""
    model_path = Path("~").expanduser() / "Downloads" / "megadetector_v2.pb"
