"""
Command Line Utility for batch detection.
"""

from argparse import ArgumentParser
from json import dump
from datetime import datetime
from tqdm import tqdm
from camtrapml.dataset import ImageDataset
from camtrapml.detection.models.megadetector import (
    MegaDetectorV2,
    MegaDetectorV3,
    MegaDetectorV4_1,
)


def parse_args():
    """
    Parse command line arguments.
    """

    parser = ArgumentParser()
    parser.add_argument(
        "model",
        type=str,
        help="Detection model to utilise [md2, md3, md4]",
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to directory containing the images",
    )
    parser.add_argument("output_path", type=str, help="Path to store the JSON output")

    parser.add_argument("--output_relative_filenames", action="store_true")

    return parser.parse_args()


def get_model(model_name):
    """
    Load a detection model based on a short name.
    """

    if model_name == "md2":
        return MegaDetectorV2()

    if model_name == "md3":
        return MegaDetectorV3()

    if model_name == "md4":
        return MegaDetectorV4_1()

    raise ValueError(f"Unknown model {model_name}")


def detection_to_json_types(detection):
    """
    Convert a detection to a JSON-compatible dictionary.
    """
    detection["conf"] = float(detection["conf"])
    detection["bbox"] = [float(x) for x in detection["bbox"]]
    detection["category"] = str(detection["category"])
    return detection


def batch_detection():
    """
    Run detection on a batch of images.
    """

    args = parse_args()

    model = get_model(args.model)

    print("Enumerating images...", end="")
    dataset = ImageDataset(args.dataset_path)
    image_paths = list(tqdm(dataset.enumerate_images()))
    print(" Done")

    results = {"images": []}

    for image_path in tqdm(image_paths):
        if args.output_relative_filenames:
            output_image_path = str(image_path.relative_to(args.dataset_path))
        else:
            output_image_path = str(image_path)

        prediction = model.detect(image_path)

        results["images"].append(
            {
                "file": output_image_path,
                "detections": [
                    detection_to_json_types(detection) for detection in prediction
                ],
            }
        )

    results["detection_categories"] = {"1": "animal", "2": "person", "3": "vehicle"}

    results["info"] = {
        "detection_completion_time": str(datetime.now()),
        "format_version": "1.0",
    }

    with open(args.output_path, "w", encoding="utf-8") as file_handle:
        dump(results, file_handle, indent=2)


if __name__ == "__main__":
    batch_detection()
