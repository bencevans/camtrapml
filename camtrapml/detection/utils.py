"""
Detection Utilities
"""
from pathlib import Path
from typing import Union
from PIL import Image, ImageDraw, ImageFont


def render_detections(
    image_path: Union[Path, str],
    detections,
    draw_box=True,
    draw_label=True,
    draw_score=True,
    class_map: Union[dict, None] = None,
):
    """
    Renders the image with the detections.

    Args:

      image_path: Path to the image to render the detections on.
      detections: List of tuples containing the bounding box coordinates and the
        class of the detected object.

    Returns:

      The rendered image as PIL.Image.
    """

    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    for detection in detections:
        x_min, y_min, x_max, y_max = detection["bbox"]
        image_width, image_height = image.size

        if draw_box:
            draw.rectangle(
                [
                    (y_min * image_width),
                    (x_min * image_height),
                    (y_max * image_width),
                    (x_max * image_height),
                ],
                outline="red",
                width=3,
            )

        if draw_label or draw_score:
            font = ImageFont.FreeTypeFont(
                "/System/Library/Fonts/Supplemental/Verdana.ttf", size=40
            )
            class_name = (
                class_map[detection["category"]]
                if class_map and detection["category"] in class_map
                else str(detection["category"])
            )
            text = class_name if draw_label else ""
            text += " {:.2f}".format(detection["conf"]) if draw_score else ""
            draw.text(
                (
                    (y_min * image_width),
                    (x_min * image_height) - 50,
                ),
                text,
                fill="red",
                font=font,
            )

    return image


def remove_detections_from_image(image: Image, detections):
    """
    Removes the detections from an image.

    Args:

      image: Image to remove the detections from.
      detections: List of detections to remove.

    Returns:

      The rendered image as PIL.Image.
    """

    image = image.copy()
    image = image.convert('RGB')

    for detection in detections:
        x_min, y_min, x_max, y_max = detection["bbox"]
        image_width, image_height = image.size


        cover_size = (int(image.width * (y_max - y_min)), int(image.height * (x_max - x_min)))
        cover_position = (int(image.width * y_min), int(image.height * x_min))

        cover = Image.new("RGBA", cover_size, (0, 0, 0, 0))

        image.paste(cover, cover_position)


    return image

def extract_detections_from_image(image: Image, detections):
    """
    Extracts the detections from an image.

    Args:

      image: Image to extract the detections from.
      detections: List of detections to extract.

    Returns:

      The rendered image as PIL.Image.
    """

    image_width, image_height = image.size

    for detection in detections:
        x_min, y_min, x_max, y_max = detection["bbox"]
        image_width, image_height = image.size

        yield image.crop(
            [
                (y_min * image_width),
                (x_min * image_height),
                (y_max * image_width),
                (x_max * image_height),
            ]
        )
