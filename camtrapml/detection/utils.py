"""
Detection Utilities
"""
from typing import Union
from PIL import Image, ImageDraw, ImageFont
from font_fredoka_one import FredokaOne  # pylint: disable=E0611
from ..image.utils import ImageSource, load_image

def render_detections(
    image_source: ImageSource,
    detections,
    draw_box=True,
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

    image = load_image(image_source)
    draw = ImageDraw.Draw(image)

    for detection in detections:
        min_y, min_x, max_y, max_x = detection["bbox"]

        if draw_box:
            draw.rectangle(
                (
                    (min_x * image.size[0]),
                    (min_y * image.size[1]),
                    (max_x * image.size[0]),
                    (max_y * image.size[1]),
                ),
                outline="red",
                width=3,
            )

        if class_map or draw_score:
            font = ImageFont.FreeTypeFont(FredokaOne, size=40)
            class_name = (
                class_map[detection["category"]]
                if class_map and detection["category"] in class_map
                else str(detection["category"])
            )
            text = class_name if class_map else ""
            text += f" {detection['conf']:.2f}" if draw_score else ""
            draw.text(
                (
                    (min_x * image.size[0]),
                    (min_y * image.size[1]) - 50,
                ),
                text,
                fill="red",
                font=font,
            )

    return image


def remove_detections_from_image(image: Image.Image, detections):
    """
    Removes the detections from an image.

    Args:

      image: Image to remove the detections from.
      detections: List of detections to remove.

    Returns:

      The rendered image as PIL.Image.
    """

    image = image.copy()
    image = image.convert("RGB")

    for detection in detections:
        min_y, min_x, max_y, max_x = detection["bbox"]
        image_width, image_height = image.size

        cover_size = (
            int(image_width * (max_x - min_x)),
            int(image_height * (max_y - min_y)),
        )
        cover_position = (int(image_width * min_x), int(image_height * min_y))

        cover = Image.new("RGBA", cover_size, (0, 0, 0, 0))

        image.paste(cover, cover_position)

    return image


def extract_detections_from_image(image: Image.Image, detections):
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
        min_y, min_x, max_y, max_x = detection["bbox"]
        image_width, image_height = image.size

        yield image.crop(
            (
                (min_x * image_width),
                (min_y * image_height),
                (max_x * image_width),
                (max_y * image_height),
            )
        )
