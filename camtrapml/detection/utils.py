"""
Detection Utilities
"""
from PIL import Image, ImageDraw, ImageFont


def render_detections(
    image_path,
    detections,
    draw_box=True,
    draw_label=True,
    draw_score=True,
    class_map=None,
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
