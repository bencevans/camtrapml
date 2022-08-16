from pathlib import Path
from typing import Union

import yolov5

from camtrapml.download import download, hash_file
from camtrapml.image.utils import ImageSource, load_image


class YOLOV5:
    """
    YOLOV5
    """

    class_map = {}

    model_path = None
    model_version = None
    model_url = ""

    def __init__(
        self,
        model_path: Union[Path, str, None] = None,
        model_url: str = "",
        model_hash: str = "",
        class_map: Union[dict, None] = None,
    ) -> None:
        """
        Initialises a Tensorflow V1 Object Detection Frozen Graph Model.

        Args:

          model_path: Path to the frozen model .pb file.
        """

        if model_path:
            self.model_path = model_path

        if model_url:
            self.model_url = model_url

        if model_hash:
            self.model_hash = model_hash

        if class_map:
            self.class_map = class_map

        self._session = None
        self._tensors = None

        if Path(self.model_path).exists() is False and self.model_url != "":
            download(self.model_url, self.model_path, "")
        elif (
            Path(self.model_path).exists() is True
            and model_hash in dir(self)
            and self.model_hash != ""
        ):
            local_hash = hash_file(self.model_path)
            if local_hash != self.model_hash:
                raise ValueError(
                    f"Hash mismatch for model {self.model_path}. Local hash: {local_hash}."
                )

    def load_model(self,) -> None:
        """
        Loads the model from the frozen model .pb file.
        """
        self._model = yolov5.load(self.model_path)

        self._model.conf = 0.25  # NMS confidence threshold
        self._model.iou = 0.45  # NMS IoU threshold
        self._model.agnostic = False  # NMS class-agnostic
        self._model.multi_label = False  # NMS multiple labels per box
        self._model.max_det = 1000  # maximum number of detections per image


    def close(self) -> None:
        """
        Closes the model.
        """
        del self._model

    def detect(self, image_source: ImageSource, min_score: float = 0.1) -> list:
        """
        Detects objects in the image.

        Args:

          image_source: PIL Image, Path to image or URL to image.
          min_score: Minimum score to consider a detection.

        Returns:

          A list of tuples containing the bounding box coordinates and the class
          of the detected object.
        """

        # Load the detection graph once.
        if self._model is None:
            self.load_model()

        # Load the image.
        results = self._model(image_source).tolist()[0]

        # print(dir())

        ouputs = []
        
        for detection in results.xywhn:
            detection = detection.tolist()[0]
            print(detection)
            bbox = detection[:4]
            conf = detection[4]
            class_id = int(detection[5])

            if conf < min_score:
                continue

            ouputs.append({
                "bbox": bbox,
                "category": class_id,
                "conf": conf,
            })
        
        return ouputs



    def __enter__(self):
        self.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# model = YOLOV5(
#     model_path=Path("/home/bencevans/Projects/yolov5/md_v5a.0.0.torchscript"),
# )

# with model:
#     boxes = model.detect(
#         image_source=Path("/pool0/datasets/ena24/ena24/435.jpg"),
#         min_score=0.1,
#     )

#     print(boxes)
