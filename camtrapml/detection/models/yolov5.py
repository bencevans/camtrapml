from pathlib import Path
from typing import Union

from torch import jit
from torchvision.transforms.functional import pil_to_tensor
from camtrapml.detection.models.base import BaseDetector

from camtrapml.download import download, hash_file
from camtrapml.image.utils import ImageSource, load_image


class YOLOV5(BaseDetector):
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
        self._model = jit.load(self.model_path)

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
        image = load_image(image_source, mode="RGB")
        image = image.resize((640, 640))
        image = pil_to_tensor(image)
        image = image.float()
        image = image.unsqueeze(0)

        self._model.eval()
        result = self._model(image)

        # Get the bounding boxes and class scores.
        results = result[0][0].detach().numpy()
        boxes = results[:, :4] / 640
        boxes = boxes.tolist()
        scores = results[:, 4].tolist()

        detections = []

        for i in range(0, len(scores)):
            if scores[i] < min_score:
                continue

            detections.append({
                # "category": self.class_map[int(box[5])],
                "score": scores[i],
                "bbox": boxes[i],
            })

        return detections

    def __enter__(self):
        self.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


model = YOLOV5(
    model_path=Path("/home/bencevans/Projects/yolov5/md_v5a.0.0.torchscript"),
)

with model:
    boxes = model.detect(
        image_source=Path("/pool0/datasets/ena24/ena24/435.jpg"),
        min_score=0.1,
    )

    print(boxes)
