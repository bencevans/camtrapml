import platform
import numpy as np
import tensorflow as tf

from logging import warn
from typing import Union
from pathlib import Path
from PIL import Image


class TF1ODAPIFrozenModel:
    """
    Tensorflow v1 (loaded using v2) Object Detection API Frozen Model Loader
    """

    class_map = None

    model_path = None
    model_version = None
    model_url = ""
    model_image_tensor_name = "image_tensor:0"
    model_boxes_tensor_name = "detection_boxes:0"
    model_scores_tensor_name = "detection_scores:0"
    model_classes_tensor_name = "detection_classes:0"

    _detection_graph = None
    _session = None
    _image_tensor = None
    _box_tensor = None
    _score_tensor = None
    _classes_tensor = None

    def __init__(
        self,
        model_path: Union[Path, str] = None,
        model_image_tensor_name: str = None,
        model_boxes_tensor_name: str = None,
        model_scores_tensor_name: str = None,
        model_classes_tensor_name: str = None,
        class_map: dict = None,
    ) -> None:
        """
        Initialises a Tensorflow V1 Object Detection Frozen Graph Model.

        Args:

          model_path: Path to the frozen model .pb file.
        """

        if model_path:
            self.model_path = model_path

        if model_image_tensor_name:
            self.model_image_tensor_name = model_image_tensor_name

        if model_boxes_tensor_name:
            self.model_boxes_tensor_name = model_boxes_tensor_name

        if model_scores_tensor_name:
            self.model_scores_tensor_name = model_scores_tensor_name

        if model_classes_tensor_name:
            self.model_classes_tensor_name = model_classes_tensor_name

        if class_map:
            self.class_map = class_map

    def load_model(self) -> None:
        """
        Loads the model from the frozen model .pb file.
        """

        # Disable the GPU on M1 Macs as there appears to be a bug where the outputs are
        # not the same as the reference implementation.
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            warn("Disabling GPU on M1 Macs")
            tf.config.set_visible_devices([], "GPU")

        detection_graph = tf.compat.v1.Graph()

        with detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.compat.v1.gfile.GFile(str(self.model_path), "rb") as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name="")

        self._session = tf.compat.v1.Session(graph=detection_graph)

        # Get the tensors we need.
        self._detection_graph = detection_graph
        self._model_image_tensor = detection_graph.get_tensor_by_name(
            self.model_image_tensor_name
        )
        self._model_boxes_tensor = detection_graph.get_tensor_by_name(
            self.model_boxes_tensor_name
        )
        self._model_scores_tensor = detection_graph.get_tensor_by_name(
            self.model_scores_tensor_name
        )
        self._model_classes_tensor = detection_graph.get_tensor_by_name(
            self.model_classes_tensor_name
        )

    def close(self) -> None:
        """
        Closes the model.
        """
        self._session.close()
        self._detection_graph = None
        self._session = None
        self._model_image_tensor = None
        self._model_boxes_tensor = None
        self._model_scores_tensor = None
        self._model_classes_tensor = None

    def detect(self, image_path: Union[Path, str], min_score: float = 0.1) -> list:
        """
        Detects objects in the image.

        Args:

          image_path: Path to the image to detect objects in.
          min_score: Minimum score to consider a detection.

        Returns:

          A list of tuples containing the bounding box coordinates and the class
          of the detected object.
        """

        # Load the detection graph once.
        if self._detection_graph is None:
            self.load_model()

        # Load the image.
        image_np = Image.open(image_path)
        image_np = image_np.convert("RGB")

        # Expand dimensions since the model expects images to have shape:
        # [batch_size, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)

        (box_tensor_out, score_tensor_out, class_tensor_out) = self._session.run(
            [
                self._model_boxes_tensor,
                self._model_scores_tensor,
                self._model_classes_tensor,
            ],
            feed_dict={self._model_image_tensor: image_np_expanded},
        )

        # Filter out the low scoring detections and return the bounding boxes.
        boxes = []
        for i in range(len(box_tensor_out[0])):
            if score_tensor_out[0][i] > min_score:
                boxes.append(
                    {
                        "category": int(class_tensor_out[0][i]),
                        "conf": score_tensor_out[0][i],
                        "bbox": box_tensor_out[0][i].tolist(),
                    }
                )

        return boxes

    def __enter__(self):
        self.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
