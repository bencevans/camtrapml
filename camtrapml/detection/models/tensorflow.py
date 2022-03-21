"""
Tensorflow Detection Model Base Classes
"""

from platform import system, machine
from warnings import warn
from typing import Union
from pathlib import Path

import numpy as np
import tensorflow as tf

from camtrapml.image.utils import ImageSource, load_image
from camtrapml.download import download, hash_file


class TF1ODAPIFrozenModel:
    """
    Tensorflow v1 (loaded using v2) Object Detection API Frozen Model Loader
    """

    class_map = None

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

        # Disable the GPU on M1 Macs as there appears to be a bug where the outputs are
        # not the same as the reference implementation.
        if system() == "Darwin" and machine() == "arm64":
            warn("Disabling GPU on M1 Macs")
            tf.config.set_visible_devices([], "GPU")

        detection_graph = tf.compat.v1.Graph()

        with detection_graph.as_default():  # pylint: disable=E1129
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.compat.v1.gfile.GFile(str(self.model_path), "rb") as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name="")

        self._session = tf.compat.v1.Session(graph=detection_graph)

        # Get the tensors we need.
        self._tensors = {
            'image': detection_graph.get_tensor_by_name(
                "image_tensor:0"
            ),
            'boxes': detection_graph.get_tensor_by_name(
                "detection_boxes:0"
            ),
            'scores': detection_graph.get_tensor_by_name(
                "detection_scores:0"
            ),
            'classes': detection_graph.get_tensor_by_name(
                "detection_classes:0"
            )
        }

    def close(self) -> None:
        """
        Closes the model.
        """
        if self._session:
            self._session.close()

        self._session = None
        self._tensors = None

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
        if self._session is None:
            self.load_model()

        # Load the image.
        image_np = load_image(image_source, mode="RGB")

        # Expand dimensions since the model expects images to have shape:
        # [batch_size, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)

        (box_tensor_out, score_tensor_out, class_tensor_out) = self._session.run(
            [
                self._tensors['boxes'],
                self._tensors['scores'],
                self._tensors['classes'],
            ],
            feed_dict={self._tensors['image']: image_np_expanded},
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
