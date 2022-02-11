import platform
from pathlib import Path
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import numpy as np


# Disable the GPU on M1 Macs as there appears to be a bug where the outputs are
# not the same as the reference implementation.
if platform.system() == "Darwin" and platform.machine() == "arm64":
    tf.config.set_visible_devices([], "GPU")


class MegaDetectorV4_1:
    """
    Animal, Human and Vehicle Object Detection Model

    https://github.com/Microsoft/CameraTraps/
    """

    classes = {1: "animal", 2: "human", 3: "vehicle"}

    def __init__(self, model_path=Path("./models/megadetector/md_v4.1.0.pb")) -> None:
        """
        Initializes the MegaDetector model.

        Args:

          model_path: Path to the frozen model .pb file.
        """
        self.model_path = model_path
        self.detection_graph = None
        self.image_tensor = None
        self.box_tensor = None
        self.score_tensor = None
        self.classes_tensor = None

    def load_model(self) -> None:
        """
        Loads the model from the frozen model .pb file.
        """

        detection_graph = tf.compat.v1.Graph()

        with detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.compat.v1.gfile.GFile(str(self.model_path), "rb") as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name="")

        self.session = tf.compat.v1.Session(graph=detection_graph)

        # Get the tensors we need.
        self.detection_graph = detection_graph
        self.image_tensor = detection_graph.get_tensor_by_name("image_tensor:0")
        self.box_tensor = detection_graph.get_tensor_by_name("detection_boxes:0")
        self.score_tensor = detection_graph.get_tensor_by_name("detection_scores:0")
        self.classes_tensor = detection_graph.get_tensor_by_name("detection_classes:0")

    def close(self) -> None:
        """
        Closes the model.
        """
        self.session.close()
        self.detection_graph = None
        self.session = None
        self.image_tensor = None
        self.box_tensor = None
        self.score_tensor = None
        self.classes_tensor = None

    def detect(self, image_path: Path, min_score=0.1) -> list:
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
        if self.detection_graph is None:
            self.load_model()

        # Load the image.
        image_np = Image.open(image_path)
        image_np = image_np.convert("RGB")

        sess = self.session

        # Expand dimensions since the model expects images to have shape:
        # [batch_size, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)

        (box_tensor_out, score_tensor_out, class_tensor_out) = sess.run(
            [self.box_tensor, self.score_tensor, self.classes_tensor],
            feed_dict={self.image_tensor: image_np_expanded},
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
