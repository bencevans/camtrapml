import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

from PIL import Image

class Inat2017InceptionV3:
    def predict(self, image: Image):
        image = image.copy()
        image = image.resize((299, 299))
        image = np.asarray(image, dtype=np.float32) / 255.0
        image = np.expand_dims(image, axis=0)
        return self.model(image)[0].numpy()

    def predict_batch(self, images: list):
        images = [image.copy() for image in images]
        images = [image.resize((299, 299)) for image in images]
        images = np.asarray([np.asarray(image, dtype=np.float32) / 255.0 for image in images], dtype=np.float32)
        return self.model(images)

    def __enter__(self):
        self.model = hub.load('https://tfhub.dev/google/inaturalist/inception_v3/feature_vector/5')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

  
