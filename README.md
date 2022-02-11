# CamTrapML

> Suite of machine learning models for the detection and classification and analysis of camera trap imagery.

## Installation

    $ pip install camtrapml

## Features

### Dataset Loader


```python
from camtrapml.dataset import Dataset
from camtrapml.image.utils import load_image, thumbnail

ena24 = Dataset(
    name="ena24",
    path="~/Datasets/ena24",
)

ena24_image_paths = list(ena24.enumerate_images())

thumbnail(load_image(ena24_image_paths[0]))
```

### Detection


```python
from camtrapml.detection.models.megadetector import MegaDetectorV4_1
from camtrapml.detection.utils import render_detections

with MegaDetectorV4_1() as detector:
    detections = detector.detect(ena24_image_paths[0])

thumbnail(
    render_detections(
        ena24_image_paths[0],
        detections,
        class_map=detector.classes
    )
)
```


