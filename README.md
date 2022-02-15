# CamTrapML

> Suite of machine learning models for the detection and classification and analysis of camera trap imagery.

## Installation

    $ pip install camtrapml

## Features

### Loading Data

Search for images in a directory, load an image and create a thumbnail.


```python
from camtrapml.dataset import Dataset
from camtrapml.image.utils import load_image, thumbnail

ena24 = Dataset(
    name="ena24",
    path="~/Datasets/ena24/ena24",
)

ena24_image_paths = list(ena24.enumerate_images())

thumbnail(load_image(ena24_image_paths[0]))
```

### EXIF Extraction

Exif extraction is a common task in gathering the metadata such as each images timestamp, camera model, focal length, etc. as well as some researchers write labelling into the EXIF data. Although this package doesn't assist with writing to EXIF, it can be used to extract if for the purposes of analysis and building datasets for training from previously labelled images.

ExifTool is required for this package to work. Installation instructions can be found [here](https://exiftool.org/install.html).

Three methods are provided for extracting EXIF data from images. Each with different performance characteristics.

**Method 1: Individual Images**


```python
from camtrapml.image.exif import extract_exif

exif = extract_exif(ena24_image_paths[0])
exif
```

**Method 2: Multiple Images**

`extract_multiple_exif` passes a list of image paths to ExifTool and returns a list of dictionaries containing the EXIF data. This is faster than `extract_exif` when multiple images are being processed as it only passes the list of image paths to ExifTool once, rather than spawning a new process for each image.


```python
from camtrapml.image.exif import extract_multiple_exif

exif = extract_multiple_exif(ena24_image_paths)
exif[0]
```

**Method 3: Multiple Images, Multiple Processes**

When processing large datasets it's become apparent that the bottleneck in extracting the EXIF information tends to be CPU. This method, spawns multiple versions of ExifTool in parallel, each with a batch of image paths. This is faster than `extract_multiple_exif` when processing large datasets as it allows for multiple processes to be spawned and the data extracted in parallel.


```python
from camtrapml.image.exif import extract_multiple_exif_fast

exif = extract_multiple_exif_fast(ena24_image_paths)
exif[0]
```

### Detection

Various Detection models have been packaged up in the camtrapml.detection subpackage. These currently include MegaDetector (v4.1, v3 and v2) and support for loading in custom Tensorflow v1.x Object Detection Frozen models.

#### Detection with MegaDetector v4.1


```python
from camtrapml.detection.models.megadetector import MegaDetectorV4_1
from camtrapml.detection.utils import render_detections

with MegaDetectorV4_1() as detector:
    detections = detector.detect(ena24_image_paths[0])

thumbnail(
    render_detections(
        ena24_image_paths[0],
        detections,
        class_map=detector.class_map
    )
)
```

#### Detection with a custom Tensorflow v1.x Object Detection Frozen model


```python
from camtrapml.detection.models.tensorflow import TF1ODAPIFrozenModel
from camtrapml.detection.utils import render_detections
from pathlib import Path

with TF1ODAPIFrozenModel(
    model_path=Path('~/Downloads/my-custom-model.pb').expanduser(),
    model_image_tensor_name='image_tensor:0',
    model_boxes_tensor_name='detection_boxes:0',
    model_scores_tensor_name='detection_scores:0',
    model_classes_tensor_name='detection_classes:0',
    class_map={
        1: "animal",
    },
) as detector:
    detections = detector.detect(ena24_image_paths[1])

thumbnail(
    render_detections(
        ena24_image_paths[1],
        detections,
        class_map=detector.class_map
    )
)
```


```python

```
