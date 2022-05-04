# CamTrapML

> CamTrapML is a Python library for Detecting, Classifying, and Analysing Wildlife [Camera Trap](https://en.wikipedia.org/wiki/Camera_trap) Imagery.

## Installation

    $ pip install camtrapml

## Features

### Loading Data

Search for images in a directory, load an image and create a thumbnail.


```python
%load_ext autoreload
%autoreload

from camtrapml.dataset import ImageDataset
from camtrapml.image.utils import load_image, thumbnail

imageset = ImageDataset(
    name="Test Images",
    path="test/fixtures/images",
)

image_paths = list(imageset.enumerate_images())

thumbnail(load_image(image_paths[0]))
```

### EXIF Extraction

EXIF extraction is a common task in gathering the metadata such as each image's timestamp, camera model, focal length, etc. Some researchers write labelling into the EXIF data. CamTrapML doesn't assist with writing to EXIF. However, there is functionality for extracting it for analysis and building datasets for training new models from previously labelled images.

ExifTool is required for this package to work. Installation instructions can be found [here](https://exiftool.org/install.html).

Three methods are available for extracting EXIF data from images. Each with different performance characteristics.

**Method 1: Individual Images**


```python
from camtrapml.image.exif import extract_exif

exif = extract_exif(image_paths[0])
exif
```

**Method 2: Multiple Images**

`extract_multiple_exif` passes a list of image paths to ExifTool and returns a list of dictionaries containing the EXIF data. This is faster than `extract_exif` when multiple images are being processed as it only passes the list of image paths to ExifTool once, rather than spawning a new process for each image.


```python
from camtrapml.image.exif import extract_multiple_exif

exif = extract_multiple_exif(image_paths)
exif[0]
```

**Method 3: Multiple Images, Multiple Processes**

When processing large datasets, it's apparent that the bottleneck in extracting the EXIF information tends to be the CPU. This method spawns multiple versions of ExifTool in parallel, each with a batch of image paths. This is faster than `extract_multiple_exif` when processing large datasets as it allows for multiple processes to be spawned and the data extracted in parallel.


```python
from camtrapml.image.exif import extract_multiple_exif_fast

exif = extract_multiple_exif_fast(image_paths)
exif[0]
```

### Detection

Various Detection models are available in the `camtrapml.detection` subpackage. These currently include MegaDetector (v4.1, v3 and v2) and support for loading in custom Tensorflow v1.x Object Detection Frozen models.

#### Detection with MegaDetector v4.1


```python
from camtrapml.detection.models.megadetector import MegaDetectorV4_1
from camtrapml.detection.utils import render_detections

with MegaDetectorV4_1() as detector:
    detections = detector.detect(image_paths[0])

thumbnail(
    render_detections(image_paths[0], detections, class_map=detector.class_map)
)
```

#### Detection with a custom Tensorflow v1.x Object Detection Frozen model


```python
!cp ~/.camtrapml/models/megadetector/v4.1.0/md_v4.1.0.pb example-custom-model.pb

from camtrapml.detection.models.tensorflow import TF1ODAPIFrozenModel
from camtrapml.detection.utils import render_detections
from pathlib import Path

with TF1ODAPIFrozenModel(
    model_path=Path("example-custom-model.pb").expanduser(),
    class_map={
        1: "animal",
    },
) as detector:
    detections = detector.detect(image_paths[1])

thumbnail(
    render_detections(image_paths[1], detections, class_map=detector.class_map)
)
```

#### Extract Detections


```python
from camtrapml.detection.models.megadetector import MegaDetectorV4_1
from camtrapml.detection.utils import extract_detections_from_image

with MegaDetectorV4_1() as detector:
    detections = detector.detect(image_paths[0])

list(extract_detections_from_image(load_image(image_paths[0]), detections))[0]
```

#### Remove Humans from Images

In order to reduce the risks of identification of humans in line with GDPR, CamTrapML provides the ability to remove humans from images. This is achieved by using the MegaDetector v3+ models to detect humans in the image, and then replacing all pixels in each human detection.


```python
from camtrapml.detection.models.megadetector import MegaDetectorV4_1
from camtrapml.detection.utils import remove_detections_from_image
from camtrapml.image.utils import load_image, thumbnail
from pathlib import Path

ct_image_with_humans = Path("test/fixtures/human_images/IMG_0254.JPG").expanduser()

with MegaDetectorV4_1() as detector:
    detections = detector.detect(ct_image_with_humans)

human_class_id = 2

thumbnail(
    remove_detections_from_image(
        load_image(ct_image_with_humans),
        [
            detection
            for detection in detections
            if detection["category"] == human_class_id and detection["conf"] > 0.5
        ],
    )
)
```


```python

```
