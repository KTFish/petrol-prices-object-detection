# petrol-prices-object-detection

The aim of this project realized by members of Artificial Intelligence Society ,,Czarna Magia" is to detect petrol price tags.

### What is the base line idea?

1. The app gets a picture of a gas station.
2. It detects the price tag (all ML and math work here)
3. Then OCR is performed to get the numbers out of the detected price tag
4. We compare whether the price is the lowest in e.g. a month, two weeks, etc. and then the inscription appears whether it is worth buying (in future more advanced methods can be implemented).

To detect the pirce tags we use YOLOv8 architesture from the [`ultralytics`](https://github.com/ultralytics/ultralytics) library implementation.

### Labeling the data

For labeling our custom dataset we used different labeling software:

- [cvat](https://app.cvat.ai/)
- [roboflow](https://universe.roboflow.com/)

### Contents

- `Demo.ipynb` notebook contains a short demo on how to use ultralytics YOLOv8 model in practice.
- `scripts` folder contains helpful code splited into modules.
- `config.yaml` is a file used for YOLO configuration. To use it, its necessary to add a absolute path to the root directory. Below the paths we need to specify the detected classes. In the current version of our model there are 3 classes: `circlek` (Circle K's gas station logo), `orlen` (Orlen's gas station logo) and `prices` (detects the price tag).
