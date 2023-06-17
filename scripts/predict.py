import ultralytics
from ultralytics import YOLO
from typing import List
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2


def predict(
    image, model, verbose: bool = False
) -> List[ultralytics.yolo.engine.results.Results]:
    """Makes predictions with YOLO and outputs the results.
    #TODO: Better description
    """

    # Transform the image (resize to 640 x 640)
    # TODO: Do I even need a `transforms`?
    # transform = transforms.Compose([transforms.Resize(size=(640, 640))])
    # image = transform(image)

    # Inference
    results = model(image)
    result = results[0]

    # Get all detected classes and turn them from float tensors to a lits of ints
    detected_classes = [int(x.item()) for x in result.boxes.cls]

    # Print information about the detected classes
    if verbose:
        print(f"Detected classes: {detected_classes}")

    return result, detected_classes


def explore_results(results, verbose: bool = True):
    result = results[0]

    # Get all detected classes and turn them from float tensors to a lits of ints
    detected_classes = [int(x.item()) for x in result.boxes.cls]

    # Print information about the detected classes
    if verbose:
        print(f"Detected classes: {detected_classes}")
        plt.imshow(result.boxes)


path_to_model = rf"C:\Users\janek\notebooks\petrol-prices-object-detection\runs\detect\train4\weights\best.pt"
model = YOLO(path_to_model)  # Load model
img_path = r"C:\Users\janek\notebooks\petrol-prices-object-detection\custom_data\orlen-ultimate-photo.jpg"


def get_cropped_bboxes(result, verbose: bool = True) -> np.ndarray:
    # Get bounding boxes
    boxes = result.boxes.xyxy.cpu().numpy()

    cropped_boxes = []
    for box in boxes:
        # Transform box
        x1, y1, x2, y2 = [int(k) for k in box]

        # Cropp
        cropped = result.orig_img[y1:y2, x1:x2]

        # Add to list of results
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)  # Translate to RBG
        cropped_boxes.append(cropped)

    if verbose:
        fig, ax = plt.subplots(nrows=len(cropped_boxes), ncols=1)
        for k in range(len(cropped_boxes)):
            ax[k].imshow(cropped_boxes[k])
            ax[k].axis("off")

    return cropped_boxes


img = r"C:\Users\janek\notebooks\petrol-prices-object-detection\custom_data/roboflow/dataset-v2/train/images/IMG_20230606_194813_jpg.rf.bf34865c4fccd8accfa7d9080b9e3f3b.jpg"
results = predict(img, model)
result = results[0]

cropped_boxes = get_cropped_bboxes(result)
