import cv2
import numpy as np
import pytesseract
import ultralytics
import matplotlib.pyplot as plt
from typing import Dict, List
from scripts.predict import predict
from collections import defaultdict


def get_cropped_bboxes(
    result: ultralytics.yolo.engine.results.Results, verbose: bool = False
) -> Dict[int, List[np.ndarray]]:
    """Returns a dictionary that maps class idx to cropped bounding boxes of that class based on the ultralytics results.

    Args:
        result (ultralytics.yolo.engine.results.Results): Result of YOLOv8 inference.
        verbose (bool, optional): Plots cropped bounding boxes. Defaults to False.

    Returns:
        Dict[int, List[np.ndarray]]: Dictionary mapping class indices to a list of cropped bounding boxes.
    """
    # Get bounding boxes
    boxes = result.boxes.xyxy.cpu().numpy()
    class_indices = [int(x.item()) for x in result.boxes.cls]

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

    # Map class indices to boudning boxes
    class_name_to_box = defaultdict(list)
    for name, box in zip(class_indices, cropped_boxes):
        class_name_to_box[name].append(box)

    return class_name_to_box


def increase_contrast(grayscale_image):
    # Apply adaptive thresholding to enhance contrast
    _, thresholded_image = cv2.threshold(
        grayscale_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )

    # Apply bitwise_not to invert the image
    inverted_image = cv2.bitwise_not(thresholded_image)

    return inverted_image


def perform_ocr(image_array: np.array) -> str:
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )
    # image = cv2.imread(image_path)
    image = cv2.cvtColor(
        image_array, cv2.COLOR_RGB2BGR
    )  # Convert RGB to BGR for OpenCV

    # Turn image into grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Increase contrast
    image = increase_contrast(grayscale_image)

    # Apply denoising using the Bilateral Filter
    denoised_image = cv2.bilateralFilter(grayscale_image, 5, 75, 75)

    # Apply smoothing using Gaussian Blur
    # smoothed_image = cv2.GaussianBlur(denoised_image, (5, 5), 0)

    plt.imshow(image, cmap="gray")
    plt.axis(False)
    plt.show()

    result = pytesseract.image_to_string(grayscale_image, lang="eng")
    return result
