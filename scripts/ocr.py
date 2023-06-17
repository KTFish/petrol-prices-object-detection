import cv2
import matplotlib.pyplot as plt
import numpy as np
from scripts.predict import predict


# Image --> Detected Price Tag --> Cropped Bounding Box
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
