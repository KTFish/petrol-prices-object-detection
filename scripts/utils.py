import cv2
import os
import matplotlib.pyplot as plt
from typing import List
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.engine.model import YOLO
from math import ceil, sqrt


def get_size(model_path: str, verbose: bool = False) -> float:
    """Returns the size of the model in MB.

    Args:
        model_path (str): Paht to YOLO model.
        verbose (bool, optional): If set to True prints out information about the model size. Defaults to False.

    Returns:
        float: Model size.
    """

    size_mb = os.path.getsize(model_path) / 1024**2  # bytes to megabytes
    if verbose:
        print(f"Size of the model: ~{size_mb:.2f} MB.")
    return size_mb


def plot_results(model: YOLO, image: str) -> None:
    """Takes results and the model and plots the results.

    Args:
        image (str): Path to image.
        model (YOLO): Model which results will be plotted.
    """
    results = model(image)
    for res in results:
        img = res.orig_img

        # res = model(img)
        res_plotted = res[0].plot()
        plt.imshow(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB))
        plt.axis(False)
        plt.show()


def plot_results_one_fig(model: YOLO, dir: str) -> None:
    """Plots detected bounding boxes on one figure for all images from the given directory.

    Args:
        model (YOLO): Model used for inference.
        dir (str): Directory with images.
    """
    # Get list of image names to plot
    images = os.listdir(dir)

    # Calculate the shape of plotting figure
    n = ceil(sqrt(len(images)))
    fig, ax = plt.subplots(nrows=n, ncols=n, figsize=(10, 10))

    # Plotting images with bounding boxes
    for i, img in enumerate(images):
        img_path = dir + rf"/{img}"  # Get path to the image
        res = model(img_path)  # Get the models predictions
        res_plotted = res[0].plot()

        ax[i // n, i % n].imshow(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB))
        ax[i // n, i % n].axis("off")

    # Turn off the axis for remaining subplots
    for j in range(i + 1, n * n):
        ax[j // n, j % n].axis("off")

    plt.show()


def load_model_run(run_id: int) -> YOLO:
    """Takes a run id and returns the best model version trained in that run.

    Args:
        run_id (int): id number of the run, used to get the path to the model.

    Returns:
        YOLO: Returns a loaded ultralytics YOLOv8 model.
    """
    path_to_trained_model = rf"runs\detect\train{run_id}\weights\best.pt"
    model = YOLO(path_to_trained_model)
    return model


# TODO: Function to check if every image has a label (os.listdir can be helpfull)


def save_image(image_array, file_path):
    # Convert the image array to BGR format for OpenCV
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    # Save the image as a .jpg file
    cv2.imwrite(file_path, image_bgr)
